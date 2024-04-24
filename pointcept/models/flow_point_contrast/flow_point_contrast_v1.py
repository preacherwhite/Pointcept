import random
from itertools import chain
import torch
import torch.nn as nn
import torch.distributed as dist
from torch_geometric.nn.pool import voxel_grid
from timm.models.layers import trunc_normal_
import pointops
from pointcept.models.builder import MODELS, build_model
from pointcept.models.utils import offset2batch
from pointcept.utils.comm import get_world_size
import numpy as np

import sys

def generate_positive_and_negative_masks(similarity_matrix, threshold, file=sys.stdout):
    """Generates positive and negative masks used by contrastive loss."""
    positive_mask = (similarity_matrix >= threshold).float()
    negative_mask = 1 - positive_mask

    # print(f"Positive Mask:\n{positive_mask}", file=file)
    # print(f"Negative Mask:\n{negative_mask}", file=file)

    return positive_mask, negative_mask

def compute_contrastive_loss(logits_list, positive_mask_list, negative_mask_list, file=sys.stdout):
    """Contrastive loss function."""
    batch_size = len(logits_list)
    losses = []

    for i in range(batch_size):
        logits = logits_list[i]
        positive_mask = positive_mask_list[i]
        negative_mask = negative_mask_list[i]

        exp_logits = torch.exp(logits)
        #print(f"Exp Logits ({i}):\n{exp_logits}", file=file)

        normalized_exp_logits = exp_logits / (exp_logits + torch.sum(exp_logits * negative_mask, dim=1, keepdim=True))
        #print(f"Normalized Exp Logits ({i}):\n{normalized_exp_logits}", file=file)

        neg_log_likelihood = -torch.log(normalized_exp_logits)
        #print(f"Negative Log Likelihood ({i}):\n{neg_log_likelihood}", file=file)

        normalized_weight = positive_mask / torch.clamp(torch.sum(positive_mask, dim=1, keepdim=True), min=1e-6)
        #print(f"Normalized Weight ({i}):\n{normalized_weight}", file=file)

        neg_log_likelihood = torch.sum(neg_log_likelihood * normalized_weight, dim=1)
        #print(f"Weighted Negative Log Likelihood ({i}):\n{neg_log_likelihood}", file=file)

        positive_mask_sum = torch.sum(positive_mask, dim=1)
        #print(f"Positive Mask Sum ({i}):\n{positive_mask_sum}", file=file)

        valid_index = 1 - (positive_mask_sum == 0).float()
        #print(f"Valid Index ({i}):\n{valid_index}", file=file)

        normalized_weight = valid_index / torch.clamp(torch.sum(valid_index), min=1e-6)
        #print(f"Normalized Valid Weight ({i}):\n{normalized_weight}", file=file)

        neg_log_likelihood = torch.sum(neg_log_likelihood * normalized_weight)
        #print(f"Final Negative Log Likelihood ({i}):\n{neg_log_likelihood}", file=file)

        losses.append(neg_log_likelihood)

    loss = torch.mean(torch.stack(losses))
    #print(f"Mean Loss: {loss}", file=file)

    if get_world_size() > 1:
        dist.all_reduce(loss)

    return loss / get_world_size()

def within_sample_contrastive_loss(features_list, similarity_matrix_list, threshold, temperature=0.07, file=sys.stdout):
    """Computes within-sample contrastive loss."""
    batch_size = len(features_list)
    logits_list = []
    positive_mask_list = []
    negative_mask_list = []

    for i in range(batch_size):
        features = features_list[i]
        similarity_matrix = similarity_matrix_list[i]

        # print(f"Features ({i}):\n{features}", file=file)
        # print(f"Similarity Matrix ({i}):\n{similarity_matrix}", file=file)
        normalized_features = features / (
            torch.norm(features, p=2, dim=1, keepdim=True) + 1e-7
        )
        logits = torch.matmul(normalized_features, normalized_features.transpose(0, 1)) / temperature
        #print(f"Logits ({i}):\n{logits}", file=file)

        positive_mask, negative_mask = generate_positive_and_negative_masks(similarity_matrix, threshold, file)

        logits_list.append(logits)
        positive_mask_list.append(positive_mask)
        negative_mask_list.append(negative_mask)

    return compute_contrastive_loss(logits_list, positive_mask_list, negative_mask_list, file)

@MODELS.register_module("FPC-v1")
class FlowPointContrast(nn.Module):
    def __init__(
        self,
        backbone,
        flow_similarity_threshold=0.8,
        color_similarity_threshold=0.7,
        proximity_threshold=0.5,
        flow_weight=1.0,
        color_weight=1.0,
        proximity_weight=1.0,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.flow_similarity_threshold = flow_similarity_threshold
        self.color_similarity_threshold = color_similarity_threshold
        self.proximity_threshold = proximity_threshold
        self.flow_weight = flow_weight
        self.color_weight = color_weight
        self.proximity_weight = proximity_weight

    def calculate_flow_similarity(self, flow):
        flow_norm = torch.norm(flow, dim=-1)
        flow_similarity = torch.matmul(flow, flow.transpose(0, 1)) / (
            flow_norm.unsqueeze(1) * flow_norm.unsqueeze(0) + 1e-8
        )
        return flow_similarity

    def calculate_color_similarity(self, colors):
        colors = colors.to(torch.float32)
        color_similarity = 1 - torch.cdist(colors, colors) / torch.sqrt(torch.tensor(colors.shape[-1]))
        return color_similarity

    def calculate_proximity_similarity(self, points, sigma=0.1):
        distances = torch.cdist(points, points)
        proximity_similarity = torch.exp(-distances / (2 * sigma ** 2))
        return proximity_similarity

    def forward(self, data_dict):
        flow = data_dict["flow"]
        colors = data_dict["color"]
        points = data_dict["coord"]

        # subset_keys = ['coord', 'grid_coord', 'offset', 'feat']
        # data_dict = {key: data_dict[key] for key in subset_keys if key in data_dict}
        features = self.backbone(data_dict).feat
        if isinstance(features, dict):
            features = features.dict
        with open("contrastive_loss_output.txt", "w") as file:
            #print('start_checking_nan')
            # print(f"points: {points}", file=file)
            # print(f"flow: {flow}", file=file)
            # print(f"colors: {colors}", file=file)
            # print(f"grid_coord: {data_dict['grid_coord']}", file=file)
            # print(f"offset: {data_dict['offset']}", file=file)
            # print(f"pre_features: {data_dict['feat']}", file=file)
            # Split flow, colors, and points based on the offset


            # print(f"features: {features}", file=file)
            offset = data_dict['offset']
            batch_size = len(offset)

            offset = torch.cat([torch.Tensor([0]).to(features.device),data_dict['offset']])
            offset = torch.round(offset).int()
            sizes = (offset[1:] - offset[:-1]).tolist()
            flow_split = torch.split(flow, sizes)
            colors_split = torch.split(colors, sizes)
            points_split = torch.split(points, sizes)
            features_split = torch.split(features, sizes)

            flow_similarity_list = []
            color_similarity_list = []
            proximity_similarity_list = []

            for i in range(batch_size):
                flow_similarity = self.calculate_flow_similarity(flow_split[i])
                color_similarity = self.calculate_color_similarity(colors_split[i])
                proximity_similarity = self.calculate_proximity_similarity(points_split[i])

                flow_similarity_list.append(flow_similarity)
                color_similarity_list.append(color_similarity)
                proximity_similarity_list.append(proximity_similarity)

            
            print(f"features_split: {features_split[0]}", file=file)
            flow_loss = within_sample_contrastive_loss(
                features_split, flow_similarity_list, self.flow_similarity_threshold, file=file
            )
            color_loss = within_sample_contrastive_loss(
                features_split, color_similarity_list, self.color_similarity_threshold,file = file
            )
            proximity_loss = within_sample_contrastive_loss(
                features_split, proximity_similarity_list, self.proximity_threshold,file = file
            )
        loss = (
            self.flow_weight * flow_loss
            + self.color_weight * color_loss
            + self.proximity_weight * proximity_loss
        )
        result_dict = {"loss": loss}
        return result_dict