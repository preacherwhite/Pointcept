import random
from itertools import chain
import torch
import torch.nn as nn

from torch_geometric.nn.pool import voxel_grid
from timm.models.layers import trunc_normal_
from segment_anything import build_sam, SamAutomaticMaskGenerator
from pointcept.models.builder import MODELS, build_model
from pointcept.models.utils import offset2batch
import numpy as np

import sys
import torch
import numpy as np
from pointcept.utils.comm import get_world_size
import torch.distributed as dist


def generate_positive_and_negative_masks(labels):
    """Generates positive and negative masks used by contrastive loss."""
    # Reshape to column vector and row vector
    tensor_col = labels.view(-1, 1)  # shape (7457, 1)
    tensor_row = labels.view(1, -1)  # shape (1, 7457)

    # Compare using torch.eq
    positive_mask = torch.eq(tensor_col, tensor_row).float()
    negative_mask = 1 - positive_mask
    return positive_mask, negative_mask

def hard_compute_contrastive_loss(logits_list, mask):
    """Contrastive loss function."""
    batch_size = len(logits_list)
    losses = []

    for i in range(batch_size):
        logits = logits_list[i]
        positive_mask, negative_mask = generate_positive_and_negative_masks(mask[i])

        exp_logits = torch.exp(logits)

        normalized_exp_logits = exp_logits / (exp_logits + torch.sum(exp_logits * negative_mask, dim=1, keepdim=True))

        neg_log_likelihood = -torch.log(normalized_exp_logits)

        normalized_weight = positive_mask / torch.clamp(torch.sum(positive_mask, dim=1, keepdim=True), min=1e-6)

        neg_log_likelihood = torch.sum(neg_log_likelihood * normalized_weight, dim=1)

        positive_mask_sum = torch.sum(positive_mask, dim=1)

        valid_index = 1 - (positive_mask_sum == 0).float()

        normalized_weight = valid_index / torch.clamp(torch.sum(valid_index), min=1e-6)

        neg_log_likelihood = torch.sum(neg_log_likelihood * normalized_weight)

        losses.append(neg_log_likelihood)

    loss = torch.mean(torch.stack(losses))

    if get_world_size() > 1:
        dist.all_reduce(loss)

    return loss / get_world_size()

def masked_contrastive_loss(features_list, mask, temperature=0.07):
    """Computes within-sample contrastive loss."""
    batch_size = len(features_list)
    logits_list = []

    for i in range(batch_size):
        features = features_list[i]

        normalized_features = features / (
            torch.norm(features, p=2, dim=1, keepdim=True) + 1e-7
        )
        logits = torch.matmul(normalized_features, normalized_features.transpose(0, 1)) / temperature

        logits_list.append(logits)

    return hard_compute_contrastive_loss(logits_list, mask)




def within_sample_contrastive_loss(features_list, sim_scores_list, tau, gamma, eta, nu):
    """Computes within-sample contrastive loss."""
    batch_size = len(features_list)
    normalized_features_list = []

    for i in range(batch_size):
        features = features_list[i]
        normalized_features = features / (torch.norm(features, p=2, dim=1, keepdim=True) + 1e-7)
        normalized_features_list.append(normalized_features)

    return compute_contrastive_loss(normalized_features_list, sim_scores_list, tau, gamma, eta, nu)

def compute_contrastive_loss(features_list, sim_scores_list, tau, gamma, eta, nu, score_higher_positive = True):
    """Contrastive loss function."""
    batch_size = len(features_list)
    losses = []

    for i in range(batch_size):
        features = features_list[i]
        sim_scores = sim_scores_list[i]

        # Calculate g_plus and g_minus
        g_plus = 1 / (1 + torch.exp(gamma * (tau - sim_scores)))
        g_minus = 1 / (1 + torch.exp(gamma * (sim_scores - tau)))
        
        # Calculate all-pairs similarity of features
        feature_similarity = torch.matmul(features, features.transpose(0, 1))
        
        # Calculate s_a_plus and s_a_minus using weighted sum
        if score_higher_positive:
            s_a_plus = feature_similarity * g_plus
            s_a_minus = feature_similarity * g_minus
        else:
            s_a_plus = feature_similarity * g_minus
            s_a_minus = feature_similarity * g_plus
        s_a_plus_exp = torch.sum(torch.exp(eta * s_a_plus), dim=1)
        s_a_minus_exp = torch.sum(torch.exp(-nu * s_a_minus), dim=1)


        # Calculate the loss for each point
        loss_per_point = (torch.log(1 + s_a_plus_exp) / eta) + \
                         (torch.log(1 + s_a_minus_exp) / nu)
        # Sum the loss over all points
        loss = torch.mean(loss_per_point)
        losses.append(loss)

    # Because the features similarity is being weighted, this needs to be negated
    loss = -torch.mean(torch.stack(losses))
    #print(loss)
    if get_world_size() > 1:
        dist.all_reduce(loss)
    return loss / get_world_size()


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
        sam_weight = 1.0,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.flow_similarity_threshold = flow_similarity_threshold
        self.color_similarity_threshold = color_similarity_threshold
        self.proximity_threshold = proximity_threshold
        self.flow_weight = flow_weight
        self.color_weight = color_weight
        self.proximity_weight = proximity_weight
        self.sam_weight = sam_weight

    def calculate_flow_similarity(self, flow):
        flow_norm = torch.norm(flow, dim=-1)
        #nomalized flow similarity, should be between -1 and 1
        flow_similarity = torch.matmul(flow, flow.transpose(0, 1)) / (
            flow_norm.unsqueeze(1) * flow_norm.unsqueeze(0) + 1e-8
        )
        return flow_similarity

    def calculate_color_similarity(self, colors):
        # Normalize colors to be between 0 and 1
        colors = colors.to(torch.float32) / 255.0
        
        # Compute the pairwise Euclidean distances between normalized color vectors
        distances = torch.cdist(colors, colors)
        
        # Normalize distances by the maximum possible distance in the normalized space
        max_distance = torch.sqrt(torch.tensor(colors.shape[-1]))  # sqrt(d) where d is the dimension of the color space
        normalized_distances = distances / max_distance
        
        # Compute similarity as 1 minus the normalized distances
        color_similarity = 1 - normalized_distances
        
        return color_similarity


    def calculate_proximity_similarity(self, points, sigma=0.1):
        distances = torch.cdist(points, points)
        proximity_similarity = torch.exp(-distances / (2 * sigma ** 2))
        return proximity_similarity

    def forward(self, data_dict):
        

        if not self.training:
            features = self.backbone(data_dict).feat
            if isinstance(features, dict):
                features = features.dict
            return features


        sam_label = data_dict["sam"]
        flow = data_dict["flow"]
        colors = data_dict["color"]
        points = data_dict["coord"]
        
        # subset_keys = ['coord', 'grid_coord', 'offset', 'feat']
        # data_dict = {key: data_dict[key] for key in subset_keys if key in data_dict}
        features = self.backbone(data_dict).feat
        if isinstance(features, dict):
            features = features.dict
        offset = data_dict['offset']
        batch_size = len(offset)

        offset = torch.cat([torch.Tensor([0]).to(features.device),data_dict['offset']])
        offset = torch.round(offset).int()
        sizes = (offset[1:] - offset[:-1]).tolist()
        flow_split = torch.split(flow, sizes)
        colors_split = torch.split(colors, sizes)
        points_split = torch.split(points, sizes)
        features_split = torch.split(features, sizes)
        sam_split = torch.split(sam_label, sizes)

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

        flow_loss = within_sample_contrastive_loss(
            features_split, flow_similarity_list, self.flow_similarity_threshold, gamma = 5.0, eta =1, nu = 1
        )

        color_loss = within_sample_contrastive_loss(
            features_split, color_similarity_list, self.color_similarity_threshold, gamma = 5.0, eta =1, nu = 1
        )

        proximity_loss = within_sample_contrastive_loss(
            features_split, proximity_similarity_list, self.proximity_threshold, gamma = 5.0, eta =1, nu = 1
        )
        #print(len(sam_split),sam_split[0].shape)
        sam_loss = masked_contrastive_loss(features_split, sam_split, temperature=0.07) 
        loss =  self.color_weight * color_loss + \
                self.flow_weight * flow_loss + \
                self.proximity_weight * proximity_loss + \
                self.sam_weight* sam_loss
    
        # print(flow_loss.item(), self.flow_weight, color_loss.item(), self.color_weight,proximity_loss.item(),self.proximity_weight,sam_loss.item(),self.sam_weight)
        # print(loss)
        result_dict = {"loss": loss}
        return result_dict