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

def generate_positive_and_negative_masks(similarity_matrix, threshold):
    """Generates positive and negative masks used by contrastive loss."""
    positive_mask = (similarity_matrix >= threshold).float()
    negative_mask = 1 - positive_mask
    return positive_mask, negative_mask


def compute_contrastive_loss_without_ignore(logits, positive_mask, negative_mask):
    """Contrastive loss function without an ignore mask."""
    exp_logits = torch.exp(logits)
    exp_sum = torch.sum(exp_logits * negative_mask, dim=2, keepdim=True)

    # Avoid division by zero
    exp_sum = torch.clamp(exp_sum, min=1e-6)

    # Compute normalized exponential logits
    normalized_exp_logits = exp_logits / (exp_logits + exp_sum)

    # Compute negative log likelihood
    neg_log_likelihood = -torch.log(normalized_exp_logits + 1e-6)  # Adding epsilon for numerical stability

    # Weighting negative log likelihood by the positive mask
    print(neg_log_likelihood.shape)
    weighted_neg_log_likelihood = neg_log_likelihood * positive_mask
    loss = torch.sum(weighted_neg_log_likelihood) / torch.sum(positive_mask)  # Normalizing by the number of positives

    if get_world_size() > 1:
        dist.all_reduce(loss)
    return loss / get_world_size()
    
def within_sample_contrastive_loss(features, similarity_matrix, threshold, temperature=0.07):
    """Computes within-sample contrastive loss."""
    logits = torch.matmul(features, features.transpose(1, 2)) 
    logits = logits / torch.Tensor(temperature).to(features.device)
    positive_mask, negative_mask = generate_positive_and_negative_masks(similarity_matrix, threshold)
    return compute_contrastive_loss_without_ignore(logits, positive_mask, negative_mask)

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

    # def calculate_similarity(self, vectors, similarity_func):
    #     batch_size = len(vectors)
    #     similarity_matrices = []

    #     for i in range(batch_size):
    #         similarity_matrix = similarity_func(vectors[i])
    #         similarity_matrices.append(similarity_matrix)

    #     return torch.stack(similarity_matrices)
    
    def calculate_flow_similarity(self, flow):
        flow_norm = torch.norm(flow, dim=-1)
        flow_similarity = torch.matmul(flow, flow.transpose(1, 2)) / (
            flow_norm.unsqueeze(2) * flow_norm.unsqueeze(1) + 1e-8
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
        batch_num = len(data_dict['offset'])
        flow = data_dict["flow"].view(batch_num, -1, data_dict["flow"].shape[1])
        colors = data_dict["color"].view(batch_num, -1, data_dict["color"].shape[1])
        points = data_dict["coord"].view(batch_num, -1, data_dict["coord"].shape[1])

        subset_keys = ['coord', 'grid_coord', 'offset', 'feat']
        data_dict = {key: data_dict[key] for key in subset_keys if key in data_dict}
        features = self.backbone(data_dict).feat

        features = features.view(batch_num, -1, features.shape[1])

        flow_similarity = self.calculate_flow_similarity(flow)
        color_similarity = self.calculate_color_similarity(colors)
        proximity_similarity = self.calculate_proximity_similarity(points)

        ignore_labels = [-1]  # Ignore labels for contrastive loss calculation

        flow_loss = within_sample_contrastive_loss(
            features, flow_similarity, self.flow_similarity_threshold, ignore_labels
        )
        color_loss = within_sample_contrastive_loss(
            features, color_similarity, self.color_similarity_threshold, ignore_labels
        )
        proximity_loss = within_sample_contrastive_loss(
            features, proximity_similarity, self.proximity_threshold, ignore_labels
        )

        loss = (
            self.flow_weight * flow_loss
            + self.color_weight * color_loss
            + self.proximity_weight * proximity_loss
        )

        result_dict = {"loss": loss}
        return result_dict