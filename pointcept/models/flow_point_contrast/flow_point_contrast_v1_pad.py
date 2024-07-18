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


def pad_batch(tensors, pad_value=0):
    max_length = max(tensor.size(0) for tensor in tensors)
    padded_tensors = []
    masks = []
    
    for tensor in tensors:
        padding_size = max_length - tensor.size(0)
        if (tensor.dim() > 1):
            padded_tensor = torch.nn.functional.pad(tensor, (0, 0, 0, padding_size), value=pad_value)
        else:
            padded_tensor = torch.nn.functional.pad(tensor, (0, padding_size), value=pad_value)
        mask = torch.ones(max_length, dtype=torch.bool)
        mask[tensor.size(0):] = False
        
        padded_tensors.append(padded_tensor)
        masks.append(mask)
    
    return torch.stack(padded_tensors), torch.stack(masks).to(tensors[0].device)

def generate_positive_and_negative_masks(labels):
    """Generates positive and negative masks used by contrastive loss."""
    labels_column = labels.unsqueeze(-1)  # Shape: (batch, size, 1)
    labels_row = labels.unsqueeze(1)  
    positive_mask = torch.eq(labels_column, labels_row).float()
    negative_mask = 1 - positive_mask
    return positive_mask, negative_mask

def hard_compute_contrastive_loss(logits, positive_mask, negative_mask):
    """Contrastive loss function."""
    exp_logits = torch.exp(logits) 
    
    normalized_exp_logits = exp_logits / (exp_logits + torch.sum(exp_logits * negative_mask, dim=2, keepdim=True))
    neg_log_likelihood = -torch.log(normalized_exp_logits)

    normalized_weight = positive_mask / torch.clamp(torch.sum(positive_mask, dim=2, keepdim=True), min=1e-6)
    neg_log_likelihood = torch.sum(neg_log_likelihood * normalized_weight, dim=2)

    positive_mask_sum = torch.sum(positive_mask, dim=2)
    valid_index = 1 - (positive_mask_sum == 0).float()
    normalized_weight = valid_index / torch.clamp(torch.sum(valid_index, dim=1, keepdim=True), min=1e-6)
    neg_log_likelihood = torch.sum(neg_log_likelihood * normalized_weight, dim=1)
    loss = torch.mean(neg_log_likelihood)

    return loss

def masked_contrastive_loss(features, mask, temperature=0.07, mask_valid=None):
    """Computes within-image supervised pixel contrastive loss."""
    normalized_features = features / (torch.norm(features, p=2, dim=-1, keepdim=True) + 1e-7)
    logits = torch.matmul(normalized_features, normalized_features.permute(0, 2, 1)) / temperature
    positive_mask, negative_mask = generate_positive_and_negative_masks(mask)
    loss =  hard_compute_contrastive_loss(logits, positive_mask, negative_mask)

    if get_world_size() > 1:
        dist.all_reduce(loss)

    return loss / get_world_size()


def within_sample_contrastive_loss(features, sim_scores, tau, gamma, eta, nu, mask=None):
    """Computes within-sample contrastive loss."""
    normalized_features = features / (torch.norm(features, p=2, dim=-1, keepdim=True) + 1e-7)
    return compute_contrastive_loss(normalized_features, sim_scores, tau, gamma, eta, nu, mask=mask)

def compute_contrastive_loss(features, sim_scores, tau, gamma, eta, nu, score_higher_positive=True, mask=None):
    """Contrastive loss function."""
    # Calculate g_plus and g_minus
    g_plus = 1 / (1 + torch.exp(gamma * (tau - sim_scores)))
    g_minus = 1 / (1 + torch.exp(gamma * (sim_scores - tau)))

    # Calculate all-pairs similarity of features
    feature_similarity = torch.matmul(features, features.transpose(-2, -1))

    # Calculate s_a_plus and s_a_minus using weighted sum
    if score_higher_positive:
        s_a_plus = feature_similarity * g_plus
        s_a_minus = feature_similarity * g_minus
    else:
        s_a_plus = feature_similarity * g_minus
        s_a_minus = feature_similarity * g_plus
    
    s_a_plus_exp = torch.sum(torch.exp(eta * s_a_plus), dim=-1)
    s_a_minus_exp = torch.sum(torch.exp(-nu * s_a_minus), dim=-1)

    # Calculate the loss for each point
    loss_per_point = (torch.log(1 + s_a_plus_exp) / eta) + (torch.log(1 + s_a_minus_exp) / nu)

    # Apply mask if provided
    if mask is not None:
        loss_per_point = loss_per_point * mask
        valid_points = mask.sum(dim=-1)
    else:
        valid_points = torch.ones(loss_per_point.shape[0], device=loss_per_point.device)

    # Sum the loss over all points and normalize
    loss = (loss_per_point.sum(dim=-1) / valid_points).mean()

    # Because the features similarity is being weighted, this needs to be negated
    loss = -loss

    if get_world_size() > 1:
        dist.all_reduce(loss)
    return loss / get_world_size()


@MODELS.register_module("FPC-v1p")
class FlowPointContrastPad(nn.Module):
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
    def calculate_flow_similarity(self, flow, mask):
        flow_norm = torch.norm(flow, dim=-1)
        # Normalized flow similarity, should be between -1 and 1
        flow_similarity = torch.matmul(flow, flow.transpose(-2, -1)) / (
            flow_norm.unsqueeze(-1) * flow_norm.unsqueeze(-2) + 1e-8
        )
        # Apply mask to zero out similarities involving padded elements
        flow_similarity = flow_similarity * mask.unsqueeze(-1) * mask.unsqueeze(-2)
        return flow_similarity

    def calculate_color_similarity(self, colors, mask):
        # Normalize colors to be between 0 and 1
        colors = colors.to(torch.float32) / 255.0
        
        # Compute the pairwise Euclidean distances between normalized color vectors
        distances = torch.cdist(colors, colors)
        
        # Normalize distances by the maximum possible distance in the normalized space
        max_distance = torch.sqrt(torch.tensor(colors.shape[-1]))  # sqrt(d) where d is the dimension of the color space
        normalized_distances = distances / max_distance
        
        # Compute similarity as 1 minus the normalized distances
        color_similarity = 1 - normalized_distances
        
        # Apply mask to zero out similarities involving padded elements
        color_similarity = color_similarity * mask.unsqueeze(-1) * mask.unsqueeze(-2)
        
        return color_similarity


    def calculate_proximity_similarity(self, points, mask, sigma=0.1):
        distances = torch.cdist(points, points)
        proximity_similarity = torch.exp(-distances / (2 * sigma ** 2))
        # Apply mask to zero out similarities involving padded elements
        proximity_similarity = proximity_similarity * mask.unsqueeze(-1) * mask.unsqueeze(-2)
        return proximity_similarity

    def forward(self, data_dict):
        flow = data_dict["flow"]
        colors = data_dict["color"]
        points = data_dict["coord"]
        sam_label = data_dict["sam"]
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
        batch_size = len(sizes)

        # Pad the tensors
        flow_padded, flow_mask = pad_batch(torch.split(flow, sizes))
        colors_padded, colors_mask = pad_batch(torch.split(colors, sizes))
        points_padded, points_mask = pad_batch(torch.split(points, sizes))
        features_padded, features_mask = pad_batch(torch.split(features, sizes))
        sam_padded, sam_mask = pad_batch(torch.split(sam_label, sizes))

        # Calculate similarities
        flow_similarity = self.calculate_flow_similarity(flow_padded, flow_mask)
        color_similarity = self.calculate_color_similarity(colors_padded, colors_mask)
        proximity_similarity = self.calculate_proximity_similarity(points_padded, points_mask)
        
        # Compute losses
        flow_loss = within_sample_contrastive_loss(
            features_padded, flow_similarity, self.flow_similarity_threshold, gamma=5.0, eta=1, nu=1, mask=features_mask
        )

        color_loss = within_sample_contrastive_loss(
            features_padded, color_similarity, self.color_similarity_threshold, gamma=5.0, eta=1, nu=1, mask=features_mask
        )

        proximity_loss = within_sample_contrastive_loss(
            features_padded, proximity_similarity, self.proximity_threshold, gamma=5.0, eta=1, nu=1, mask=features_mask
        )

        sam_loss = masked_contrastive_loss(features_padded, sam_padded, temperature=self.sam_weight, mask_valid=features_mask)
        
        loss = (
            self.flow_weight * flow_loss
            + self.color_weight * color_loss
            + self.proximity_weight * proximity_loss
            + self.sam_weight * sam_loss
        )
        result_dict = {"loss": loss}
        return result_dict