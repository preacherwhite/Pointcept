import random
from itertools import chain
import torch
import torch.nn as nn

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

def hard_compute_contrastive_loss(logits_list, mask, skip_mask = False):
    """Contrastive loss function."""
    batch_size = len(logits_list)
    losses = []

    for i in range(batch_size):
        logits = logits_list[i]
        if not skip_mask:
            positive_mask, negative_mask = generate_positive_and_negative_masks(mask[i])
        else:
            positive_mask = mask[i]
            negative_mask = 1-mask[i]

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

def masked_contrastive_loss(features_list, mask, temperature=0.07, skip_mask = False):
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

    return hard_compute_contrastive_loss(logits_list, mask, skip_mask)

def create_mask_and_compute_loss(features_list, sim_scores_list, threshold, temperature=0.07):
    """Creates a mask for the sim_scores_list using the threshold, then calls masked_contrastive_loss."""
    batch_size = len(sim_scores_list)
    mask = []

    for i in range(batch_size):
        sim_scores = sim_scores_list[i]
        submask = sim_scores > threshold
        submask = submask.float()
        mask.append(submask)
    return masked_contrastive_loss(features_list, mask, temperature, skip_mask=True)



def within_sample_contrastive_loss(features_list, sim_scores_list, tau, gamma, eta, nu, temperature = 0.07):
    """Computes within-sample contrastive loss."""
    batch_size = len(features_list)
    normalized_features_list = []

    for i in range(batch_size):
        features = features_list[i]
        normalized_features = features / (torch.norm(features, p=2, dim=1, keepdim=True) + 1e-7)
        normalized_features_list.append(normalized_features)

    return compute_contrastive_loss(normalized_features_list, sim_scores_list, tau, gamma, eta, nu, temperature)

def compute_contrastive_loss(features_list, sim_scores_list, tau, gamma, eta, nu, score_higher_positive = True, temperature = 0.07):
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

def subsample_features(features, sample_ratio):
    """
    Subsamples the features before similarity matrix computation.
    
    Args:
        features (torch.Tensor): The feature tensor.
        sample_ratio (float): The ratio of features to sample.
        
    Returns:
        Subsampled features and indices used for sampling.
    """
    num_points = features.shape[0]
    sample_size = max(1, int(sample_ratio * num_points))  # Ensure at least one point is sampled
    sampled_indices = random.sample(range(num_points), sample_size)
    
    sampled_indices = torch.tensor(sampled_indices).to(features.device)  # Move indices to the same device

    # Subsample the features
    sampled_features = features[sampled_indices]
    
    return sampled_features, sampled_indices

def subsample_local_neighborhood(features, points, num_samples):
    """
    Subsamples the features by selecting a local neighborhood around a randomly chosen center point.
    
    Args:
        features (torch.Tensor): The feature tensor.
        points (torch.Tensor): The point coordinates tensor.
        num_samples (int): The number of samples to select in the local neighborhood.
        
    Returns:
        Subsampled features and indices used for sampling.
    """
    num_points = points.shape[0]
    center_index = random.randint(0, num_points - 1)  # Randomly pick a center point
    center = points[center_index]
    
    # Calculate distances from the center point
    distances = torch.norm(points - center, dim=1)
    sorted_indices = torch.argsort(distances)
    
    # Select the closest points to the center point
    sampled_indices = sorted_indices[:num_samples]
    
    # Subsample the features
    sampled_features = features[sampled_indices]
    
    return sampled_features, sampled_indices

def subsample_batch_local_neighborhood(features_split, points_split, sample_ratio):
    """
    Subsamples the features by selecting a local neighborhood around a randomly chosen center point.
    """
    subsampled_features = []
    sampled_indices_list = []
    for i in range(len(features_split)):
        features = features_split[i]
        points = points_split[i]
        num_samples = int(sample_ratio * features.shape[0])
        sampled_features, sampled_indices = subsample_local_neighborhood(features, points, num_samples)
        subsampled_features.append(sampled_features)
        sampled_indices_list.append(sampled_indices)
    return subsampled_features, sampled_indices_list

def subsample_batch(features_split, sample_ratio):
    """
    Subsamples features before similarity matrix computation for a batch of data.
    
    Args:
        features_split (list): List of feature tensors for each batch.
        sample_ratio (float): Ratio of features to sample.
    
    Returns:
        Subsampled features and corresponding sampled indices for each batch.
    """
    subsampled_features = []
    sampled_indices_list = []
    
    for features in features_split:
        sampled_features, sampled_indices = subsample_features(features, sample_ratio)
        subsampled_features.append(sampled_features)
        sampled_indices_list.append(sampled_indices)
    
    return subsampled_features, sampled_indices_list


@MODELS.register_module("FPC-v1-W")
class FlowPointContrastWaymo(nn.Module):
    def __init__(
        self,
        backbone,
        flow_similarity_threshold=0.8,
        color_similarity_threshold=0.8,
        proximity_threshold=0.8,
        bilateral_threshold = 0.8,
        flow_weight=3.0,
        color_weight=1.0,
        proximity_weight=1.0,
        sam_weight = 1.0,
        sample_ratio = 0.5,
        sigma_color = 0.5,
        sigma_proximity = 2,
        bilateral_weight = 1.0,
        use_soft_contrastive = False,
        flow_norm_threshold=3,  # New parameter for flow norm threshold
        use_local_neighborhood = False,
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
        self.sample_ratio = sample_ratio
        self.sigma_color = sigma_color
        self.sigma_proximity = sigma_proximity
        self.use_soft_contrastive = use_soft_contrastive
        self.bilateral_weight = bilateral_weight
        self.bilateral_threshold = bilateral_threshold
        self.flow_norm_threshold = flow_norm_threshold
        self.use_local_neighborhood = use_local_neighborhood
    def calculate_flow_similarity(self, flow):
        flow_norm = torch.norm(flow, dim=-1)
        #nomalized flow similarity, should be between -1 and 1
        flow_similarity = torch.matmul(flow, flow.transpose(0, 1)) / (
            flow_norm.unsqueeze(1) * flow_norm.unsqueeze(0) + 1e-8
        )
        return flow_similarity

    def calculate_color_similarity(self, colors):
        # Normalize colors to be between 0 and 1
        colors = colors.to(torch.float32)
        
        # Compute the pairwise Euclidean distances between normalized color vectors
        distances = torch.cdist(colors, colors)
        
        # Normalize distances by the maximum possible distance in the normalized space
        max_distance = torch.sqrt(torch.tensor(colors.shape[-1]))  # sqrt(d) where d is the dimension of the color space
        normalized_distances = distances / max_distance

        color_similarity = torch.exp(-normalized_distances/(2*self.sigma_color**2))
        
        return color_similarity


    def calculate_proximity_similarity(self, points):
        distances = torch.cdist(points, points)
        proximity_similarity = torch.exp(-distances / (2 * self.sigma_proximity ** 2))
        return proximity_similarity
    
    def calculate_bilateral_similarity(self, colors, points):
    # Normalize colors to be between 0 and 1
        colors = colors.to(torch.float32)
        
        # Compute color similarity
        color_distances = torch.cdist(colors, colors)
        max_color_distance = torch.sqrt(torch.tensor(colors.shape[-1]))  # sqrt(d) where d is the dimension of the color space
        normalized_color_distances = color_distances / max_color_distance
        color_similarity = torch.exp(-normalized_color_distances / (2 * self.sigma_color ** 2))
        
        # Compute proximity similarity
        proximity_distances = torch.cdist(points, points)
        proximity_similarity = torch.exp(-proximity_distances / (2 * self.sigma_proximity ** 2))
        
        # Combine the similarities using the product
        bilateral_similarity = color_similarity * proximity_similarity
    
        return bilateral_similarity

    def forward(self, data_dict):
        if not self.training:
            features = self.backbone(data_dict).feat
            if isinstance(features, dict):
                features = features.dict
            return features

        flow = data_dict["flow"]
        colors = data_dict["color"]
        points = data_dict["coord"]
        sam_label = data_dict["sam"]

        features = self.backbone(data_dict).feat
        if isinstance(features, dict):
            features = features.dict
        
        offset = data_dict['offset']
        batch_size = len(offset)

        offset = torch.cat([torch.Tensor([0]).to(features.device), data_dict['offset']])
        offset = torch.round(offset).int()
        sizes = (offset[1:] - offset[:-1]).tolist()
        flow_split = torch.split(flow, sizes)
        colors_split = torch.split(colors, sizes)
        points_split = torch.split(points, sizes)
        features_split = torch.split(features, sizes)    
        sam_label_split = torch.split(sam_label, sizes)

        # Filter out all points where color is zero
        non_zero_color_indices = [torch.nonzero(colors_split[i].sum(dim=1), as_tuple=True)[0] for i in range(batch_size)]
        colors_split = [colors_split[i][non_zero_color_indices[i]] for i in range(batch_size)]
        points_split = [points_split[i][non_zero_color_indices[i]] for i in range(batch_size)]
        features_split = [features_split[i][non_zero_color_indices[i]] for i in range(batch_size)]
        flow_split = [flow_split[i][non_zero_color_indices[i]] for i in range(batch_size)]
        sam_label_split = [sam_label_split[i][non_zero_color_indices[i]] for i in range(batch_size)]

        if self.use_local_neighborhood:
            # Randomly decide whether to use local neighborhood or global sampling
            sampled_features_split, sampled_indices_list = subsample_batch_local_neighborhood(features_split, points_split, self.sample_ratio)
        else:
            sampled_features_split, sampled_indices_list = subsample_batch(features_split, self.sample_ratio)
        
        flow_similarity_list = []
        color_similarity_list = []
        proximity_similarity_list = []
        bilateral_similarity_list = []
        flow_sampled_features_list = []
        sam_sampled_label_list = []
        for i in range(batch_size):
            sampled_colors = None
            sampled_points = None
            # Calculate the similarity matrices on subsampled data
            if self.flow_weight > 0:
                # Apply flow norm threshold
                flow_norm = torch.norm(flow_split[i], dim=-1)
                flow_mask = flow_norm > self.flow_norm_threshold
                
                # Only use points with flow norm above threshold
                filtered_flow = flow_split[i][flow_mask]
                filtered_features = features_split[i][flow_mask]
                flow_sampled_features_list.append(filtered_features)
                if filtered_flow.shape[0] > 1:  # Ensure we have at least 2 points for similarity calculation
                    flow_similarity = self.calculate_flow_similarity(filtered_flow)
                    flow_similarity_list.append(flow_similarity)
                else:
                    flow_similarity_list.append(None)  # Indicate that this batch item should be skipped for flow loss


            if self.color_weight > 0:
                sampled_colors = colors_split[i][sampled_indices_list[i]]
                color_similarity = self.calculate_color_similarity(sampled_colors)
                color_similarity_list.append(color_similarity)

            if self.proximity_weight > 0:
                sampled_points = points_split[i][sampled_indices_list[i]]
                proximity_similarity = self.calculate_proximity_similarity(sampled_points)
                proximity_similarity_list.append(proximity_similarity)

            if self.bilateral_weight > 0:
                if sampled_colors is None:
                    sampled_colors = colors_split[i][sampled_indices_list[i]]
                if sampled_points is None:
                    sampled_points = points_split[i][sampled_indices_list[i]]
                bilateral_similarity = self.calculate_bilateral_similarity(sampled_colors, sampled_points)
                bilateral_similarity_list.append(bilateral_similarity)

            if self.sam_weight > 0:
                sam_label_sampled = sam_label_split[i][sampled_indices_list[i]]
                sam_sampled_label_list.append(sam_label_sampled)
        #Calculate contrastive loss using the subsampled features and similarity matrices
        flow_loss = 0.0
        color_loss = 0.0
        proximity_loss = 0.0
        bilateral_loss = 0.0
        sam_loss = 0.0
        if self.use_soft_contrastive:
            if self.flow_weight > 0:
                flow_loss = within_sample_contrastive_loss(
                    flow_sampled_features_list, flow_similarity_list, self.flow_similarity_threshold, gamma=5.0, eta=1, nu=1, temperature = 0.07
                )

            if self.color_weight > 0:
                color_loss = within_sample_contrastive_loss(
                sampled_features_split, color_similarity_list, self.color_similarity_threshold, gamma=5.0, eta=1, nu=1, temperature = 0.07
            )

            if self.proximity_weight > 0:
                proximity_loss = within_sample_contrastive_loss(
                    sampled_features_split, proximity_similarity_list, self.proximity_threshold, gamma=5.0, eta=1, nu=1, temperature = 0.07
                )

            if self.bilateral_weight > 0:
                bilateral_loss = within_sample_contrastive_loss(
                    sampled_features_split, bilateral_similarity_list, self.bilateral_threshold, gamma=5.0, eta=1, nu=1, temperature = 0.07
                )

        else:
            if self.flow_weight > 0:
                flow_loss = create_mask_and_compute_loss(
                    flow_sampled_features_list, flow_similarity_list, self.flow_similarity_threshold, temperature=0.07
                )

            if self.color_weight > 0:
                color_loss = create_mask_and_compute_loss(
                    sampled_features_split, color_similarity_list, self.color_similarity_threshold, temperature=0.07
                )

            if self.proximity_weight > 0:
                proximity_loss = create_mask_and_compute_loss(
                    sampled_features_split, proximity_similarity_list, self.proximity_threshold, temperature=0.07
                )

            if self.bilateral_weight > 0:
                bilateral_loss = create_mask_and_compute_loss(
                    sampled_features_split, bilateral_similarity_list, self.bilateral_threshold, temperature=0.07
                )
        if self.sam_weight > 0:
            sam_loss = masked_contrastive_loss(sampled_features_split, sam_sampled_label_list, temperature=0.07) 
        loss = self.flow_weight * flow_loss + self.color_weight * color_loss + self.proximity_weight * proximity_loss + self.bilateral_weight * bilateral_loss + self.sam_weight * sam_loss

        result_dict = {"loss": loss}
        return result_dict