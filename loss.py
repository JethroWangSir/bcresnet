import torch
import torch.nn as nn
import torch.nn.functional as F

def softmax_loss(logits, targets):
    """
    Computes the softmax cross-entropy loss.
    :param logits: Tensor of shape (N, C) where N is batch size and C is number of classes.
    :param targets: Tensor of shape (N,) with class indices.
    :return: Scalar loss value.
    """
    
    return F.cross_entropy(logits, targets, ignore_index=-1)

# def weighted_focal_loss(logits, targets, alpha=0.25, gamma=2.0, weight=None):
#     """
#     Computes the weighted focal loss.
#     :param logits: Tensor of shape (N, C) where N is batch size and C is number of classes.
#     :param targets: Tensor of shape (N,) with class indices.
#     :param alpha: Weighting factor for class imbalance.
#     :param gamma: Focusing parameter to down-weight easy examples.
#     :param weight: Optional tensor of shape (C,) containing class weights.
#     :return: Scalar loss value.
#     """

#     probs = F.softmax(logits, dim=1)  # Convert logits to probabilities
#     targets_one_hot = F.one_hot(targets, num_classes=2).float()
    
#     pt = (probs * targets_one_hot).sum(dim=1)  # Get the probability of the true class
#     log_pt = torch.log(pt + 1e-8)  # Avoid log(0)
    
#     focal_weight = (1 - pt) ** gamma
    
#     if weight is not None:
#         class_weights = weight[targets]  # Apply class-wise weights
#         focal_weight *= class_weights
    
#     loss = -alpha * focal_weight * log_pt
#     return loss.mean()

def weighted_focal_loss(logits, targets, gamma=2.0):
    """
    Computes the weighted focal loss with class weights inversely proportional to class frequencies.
    
    :param logits: Tensor of shape (N, C) where N is batch size and C is number of classes.
    :param targets: Tensor of shape (N,) with class indices.
    :param gamma: Focusing parameter to down-weight easy examples.
    :return: Scalar loss value.
    """

    # Compute class counts
    num_classes = 2
    class_counts = torch.bincount(targets, minlength=num_classes).float()
    
    # Compute inverse class weights
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights /= class_weights.sum()  # Normalize weights
    
    probs = F.softmax(logits, dim=1)  # Convert logits to probabilities
    targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()
    
    pt = (probs * targets_one_hot).sum(dim=1)  # Get the probability of the true class
    log_pt = torch.log(pt + 1e-8)  # Avoid log(0)
    
    focal_weight = (1 - pt) ** gamma
    sample_weights = class_weights[targets]  # Apply inverse class weights
    focal_weight *= sample_weights
    
    loss = -focal_weight * log_pt
    return loss.mean()
