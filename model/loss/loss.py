# Copyright (c) Meta Platforms, Inc. All Rights Reserved
from typing import Any, List, Union, Tuple
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class NeRFMSELoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, inputs, targets):
        loss_coarse = torch.zeros([1], device=targets.device)
        loss_fine = torch.zeros([1], device=targets.device)
        if 'rgb_coarse' in inputs:
            loss_coarse = self.loss(inputs['rgb_coarse'], targets)
        if 'rgb_fine' in inputs:
            loss_fine = self.loss(inputs['rgb_fine'], targets)
        return loss_coarse, loss_fine


class NeRFSemanticsLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, inputs, targets, key):
        loss_coarse = torch.zeros([1], device=targets.device)
        loss_fine = torch.zeros([1], device=targets.device)
        if f'{key}_coarse' in inputs:
            loss_coarse = self.loss(inputs[f'{key}_coarse'], targets)
        if f'{key}_fine' in inputs:
            loss_fine = self.loss(inputs[f'{key}_fine'], targets)
        return loss_coarse, loss_fine


class TVLoss(nn.Module):

    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.size_tensor(x[:, :, 1:, :]) + 1e-4
        count_w = self.size_tensor(x[:, :, :, 1:]) + 1e-4
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def size_tensor(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


def get_semantic_weights(reweight_classes, fg_classes, num_semantic_classes):
    weights = torch.ones([num_semantic_classes]).float()
    if reweight_classes:
        weights[fg_classes] = 2
    return weights


class InstanceMSELoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, rgb_all_instance, targets, instances):
        targets = targets.unsqueeze(1).expand(-1, rgb_all_instance.shape[-1], -1)
        instances = instances.unsqueeze(-1).expand(-1, rgb_all_instance.shape[-1])
        instance_values = torch.tensor(list(range(rgb_all_instance.shape[-1]))).to(instances.device).unsqueeze(0).expand(instances.shape[0], -1)
        instance_mask = instances == instance_values
        loss = self.loss(rgb_all_instance.permute((0, 2, 1)).reshape(-1, 3)[instance_mask.view(-1), :], targets.reshape(-1, 3)[instance_mask.view(-1), :])
        return loss


class MaskedNLLLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss = torch.nn.NLLLoss(reduction='mean')

    def forward(self, output_instances, instances, semantics, invalid_class):
        if invalid_class is None:
            return self.loss(output_instances, instances)
        mask = semantics != invalid_class
        return self.loss(output_instances[mask, :], instances[mask])


class SCELoss(torch.nn.Module):

    def __init__(self, alpha, beta, class_weights):
        super(SCELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.class_weights = class_weights
        self.cross_entropy = torch.nn.CrossEntropyLoss(weight=class_weights, reduction='none')

    def forward(self, pred, labels_probabilities):
        # CCE
        ce = self.cross_entropy(pred, labels_probabilities)

        # RCE
        weights = torch.tensor(self.class_weights, device=pred.device).unsqueeze(0)
        pred = F.softmax(pred * weights, dim=1)
        pred = torch.clamp(pred, min=1e-8, max=1.0)
        label_clipped = torch.clamp(labels_probabilities, min=1e-8, max=1.0)

        rce = torch.sum(-1 * (pred * torch.log(label_clipped) * weights), dim=1)

        # Loss
        loss = self.alpha * ce + self.beta * rce
        return loss

        
class CorrespondencesLoss(nn.Module):

    def __init__(self):
        super(CorrespondencesLoss, self).__init__()
        
    def compute_diff_loss(self, diff: torch.Tensor, weights: torch.Tensor=None, mask: torch.Tensor=None, loss_type: str='huber', dim=-1):
        if loss_type.lower() == 'epe':
            loss = torch.norm(diff, 2, dim, keepdim=True)
        elif loss_type.lower() == 'l1': 
            loss = torch.abs(diff)
        elif loss_type.lower() == 'mse':
            loss = diff**2
        elif loss_type.lower() == 'huber':
            delta = 1.
            loss = nn.functional.huber_loss(diff, torch.zeros_like(diff), reduction='none', delta=delta)
        else:
            raise ValueError('Wrong loss type: {}'.format(loss_type))
        
        if weights is not None:
            assert len(weights.shape) == len(loss.shape)
        
            loss = loss * weights
            
        if mask is not None:
            assert len(mask.shape) == len(loss.shape)
            loss = loss * mask.float()
            return loss.sum() / (mask.float().sum() + 1e-6)
        return loss.sum() / (loss.nelement() + 1e-6)

    def compute_render_and_repro_loss_w_repro_thres(self, pixels_in_self_int: torch.Tensor, 
                                                    depth_rendered_self: torch.Tensor, intr_self: torch.Tensor, 
                                                    pixels_in_other: torch.Tensor, depth_rendered_other: torch.Tensor, 
                                                    intr_other: torch.Tensor, T_self2other: torch.Tensor, conf_values: torch.Tensor,):

        pts_self_repr_in_other, depth_self_repr_in_other = batch_project_to_other_img(
            pixels_in_self_int.float(), di=depth_rendered_self, 
            Ki=intr_self, Kj=intr_other, T_itoj=T_self2other, return_depth=True)
        
        loss = torch.norm(pts_self_repr_in_other - pixels_in_other, dim=-1, keepdim=True) # [N_rays, 1]
        valid = torch.ones_like(loss).bool()
        
        loss_corres = self.compute_diff_loss(diff=pts_self_repr_in_other - pixels_in_other, weights=conf_values, mask=valid, dim=-1)
        
        return loss_corres
    
    
    def forward(self, pixels_in_self, self_depth, self_intrinsics, self_c2w,
                      pixels_in_other, other_depth, other_intrinsics, other_c2w, corres_confidences):
        
        other_w2c = pose_inverse_4x4(other_c2w)
        T_self2other = other_w2c @ self_c2w
        
        loss_corres = self.compute_render_and_repro_loss_w_repro_thres\
            (pixels_in_self, self_depth, self_intrinsics, pixels_in_other, 
             other_depth, other_intrinsics, T_self2other, corres_confidences)

        loss_corres_ = self.compute_render_and_repro_loss_w_repro_thres\
            (pixels_in_other, other_depth, other_intrinsics, pixels_in_self, 
             self_depth, self_intrinsics, pose_inverse_4x4(T_self2other), corres_confidences)  
             
        loss_corres += loss_corres_
        
        return loss_corres / 2
            
def pose_inverse_4x4(mat: torch.Tensor, use_inverse: bool=False):
    """
    Transforms world2cam into cam2world or vice-versa, without computing the inverse.
    Args:
        mat (torch.Tensor): pose matrix (B, 4, 4) or (4, 4)
    """
    # invert a camera pose
    out_mat = torch.zeros_like(mat)

    if len(out_mat.shape) == 3:
        # must be (B, 4, 4)
        out_mat[:, 3, 3] = 1
        R,t = mat[:, :3, :3],mat[:,:3, 3:]
        R_inv = R.inverse() if use_inverse else R.transpose(-1,-2)
        t_inv = (-R_inv@t)[..., 0]

        pose_inv = torch.cat([R_inv,t_inv[...,None]],dim=-1) # [...,3,4]

        out_mat[:, :3] = pose_inv
    else:
        out_mat[3, 3] = 1
        R,t = mat[:3, :3], mat[:3, 3:]
        R_inv = R.inverse() if use_inverse else R.transpose(-1,-2)
        t_inv = (-R_inv@t)[..., 0]
        pose_inv = torch.cat([R_inv,t_inv[...,None]],dim=-1) # [3,4]
        out_mat[:3] = pose_inv
    # assert torch.equal(out_mat, torch.inverse(mat))
    return out_mat
    
def to_homogeneous(points: Union[torch.Tensor, np.ndarray]):
    """Convert N-dimensional points to homogeneous coordinates.
    Args:
        points: torch.Tensor or numpy.ndarray with size (..., N).
    Returns:
        A torch.Tensor or numpy.ndarray with size (..., N+1).
    """
    if isinstance(points, torch.Tensor):
        pad = points.new_ones(points.shape[:-1]+(1,))
        return torch.cat([points, pad], dim=-1)
    elif isinstance(points, np.ndarray):
        pad = np.ones((points.shape[:-1]+(1,)), dtype=points.dtype)
        return np.concatenate([points, pad], axis=-1)
    else:
        raise ValueError

def from_homogeneous(points: Union[torch.Tensor, np.ndarray]):
    """Remove the homogeneous dimension of N-dimensional points.
    Args:
        points: torch.Tensor or numpy.ndarray with size (..., N+1).
    Returns:
        A torch.Tensor or numpy ndarray with size (..., N).
    """
    return points[..., :-1] / (points[..., -1:] + 1e-6)
    
def batch_project_to_other_img(kpi: torch.Tensor, di: torch.Tensor, 
                               Ki: torch.Tensor, Kj: torch.Tensor, 
                               T_itoj: torch.Tensor, 
                               return_depth=False) -> torch.Tensor:
    """
    Project pixels of one image to the other. 
    Args:
        kpi: BxNx2 coordinates in pixels of image i
        di: BxN, corresponding depths of image i
        Ki: intrinsics of image i, Bx3x3
        Kj: intrinsics of image j, Bx3x3
        T_itoj: Transform matrix from coordinate system of i to j, Bx4x4
        return_depth: Bool

    Returns:
        kpi_j: Pixels projection in image j, BxNx2
        di_j: Depth of the projections in image j, BxN
    """
    if len(di.shape) == len(kpi.shape):
        # di must be BxNx1
        di = di.squeeze(-1)
    kpi_3d_i = to_homogeneous(kpi) @ torch.inverse(Ki).transpose(-1, -2)
    kpi_3d_i = kpi_3d_i * di[..., None]  # non-homogeneous coordinates
    kpi_3d_j = from_homogeneous(
            to_homogeneous(kpi_3d_i) @ T_itoj.transpose(-1, -2))
    kpi_j = from_homogeneous(kpi_3d_j @ Kj.transpose(-1, -2))
    if return_depth:
        di_j = kpi_3d_j[..., -1]
        return kpi_j, di_j
    return kpi_j