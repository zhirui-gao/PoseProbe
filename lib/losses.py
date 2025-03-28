import torch
import torch.nn.functional as F
from easydict import EasyDict as edict
import torch.nn as nn
import math
def deform_implicit_loss(model_output, use_deform):
    gradient_sdf = model_output['gradient']
    grad_constraint = torch.abs(
        gradient_sdf.norm(dim=-1) - 1)  # (torch.linalg.norm(gradient_sdf, ord=2, dim=-1) - 1.0) ** 2
    loss_dict = {'grad_constraint': grad_constraint.mean()}  # *0.1
    if use_deform:
        grad_deform_constraint = model_output['grad_deform'].norm(dim=-1)
        # minimal correction prior
        sdf_correct_constraint = torch.abs(model_output['sdf_correct'])
        # minimal deform prior
        sdf_deform_constraint =  torch.abs(model_output['sdf_deform'])#model_output['deform'].norm(dim=-1)
        loss_dict.update({
            'grad_deform_constraint':grad_deform_constraint.mean(),
            'sdf_correct_constraint':sdf_correct_constraint.mean(),
            'sdf_deform_constraint':sdf_deform_constraint.mean(),
            # 'grad_temp_constraint':grad_temp_constraint.mean(),
            # 'normal_constraint':normal_constraint.mean(),
            })
    return loss_dict

def rendering_loss(rgb_marched, target, mask):
    loss_mse_render = F.mse_loss(rgb_marched * mask, target * mask, reduction='sum') \
                      / (mask.sum() * 3)
    return loss_mse_render
def dynamic_weight(initial_weight, final_weight, iteration, total_iterations):
    decay_rate = math.log(final_weight / initial_weight) / total_iterations
    return initial_weight * math.exp(decay_rate * iteration)

def object_losses(model_output, cfg_train, target, mask, iteration, total_iterations ,use_deform):
    # use_deform = False
    loss_scalars, loss_weight = edict(), edict()

    # rendering
    loss_mse_render = rendering_loss(model_output['rgb_marched'],target, mask)
    loss_scalars.img_render = loss_mse_render
    loss_weight.img_render = cfg_train.weight_main
    pout = model_output['alphainv_cum'].clamp(1e-6, 1 - 1e-6)
    entropy_last_loss = -(pout * torch.log(pout) + (1 - pout) * torch.log(1 - pout)).mean()
    loss_scalars.weight_entropy_last = entropy_last_loss
    loss_weight.weight_entropy_last = 0.01

    if cfg_train.weight_tv_k0>0:
        loss_scalars.tv_k0 = model_output['k0_tv']
        loss_weight.tv_k0 = cfg_train.weight_tv_k0
    # implicit loss
    implicit_loss = deform_implicit_loss(model_output, use_deform)
    loss_scalars.grad_constraint = implicit_loss['grad_constraint']
    loss_weight.grad_constraint = 1.0
    if use_deform:
        weight = dynamic_weight(1e-1, 1e-3, iteration, total_iterations)
        loss_scalars.grad_deform_constraint = implicit_loss['grad_deform_constraint']
        loss_weight.grad_deform_constraint = weight

        loss_scalars.sdf_correct_constraint = implicit_loss['sdf_correct_constraint']
        loss_weight.sdf_correct_constraint = weight

        loss_scalars.sdf_deform_constraint = implicit_loss['sdf_deform_constraint']
        loss_weight.sdf_deform_constraint = weight

    # mask loss
    loss_mask_render = F.binary_cross_entropy(model_output['cum_weights'].clip(1e-3, 1.0 - 1e-3), mask)
    loss_scalars.mask_render = loss_mask_render
    loss_weight.mask_render = cfg_train.weight_mask

    loss = 0
    for key, value in loss_scalars.items():
        loss = loss + value* loss_weight[key]

    return loss_scalars, loss_weight, loss


def compute_diff_loss(loss_type: str, diff: torch.Tensor, weights: torch.Tensor = None,
                      var: torch.Tensor = None, mask: torch.Tensor = None, dim=-1, delta=1.):
    if loss_type.lower() == 'epe':
        loss = torch.norm(diff, 2, dim, keepdim=True)
    elif loss_type.lower() == 'l1':
        loss = torch.abs(diff)
    elif loss_type.lower() == 'mse':
        loss = diff ** 2
    elif loss_type.lower() == 'huber':
        loss = nn.functional.huber_loss(diff, torch.zeros_like(diff), reduction='none', delta=delta)
    else:
        raise ValueError('Wrong loss type: {}'.format(loss_type))

    if weights is not None:
        assert len(weights.shape) == len(loss.shape)

        loss = loss * weights

    if var is not None:
        eps = torch.tensor(1e-3)
        loss = loss / (torch.maximum(var, eps)) + torch.log(torch.maximum(var, eps))

    if mask is not None:
        assert len(mask.shape) == len(loss.shape)
        loss = loss * mask.float()
        return loss.sum() / (mask.float().sum() + 1e-6)
    return loss.sum() / (loss.nelement() + 1e-6)