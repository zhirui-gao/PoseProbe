"""
 Copyright 2022 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """


import torch
from easydict import EasyDict as edict
from typing import Callable, Sequence, List, Mapping, MutableMapping, Tuple, Union, Dict
from typing import Any, Optional
from lib.bg_nerf.source.utils.camera import pose_inverse_4x4
from lib.bg_nerf.source.utils.config_utils import override_options
from lib.bg_nerf.source.training.core.base_corres_loss import CorrespondenceBasedLoss
from lib.bg_nerf.source.utils.camera import pose_inverse_4x4
from lib.bg_nerf.source.utils.geometry.batched_geometry_utils import batch_project_to_other_img


class CorrespondencesPairRenderDepthAndGet3DPtsAndReproject(CorrespondenceBasedLoss):
    """The main class for the correspondence loss of SPARF. It computes the re-projection error
    between previously extracted correspondences relating the input views. The projection
    is computed with the rendered depth from the NeRF and the current camera pose estimates. 
    """
    def __init__(self, opt: Dict[str, Any], nerf_net: torch.nn.Module, flow_net: torch.nn.Module, 
                 train_data: Dict[str, Any], device: torch.device):
        super().__init__(opt, nerf_net, flow_net, train_data, device)
        default_cfg = edict({'diff_loss_type': 'huber', 
                             'compute_photo_on_matches': False, 
                             'renderrepro_do_pixel_reprojection_check': False, 
                             'renderrepro_do_depth_reprojection_check': False, 
                             'renderrepro_pixel_reprojection_thresh': 10., 
                             'renderrepro_depth_reprojection_thresh': 0.1, 
                             'use_gt_depth': False,  # debugging
                             'use_gt_correspondences': False,  # debugging
                             'use_dummy_all_one_confidence': False # debugging
                             })
        self.opt = override_options(self.opt, default_cfg)
        self.opt = override_options(self.opt, opt)

    def compute_loss(self, opt: Dict[str, Any], data_dict: Dict[str, Any],
                     output_dict: Dict[str, Any], iteration: int, mode: str = None, plot: bool = False
                     ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Args:
            opt (edict): settings
            data_dict (edict): Input data dict. Contains important fields:
                            - Image: GT images, (B, 3, H, W)
                            - intr: intrinsics (B, 3, 3)
                            - pose: gt w2c poses (B, 3, 4)
                            - pose_w2c: current estimates of w2c poses (B, 3, 4). When the camera poses
                            are fixed to gt, pose=pose_w2c. Otherwise, pose_w2c is being optimized.
                            - idx: idx of the images (B)
                            - depth_gt (optional): gt depth, (B, 1, H, W)
                            - valid_depth_gt (optional): (B, 1, H, W)
            output_dict (edict): Will not be used here, because rendering must be where
                                 a match is available.
            iteration (int)
            mode (str, optional): Defaults to None.
            plot (bool, optional): Defaults to False.
        """

        if mode != 'train':
            # only during training
            return {}, {}, {}

        loss_dict, stats_dict, plotting_dict = self.compute_loss_pairwise \
            (opt, data_dict, output_dict, iteration, mode, plot)
        if self.opt.gradually_decrease_corres_weight:
            # gamma = 0.1**(max(iteration - self.opt.start_iter_photometric, 0)/self.opt.max_iter)
            # reduce the corres weight by 2 every x iterations
            iter_start_decrease_corres_weight = self.opt.ratio_start_decrease_corres_weight * self.opt.max_iter \
                if self.opt.ratio_start_decrease_corres_weight is not None \
                else self.opt.iter_start_decrease_corres_weight
            if iteration < iter_start_decrease_corres_weight:
                gamma = 1.
            else:
                gamma = 2 ** (
                            (iteration - iter_start_decrease_corres_weight) // self.opt.corres_weight_reduct_at_x_iter)
            loss_dict['corres'] = loss_dict['corres'] / gamma
        return loss_dict, stats_dict, plotting_dict


    def compute_render_and_repro_loss_w_repro_thres(self, opt: Dict[str, Any], pixels_in_self_int: torch.Tensor, 
                                                    depth_rendered_self: torch.Tensor, intr_self: torch.Tensor, 
                                                    pixels_in_other: torch.Tensor, depth_rendered_other: torch.Tensor, 
                                                    intr_other: torch.Tensor, T_self2other: torch.Tensor, 
                                                    conf_values: torch.Tensor, stats_dict: Dict[str, Any], 
                                                    return_valid_mask: bool=False
                                                    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Computes the actual re-projection error loss between 'self' and 'other' images, 
        along with possible filterings. 
        
        Args:
            opt (edict): settings
            pixels_in_self_int (torch.Tensor): (N, 2)
            depth_rendered_self (torch.Tensor): (N)
            intr_self (torch.Tensor): (3, 3)
            pixels_in_other (torch.Tensor): (N, 2)
            depth_rendered_other (torch.Tensor): (N)
            intr_other (torch.Tensor): (3, 3)
            T_self2other (torch.Tensor): (4, 4)
            conf_values (torch.Tensor): (N, 1)
            stats_dict (dict): dict to keep track of statistics to be logged
            return_valid_mask (bool, optional): Defaults to False.
        """
        pts_self_repr_in_other, depth_self_repr_in_other = batch_project_to_other_img(
            pixels_in_self_int.float(), di=depth_rendered_self, 
            Ki=intr_self, Kj=intr_other, T_itoj=T_self2other, return_depth=True)

        loss = torch.norm(pts_self_repr_in_other - pixels_in_other, dim=-1, keepdim=True) # [N_rays, 1]
        valid = torch.ones_like(loss).bool()
        if opt.renderrepro_do_pixel_reprojection_check:
            valid_pixel = loss.detach().le(opt.renderrepro_pixel_reprojection_thresh)
            valid = valid & valid_pixel
            stats_dict['perc_val_pix_rep'] = valid_pixel.sum().float() / (valid_pixel.nelement() + 1e-6)

        if opt.renderrepro_do_depth_reprojection_check:
            valid_depth = torch.abs(depth_rendered_other - depth_self_repr_in_other) / (depth_rendered_other + 1e-6)
            valid_depth = valid_depth.detach().le(opt.renderrepro_depth_reprojection_thresh)
            valid = valid & valid_depth.unsqueeze(-1)
            stats_dict['perc_val_depth_rep'] = valid_depth.sum().float() / (valid_depth.nelement() + 1e-6)

        loss_corres = self.compute_diff_loss(loss_type=opt.diff_loss_type, diff=pts_self_repr_in_other - pixels_in_other, 
                                             weights=conf_values, mask=valid, dim=-1)

        if return_valid_mask:
            return loss_corres, stats_dict, valid
        return loss_corres, stats_dict

    def compute_loss_pairwise(self, opt: Dict[str, Any], data_dict: Dict[str, Any],
                              output_dict: Dict[str, Any], iteration: int, mode: str=None, plot: bool=False
                              ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        stats_dict, plotting_dict, loss_dict = {}, {}, {'corres': torch.tensor(0., requires_grad=True).to(self.device),
                                                        'render_matches': torch.tensor(0., requires_grad=True).to(
                                                            self.device)}

        if mode != 'train':
            # only during training
            return loss_dict, stats_dict, plotting_dict

        if iteration < self.opt.start_iter.corres:
            # if the correspondence loss is only added after x iterations
            return loss_dict, stats_dict, plotting_dict

        # the actual render and reproject code
        images = data_dict['image'].permute(0,2,3,1)
        B, H, W  = images.shape[:3]

        id_self, id_other = data_dict.corrs_id[0], data_dict.corrs_id[1]
        mask = data_dict.mconf_scene[id_self]>0
        pixels_in_self, pixels_in_other = data_dict.coord1_scene[id_self][mask], data_dict.coord0_scene[id_self][mask]
        conf_values = data_dict.mconf_scene[id_self][mask].unsqueeze(1)

        intr_self, intr_other =  data_dict.intr[id_self], data_dict.intr[id_other]

        pose_w2c_self = torch.eye(4).to(data_dict.poses_w2c.device)
        pose_w2c_self[:3, :4] = data_dict.poses_w2c[id_self] # the pose itself is just (3, 4)
        pose_w2c_other = torch.eye(4).to(data_dict.poses_w2c.device)
        pose_w2c_other[:3, :4] = data_dict.poses_w2c[id_other]

        # in case there are too many values, subsamples
        if pixels_in_self.shape[0] > self.opt.nerf.rand_rays // 2:
            random_values = torch.randperm(pixels_in_self.shape[0],device=self.device)[:self.opt.nerf.rand_rays//2]
            pixels_in_self = pixels_in_self[random_values]
            pixels_in_other = pixels_in_other[random_values]
            conf_values = conf_values[random_values]
        
        rets = self.net.render_image_at_specific_pose_and_rays(self.opt, data_dict,
                                                               torch.stack([pose_w2c_self[:3], pose_w2c_other[:3]]),
                                                                torch.stack([intr_self,intr_other]), H, W,
                                                                pixels=torch.stack([pixels_in_self,pixels_in_other]),
                                                                mode='train', iter=iteration)


        # compute the correspondence loss
        # for each image, project the pixel to the other image according to current estimate of pose 
        depth_rendered_self = rets.depth[0].squeeze(-1)
        depth_rendered_other = rets.depth[1].squeeze(-1)
        # ret_self.depth is [1, N_ray, 1], it needs to be [N_ray]
        stats_dict['depth_in_corr_loss'] = depth_rendered_self.detach().mean()

        T_self2other = pose_w2c_other @ pose_inverse_4x4(pose_w2c_self)

        # [N_ray, 2] and [N_ray]
        loss_corres, stats_dict = self.compute_render_and_repro_loss_w_repro_thres\
            (self.opt, pixels_in_self, depth_rendered_self, intr_self, 
             pixels_in_other, depth_rendered_other, intr_other, T_self2other, 
             conf_values, stats_dict)


        loss_corres_, stats_dict = self.compute_render_and_repro_loss_w_repro_thres\
            (self.opt, pixels_in_other, depth_rendered_other, intr_other, 
             pixels_in_self, depth_rendered_self, intr_self, pose_inverse_4x4(T_self2other), 
             conf_values, stats_dict)  
        loss_corres += loss_corres_

        if 'depth_fine' in rets.keys():
            depth_rendered_self = rets.depth_fine[0].squeeze(-1)
            depth_rendered_other = rets.depth_fine[1].squeeze(-1)

            loss_corres_, stats_dict = self.compute_render_and_repro_loss_w_repro_thres\
                (self.opt, pixels_in_self, depth_rendered_self, intr_self, 
                pixels_in_other, depth_rendered_other, intr_other, T_self2other, 
                conf_values, stats_dict)
            loss_corres += loss_corres_

            loss_corres_, stats_dict = self.compute_render_and_repro_loss_w_repro_thres\
                (self.opt, pixels_in_other, depth_rendered_other, intr_other, 
                pixels_in_self, depth_rendered_self, intr_self, pose_inverse_4x4(T_self2other), 
                conf_values, stats_dict)  
            loss_corres += loss_corres_
        loss_corres = loss_corres / 4. if 'depth_fine' in rets.keys() else loss_corres / 2.
        loss_dict['corres'] = loss_corres
        return loss_dict, stats_dict, plotting_dict



