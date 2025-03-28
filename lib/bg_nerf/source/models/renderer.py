import os
import os.path as osp
from typing import List, Union, Any, Tuple, Dict

import imageio
import lpips
import numpy as np
import torch
import torchvision.transforms.functional as torchvision_F
import tqdm
from easydict import EasyDict as edict

from eval import prealign_w2c_small_camera_systems, prealign_w2c_large_camera_systems, evaluate_camera_alignment
from lib.bg_nerf.source.datasets.rendering_path import generate_spiral_path, generate_spiral_path_dtu
from lib.bg_nerf.source.models.frequency_nerf import FrequencyEmbedder, NeRF
from lib.bg_nerf.source.training.core.loss_factory import define_loss
from lib.bg_nerf.source.training.core.metrics import compute_metrics, compute_depth_error_on_rays, compute_mse_on_rays
from lib.bg_nerf.source.training.core.metrics import compute_metrics_masked
from lib.bg_nerf.source.training.core.sampling_strategies import RaySamplingStrategy
from lib.bg_nerf.source.utils import camera
from lib.bg_nerf.source.utils.geometry.align_trajectories import \
    (backtrack_from_aligning_and_scaling_to_first_cam, backtrack_from_aligning_the_trajectory)
from lib.bg_nerf.source.utils.summary_board import SummaryBoard
from lib.bg_nerf.source.utils.timer import Timer
from lib.bg_nerf.source.utils.torch import get_log_string
from lib.bg_nerf.source.utils.torch import release_cuda, to_cuda
from lib.bg_nerf.source.utils.vis_rendering import colorize_np, img_HWC2CHW
from lib.bg_nerf.third_party.pytorch_ssim.ssim import ssim as ssim_loss


class Graph(torch.nn.Module):
    """NeRF model (mlp prediction + rendering). """
    def __init__(self, opt: Dict[str, Any], device: torch.device):
        super().__init__()
        self.opt = opt
        self.device = device
        self.define_renderer(opt)
        self.data_dict = edict()
        self.iteration_nerf = 0
        self.inner_iteration = 0
        self.best_val = float("Inf")  # absolute best val, to know when to save the checkpoint.
        self.epoch_of_best_val = 0
        self.epoch=0
        self.current_best_val = 0
        self.lpips_loss = lpips.LPIPS(net="alex").to(self.device)

    def define_renderer(self, opt: Dict[str, Any]):
        self.nerf = NeRF(opt).to(self.device)
        if opt.nerf.fine_sampling:
            self.nerf_fine = NeRF(opt, is_fine_network=True).to(self.device)
        self.embedder_pts = FrequencyEmbedder(self.opt)
        self.embedder_view = FrequencyEmbedder(self.opt)
        return

    def re_initialize(self):
        self.nerf.initialize()
        if self.opt.nerf.fine_sampling:
            self.nerf_fine.initialize()

    def get_network_components(self):
        ret = [self.nerf]
        if self.opt.nerf.fine_sampling:
            ret.append(self.nerf_fine)
        return ret

    def L1_loss(self, pred: torch.Tensor,label: torch.Tensor):
        loss = (pred.contiguous()-label).abs()
        return loss.mean()

    def MSE_loss(self, pred: torch.Tensor, label: torch.Tensor, mask: torch.Tensor = None):
        loss = (pred.contiguous()-label)**2
        if mask is not None:
            loss = loss[mask]
        return loss.mean()



    def define_loss_module(self, coord0_scene, coord1_scene, mconf_scene, vgg_features=None):
        flow_net = self.flow_net if self.opt.use_flow else None
        self.loss_module = define_loss(self.opt.loss_type, self.opt, self, self.data_dict,
                                       self.device, flow_net=flow_net)
        self.data_dict.coord0_scene, self.data_dict.coord1_scene, self.data_dict.mconf_scene = coord0_scene, coord1_scene, mconf_scene,
        self.data_dict.vgg_features =vgg_features



    def load_dataset(self,idx, image, depth_range, intr):
        self.data_dict['idx'] =  idx
        self.data_dict['image'] =  image
        self.data_dict['depth_range'] =  depth_range
        self.data_dict['intr'] =  intr
        # pixels/ray sampler
        self.sampling_strategy = RaySamplingStrategy(self.opt, data_dict=self.data_dict, device=self.device)



    def get_pose(self, opt: Dict[str, Any], data_dict: Dict[str, Any], mode: str = None):
        return self.get_w2c_pose(opt, data_dict, mode)

    def get_c2w_pose(self, opt: Dict[str, Any], data_dict: Dict[str, Any], mode: str = None):
        return camera.pose.invert(self.get_w2c_pose(opt, data_dict, mode))

    def forward(self,opt: Dict[str, Any], data_dict: Dict[str, Any], iter: int,
                img_idx: Union[List[int], int] = None, mode: str = None) -> Dict[str, Any]:
        """Rendering of a random subset of pixels or all pixels for all images of the data_dict.

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
            iter (int): iteration
            img_idx (list, optional): In case we want to render from only a subset of the
                                      images contained in data_dict.image. Defaults to None.
            mode (str, optional): Defaults to None.
        """
        batch_size = len(data_dict.idx)
        pose = self.get_w2c_pose(opt,data_dict,mode=mode) # current w2c pose estimates
        # same than data_dict.poses_w2c

        # render images
        H, W = data_dict.image.shape[-2:]

        if opt.nerf.depth.param == 'inverse':
            depth_range = opt.nerf.depth.range
        else:
            # use the one from the dataset
            depth_range = data_dict.depth_range[0]

        if img_idx is not None:
            # we only render some images of the batch, not all of them
            ray_idx = None
            nbr_img = len(img_idx) if isinstance(img_idx, list) else 1
            if opt.nerf.rand_rays and mode in ["train","test-optim"]:
                # sample random rays for optimization
                ray_idx = torch.randperm(H*W,device=self.device)[:opt.nerf.rand_rays//nbr_img]  # the same rays are rendered for all the images here
            ret = self.render_image_at_specific_rays(opt, data_dict, img_idx, ray_idx=ray_idx)
            if ray_idx is not None:
                ret.ray_idx = ray_idx
            ret.idx_img_rendered = img_idx
            return ret

        # we render all images
        if opt.nerf.rand_rays and mode in ["train","test-optim"]:
            # sample random rays for optimization
            ray_idx = torch.randperm(H*W,device=self.device)[:opt.nerf.rand_rays//batch_size]
            # the same rays are rendered for all the images here
            ret = self.render(opt,pose,intr=data_dict.intr,ray_idx=ray_idx,mode=mode, H=H,
                              W=W, depth_range=depth_range, iter=iter) # [B,N,3],[B,N,1]
            ret.ray_idx = ray_idx
        else:
            # render full image (process in slices)
            ret = self.render_by_slices(opt,pose,intr=data_dict.intr,mode=mode,
                                        H=H, W=W, depth_range=depth_range, iter=iter) if opt.nerf.rand_rays else \
                  self.render(opt,pose,intr=data_dict.intr,mode=mode, H=H, W=W,
                              depth_range=depth_range, iter=iter) # [B,HW,3],[B,HW,1]

        ret.idx_img_rendered = torch.arange(start=0, end=batch_size).to(self.device)
        # all the images were rendered, corresponds to their index in the tensor
        return ret


    @torch.no_grad()
    def visualize(self,opt: Dict[str, Any],data_dict: Dict[str, Any],output_dict: Dict[str, Any],
                  step: int=0, split: str="train",eps: float=1e-10
                  ) -> Dict[str, Any]:
        """Creates visualization of renderings and gt. Here N is HW

        Attention:
            ground-truth image has shape (B, 3, H, W)
            rgb rendering has shape (B, H*W, 3)
        Args:
            opt (edict): settings
            data_dict (edict): Input data dict. Contains important fields:
                           - Image: GT images, (B, 3, H, W)
                           - intr: intrinsics (B, 3, 3)
                           - idx: idx of the images (B)
                           - depth_gt (optional): gt depth, (B, 1, H, W)
                           - valid_depth_gt (optional): (B, 1, H, W)
            output_dict (edict): Output dict from the renderer. Contains important fields
                             - idx_img_rendered: idx of the images rendered (B), useful
                             in case you only did rendering of a subset
                             - ray_idx: idx of the rays rendered, either (B, N) or (N)
                             - rgb: rendered rgb at rays, shape (B, N, 3)
                             - depth: rendered depth at rays, shape (B, N, 1)
                             - rgb_fine: rendered rgb at rays from fine MLP, if applicable, shape (B, N, 3)
                             - depth_fine: rendered depth at rays from fine MLP, if applicable, shape (B, N, 1)
            step (int, optional): Defaults to 0.
            split (str, optional): Defaults to "train".
        """

        plotting_stats = {}
        to_plot, to_plot_fine = [], []

        # compute scaling factor for the depth, if the poses are optimized
        scaling_factor_for_pred_depth = 1.
        if self.settings.model == 'joint_pose_nerf_training' and hasattr(self, 'sim3_est_to_gt_c2w'):
            # adjust the rendered depth, since the optimized scene geometry and poses are valid up to a 3D
            # similarity, compared to the ground-truth.
            scaling_factor_for_pred_depth = (self.sim3_est_to_gt_c2w.trans_scaling_after * self.sim3_est_to_gt_c2w.s) \
                if self.sim3_est_to_gt_c2w.type == 'align_to_first' else self.sim3_est_to_gt_c2w.s

        H, W = data_dict.image.shape[-2:]
        # cannot visualize if it is not rendering the full image!
        depth_map = output_dict.depth.view(-1,H,W,1)[0] * scaling_factor_for_pred_depth
        # [B,H,W, 1] and then (H, W, 1)
        depth_map_var = output_dict.depth_var.view(-1,H,W,1)[0] # [B,H,W, 1] and then (H, W, 1)
        # invdepth = (1-var.depth)/var.opacity if opt.camera.ndc else 1/(var.depth/var.opacity+eps)
        # invdepth_map = invdepth.view(-1,opt.H,opt.W,1)[0] # [B,H,W, 1] and then (H, W, 1)
        rgb_map = output_dict.rgb.view(-1,H,W,3)[0] # [B,H,W, 3] and then (H, W, 3)
        rgb_map_var = output_dict.rgb_var.view(-1, H, W, 1)[0]  # (H, W, 1)
        idx_img_rendered = output_dict.idx_img_rendered[0]

        opacity = output_dict.opacity.view(-1, H, W, 1)[0]

        image = (data_dict.image.permute(0, 2, 3, 1)[idx_img_rendered].cpu().numpy() * 255.).astype(np.uint8) # (B, 3, H, W), then (H, W, 3)
        depth_range = None
        if hasattr(data_dict, 'depth_range') and opt.nerf.depth.param == 'metric':
            depth_range = data_dict.depth_range[idx_img_rendered].cpu().numpy().tolist()
            depth_range[0] = min(depth_range[0], depth_map.min().item())
            depth_range[1] = max(depth_range[1], depth_map.max().item())

        fine_pred_rgb_np_uint8 = (255 * np.clip(rgb_map.cpu().numpy(), a_min=0, a_max=1.)).astype(np.uint8)

        # apperance color error
        image_rgb_map_error = np.sum((image/255. - fine_pred_rgb_np_uint8/255.)**2, axis=-1)
        image_rgb_map_error = (colorize_np(image_rgb_map_error, range=(0, 1)) * 255.).astype(np.uint8)

        pred_image_var_colored = colorize_np(rgb_map_var.cpu().squeeze().numpy())
        pred_image_var_colored = (255 * pred_image_var_colored).astype(np.uint8)

        fine_pred_depth_colored = colorize_np(depth_map.cpu().squeeze().numpy(), range=depth_range)
        fine_pred_depth_colored = (255 * fine_pred_depth_colored).astype(np.uint8)

        pred_depth_var_colored = colorize_np(depth_map_var.cpu().squeeze().numpy())
        pred_depth_var_colored = (255 * pred_depth_var_colored).astype(np.uint8)

        opacity = (opacity.cpu().squeeze().unsqueeze(-1).repeat(1, 1, 3).numpy() * 255).astype(np.uint8)
        to_plot += [torch.from_numpy(x.astype(np.float32)/255.) for x in
                    [image, fine_pred_rgb_np_uint8, fine_pred_depth_colored, opacity,
                    pred_image_var_colored, pred_depth_var_colored, image_rgb_map_error]]

        name = 'gt-predc-depthc-acc-rgbvarc-depthvarc-err'

        if 'depth_fine' in output_dict:
            depth_map = output_dict.depth_fine.view(-1,H,W,1)[0] * scaling_factor_for_pred_depth
            # [B,H,W, 1] and then (H, W, 1)
            depth_map_var = output_dict.depth_var_fine.view(-1,H,W,1)[0] # [B,H,W, 1] and then (H, W, 1)
            # invdepth = (1-var.depth_fine)/var.opacity_fine if opt.camera.ndc else 1/(var.depth_fine/var.opacity_fine+eps)
            # invdepth_map = invdepth.view(-1,opt.H,opt.W,1)[0]
            rgb_map = output_dict.rgb_fine.view(-1,H,W,3)[0]
            rgb_map_var = output_dict.rgb_var_fine.view(-1, H, W, 1)[0]  # (H, W, 1)

            opacity = output_dict.opacity_fine.view(-1, H, W, 1)[0]

            fine_pred_rgb_np_uint8 = (255 * np.clip(rgb_map.cpu().numpy(), a_min=0, a_max=1.)).astype(np.uint8)

            # apperance color error
            image_rgb_map_error = np.sum((image/255. - fine_pred_rgb_np_uint8/255.)**2, axis=-1)
            image_rgb_map_error = (colorize_np(image_rgb_map_error, range=(0, 1)) * 255.).astype(np.uint8)

            pred_image_var_colored = colorize_np(rgb_map_var.cpu().squeeze().numpy())
            pred_image_var_colored = (255 * pred_image_var_colored).astype(np.uint8)

            fine_pred_depth_colored = colorize_np(depth_map.cpu().squeeze().numpy(), range=depth_range)
            fine_pred_depth_colored = (255 * fine_pred_depth_colored).astype(np.uint8)

            pred_depth_var_colored = colorize_np(depth_map_var.cpu().squeeze().numpy())
            pred_depth_var_colored = (255 * pred_depth_var_colored).astype(np.uint8)

            opacity = (opacity.cpu().squeeze().unsqueeze(-1).repeat(1, 1, 3).numpy() * 255).astype(np.uint8)
            to_plot_fine += [torch.from_numpy(x.astype(np.float32)/255.) for x in  \
                [image, fine_pred_rgb_np_uint8, fine_pred_depth_colored, pred_depth_var_colored, opacity,
                pred_image_var_colored, image_rgb_map_error]]

        if 'depth_gt' in data_dict.keys():
            depth_gt = data_dict.depth_gt[idx_img_rendered]
            depth_gt_colored = (255 * colorize_np(depth_gt.cpu().squeeze().numpy(), range=depth_range)).astype(np.uint8)
            to_plot += [torch.from_numpy(depth_gt_colored.astype(np.float32)/255.)]
            if len(to_plot_fine) > 0:
                to_plot_fine += [torch.from_numpy(depth_gt_colored.astype(np.float32)/255.)]
            name += '-depthgt'

        to_plot_img = torch.stack(to_plot, dim=0) # (N, H, W, 3)
        if len(to_plot_fine) > 0:
            to_plot_img = torch.cat((to_plot_img, torch.stack(to_plot_fine, dim=0)), dim=1) # (N, 2H, W, 3)
        to_plot = img_HWC2CHW(to_plot_img)  # (N, 3, 2H, W)
        plotting_stats[f'{split}_{step}_{name}'] = to_plot
        return plotting_stats

    @torch.no_grad()
    def make_result_dict(self, opt: Dict[str, Any], data_dict: Dict[str, Any], output_dict: Dict[str, Any],
                         loss: Dict[str, Any], metric: Dict[str, Any] = None,
                         split: str = 'train') -> Dict[str, Any]:
        """Make logging dict. Corresponds to dictionary which will be saved in tensorboard and also logged"""
        stats_dict = {}
        for key, value in loss.items():
            if key == "all": continue
            stats_dict["loss_{}".format(key)] = value
        if metric is not None:
            for key, value in metric.items():
                stats_dict["{}".format(key)] = value



        # log progress bar
        if hasattr(self.nerf, 'progress'):
            stats_dict['cf_pe_progress'] = self.nerf.progress.data.item()

        # log PSNR
        if 'mse' in output_dict.keys():
            # compute PSNR
            psnr = -10 * output_dict['mse'].log10()
            stats_dict['PSNR'] = psnr
            if 'mse_fine' in output_dict.keys() and output_dict['mse_fine'] is not None:
                psnr = -10 * output_dict['mse_fine'].log10()
                stats_dict['PSNR_fine'] = psnr

        # if depth is available, compute depth error
        if 'depth_gt' in data_dict.keys() and 'depth' in output_dict.keys():
            scaling_factor_for_pred_depth = 1.
            if self.settings.model == 'joint_pose_nerf_training' and \
                    hasattr(self, 'sim3_est_to_gt_c2w'):
                # adjust the scaling of the depth since the optimized scene geometry + poses
                # are all valid up to a 3D similarity
                scaling_factor_for_pred_depth = (
                            self.sim3_est_to_gt_c2w.trans_scaling_after * self.sim3_est_to_gt_c2w.s) \
                    if self.sim3_est_to_gt_c2w.type == 'align_to_first' else self.sim3_est_to_gt_c2w.s

            abs_e, rmse = compute_depth_error_on_rays(data_dict, output_dict, output_dict.depth,
                                                      scaling_factor_for_pred_depth=scaling_factor_for_pred_depth)
            stats_dict['depth_abs'] = abs_e
            stats_dict['depth_rmse'] = rmse
            if 'depth_fine' in output_dict.keys():
                abs_e, rmse = compute_depth_error_on_rays(data_dict, output_dict, output_dict.depth_fine,
                                                          scaling_factor_for_pred_depth=scaling_factor_for_pred_depth)
                stats_dict['depth_abs_fine'] = abs_e
                stats_dict['depth_rmse_fine'] = rmse
        return stats_dict

    def train_step(self, iteration: int
                   ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Forward pass of the training step. Loss computation
        Args:
            iteration (int):
            data_dict (edict): Scene data. dict_keys(['idx', 'image', 'intr', 'pose'])
                                image: tensor of images with shape (B, 3, H, W)
                                intr: tensor of intrinsic matrices of shape (B, 3, 3)
                                pose: tensor of ground-truth w2c poses of shape (B, 3, 4)
                                idx: indices of images, shape is (B)

        Returns:
            output_dict: output from the renderer
            results_dict: dict to log, also contains the loss for back-propagation
            plotting_dict: dict with figures to log to tensorboard.
        """
        plot = False
        if iteration % self.settings.vis_steps == 0:
            plot = True

        # sample rays from all images, and render depth and rgb from those viewpoint directions.
        rays = self.sampling_strategy(self.settings.nerf.rand_rays,
                                      sample_in_center=iteration < self.settings.precrop_iters)
        output_dict = self.render_image_at_specific_rays(self.settings, self.data_dict,
                                                             ray_idx=rays, iter=iteration, mode="train")

        loss_dict, stats_dict, plotting_dict = self.loss_module.compute_loss \
            (self.settings, self.data_dict, output_dict, mode="train", plot=plot, iteration=iteration)
        # metrics for logging
        output_dict['mse'], output_dict['mse_fine'] = compute_mse_on_rays(self.data_dict, output_dict)
        results_dict = self.make_result_dict(self.settings, self.data_dict, output_dict, loss_dict)
        results_dict['avg_pred_depth'] = output_dict.depth.detach().mean()
        results_dict.update(stats_dict)
        results_dict['loss'] = loss_dict['all']  # the actual loss, used to back-propagate

        if plot:
            # render the full image
            with torch.no_grad():
                B = self.data_dict.image.shape[0]
                self.eval()
                img_idx = np.random.randint(B)
                output_dict = self.render_image_at_specific_rays \
                    (self.settings, self.data_dict, img_idx=img_idx, iter=iteration, mode="train")
                # will render the full image
                plotting_dict_ = self.visualize(self.settings, self.data_dict, output_dict,
                                                split='train')
                self.train()
                plotting_dict.update(plotting_dict_)

        # update the progress bar
        if hasattr(self.nerf, 'progress') and self.settings.barf_c2f is not None and self.settings.apply_cf_pe:
            # will update progress here
            self.nerf.progress.data.fill_(self.iteration_nerf / self.settings.max_iter)
            if self.settings.nerf.fine_sampling and hasattr(self.nerf_fine, 'progress'):
                self.nerf_fine.progress.data.fill_(self.iteration_nerf / self.settings.max_iter)

        return output_dict, results_dict, plotting_dict

    def train_iteration_nerf_pose_flow(self):
        """ Run one iteration of training
                The nerf mlp is optimized
                The poses are also potentially optimized
                """
        self.iteration_nerf += 1

        # forward
        output_dict, result_dict, plotting_dict = self.train_step(self.iteration)

        # # backward & optimization
        # self.update_parameters(result_dict['loss'])
        return output_dict, result_dict, plotting_dict

    def train_iteration(self, iteration):
        self.iteration = iteration
        output_dict, result_dict, plotting_dict = self.train_iteration_nerf_pose_flow()
        return output_dict, result_dict, plotting_dict
    def render_image_at_specific_pose_and_rays(self, opt: Dict[str, Any], data_dict: Dict[str, Any], pose: torch.Tensor,
                                               intr: torch.Tensor, H: int, W: int, iter: int,
                                               pixels: torch.Tensor = None, ray_idx: torch.Tensor = None,
                                               mode: str = 'train') -> Dict[str, Any]:
        """Rendering of a specified set of pixels (or all) at a predefined pose.

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
            pose: w2c poses at which to render (L, 3, 4) or (3, 4)
            intr: corresponding intr matrices (L, 3, 3) or (3, 3)
            H, W: size of rendered image
            iter (int): iteration
            pixels, ray_idx: if any of the two is specified, will render only at these locations.
                             (L, N, 2) or (N, 2)  / (L, N) or (N)
            mode (str, optional): Defaults to None.
        """
        if len(pose.shape) == 2:
            pose = pose.unsqueeze(0)
        if len(intr.shape) == 2:
            intr = intr.unsqueeze(0)

        if opt.nerf.depth.param == 'inverse':
            depth_range = opt.nerf.depth.range
        else:
            # use the one from the dataset
            depth_range = data_dict.depth_range[0]

        if ray_idx is None and pixels is None:
            # rendere the full image
            ret = self.render_by_slices(opt,pose,intr=intr,mode=mode, H=H, W=W, \
                depth_range=depth_range, iter=iter) if opt.nerf.rand_rays else \
                  self.render(opt,pose,intr=intr,mode=mode, H=H, W=W, \
                      depth_range=depth_range, iter=iter) # [B,HW,3],[B,HW,1]
        else:
            # render only the ray_idx or pixels specified.
            ret = self.render(opt,pose,intr=intr,pixels=pixels,ray_idx=ray_idx,mode=mode,
                              H=H, W=W, depth_range=depth_range, iter=iter) # [B,N,3],[B,N,1]
            ret.ray_idx = ray_idx
        # ret.update(rgb_fine=rgb_fine,depth_fine=depth_fine,opacity_fine=opacity_fine) # [1, N_rays, K] or [N, N_rays, K]
        return ret

    def render_image_at_specific_rays(self, opt: Dict[str, Any], data_dict: Dict[str, Any], iter: int,
                                      img_idx: Union[List[int], int] = None, pixels: torch.Tensor = None,
                                      ray_idx: torch.Tensor = None, mode: str = 'train') -> Dict[str, Any]:
        """Rendering of a specified set of pixels for all images of data_dict.

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
            iter (int): iteration
            pixels, ray_idx: if any of the two is specified, will render only at these locations.
                             (L, N, 2) or (N, 2)  / (L, N) or (N).
                             where L is len(img_idx) is img_idx is not None else the number of images in the batch, ie L=B
            img_idx (list, optional): In case we want to render from only a subset of the
                                      images contained in data_dict.image. Defaults to None.
            mode (str, optional): Defaults to None.
        """
        pose = self.get_w2c_pose(opt,data_dict,mode=mode)
        intr = data_dict.intr
        batch_size = pose.shape[0]

        if img_idx is not None:
            if isinstance(img_idx, (tuple, list)):
                N = len(img_idx)
                pose = pose[img_idx].view(-1, 3, 4)  # (N, 3, 4)
                intr = intr[img_idx].view(-1, 3, 3)  # (N, 3, 3)
            else:
                pose = pose[img_idx].unsqueeze(0)  # (1, 3, 4)
                intr = intr[img_idx].unsqueeze(0)  # (1, 3, 3)
                img_idx = [img_idx]  # make it a list of a single element
        H, W = data_dict.image.shape[-2:]

        if opt.nerf.depth.param == 'inverse':
            depth_range = opt.nerf.depth.range
        else:
            # use the one from the dataset
            depth_range = data_dict.depth_range[0]

        if ray_idx is None and pixels is None:
            # rendere the full image
            ret = self.render_by_slices(opt,pose,intr=intr,mode=mode, H=H, W=W, depth_range=depth_range, iter=iter) if opt.nerf.rand_rays else \
                  self.render(opt,pose,intr=intr,mode=mode, H=H, W=W, depth_range=depth_range, iter=iter) # [B,HW,3],[B,HW,1]
        else:
            ret = self.render(opt,pose,intr=intr,pixels=pixels,ray_idx=ray_idx,mode=mode,
                              H=H, W=W, depth_range=depth_range, iter=iter) # [B,N,3],[B,N,1]
            ret.ray_idx = ray_idx
        # ret.update(rgb_fine=rgb_fine,depth_fine=depth_fine,opacity_fine=opacity_fine) # [1, N_rays, K] or [N, N_rays, K]
        ret.idx_img_rendered = torch.from_numpy(np.array(img_idx)).to(self.device) if img_idx is not None else \
            torch.arange(start=0, end=batch_size).to(self.device)
        return ret

    def render(self, opt: Dict[str, Any], pose: torch.Tensor, H: int, W: int, intr: torch.Tensor, pixels: torch.Tensor = None,
               ray_idx: torch.Tensor = None, depth_range: List[float] = None, iter: int = None, mode: int = None) -> Dict[str, Any]:
        """Main rendering function

        Args:
            opt (edict): settings
            pose (torch.Tensor): w2c poses at which to render (L, 3, 4) or (3, 4)
            intr (torch.Tensor): corresponding intr matrices (L, 3, 3) or (3, 3)
            H, W (int, int): size of rendered image
            depth_range (list): Min and max value for the depth sampling along the ray.
                                The same for all rays here.
            iter (int): iteration
            pixels, ray_idx  (torch.Tensor): if any of the two is specified, will render only at these locations.
                             (L, N, 2) or (N, 2)  / (L, N) or (N)
            mode (str, optional): Defaults to None.
        """
        batch_size = len(pose)

        if pixels is not None:
            center, ray = camera.get_center_and_ray_at_pixels(pose, pixels, intr=intr)
            # [B, N_rays, 3] and [B, N_rays, 3]
        else:
            # from ray_idx
            center,ray = camera.get_center_and_ray(pose, H, W, intr=intr) # [B,HW,3]
            while ray.isnan().any(): # TODO: weird bug, ray becomes NaN arbitrarily if batch_size>1, not deterministic reproducible
                center,ray = camera.get_center_and_ray(pose, H, W,intr=intr) # [B,HW,3]

            if ray_idx is not None:
                if len(ray_idx.shape) == 2 and ray_idx.shape[0] == batch_size:
                    n_rays = ray_idx.shape[-1]
                    #ray_idx is B, N-rays, ie different ray indices for each image in batch
                    batch_idx_flattened = torch.arange(start=0, end=batch_size).unsqueeze(-1).repeat(1, n_rays).long().view(-1)
                    ray_idx_flattened = ray_idx.reshape(-1).long()

                    # consider only subset of rays
                    center,ray = center[batch_idx_flattened, ray_idx_flattened],ray[batch_idx_flattened, ray_idx_flattened]  # [B*N_rays, 3]
                    center = center.reshape(batch_size, n_rays, 3)
                    ray = ray.reshape(batch_size, n_rays, 3)
                else:
                    # consider only subset of rays
                    # ray_idx is (-1)
                    center,ray = center[:,ray_idx],ray[:,ray_idx]  # [B, N_rays, 3]

        if opt.camera.ndc:
            # convert center/ray representations to NDC
            center,ray = camera.convert_NDC(opt,center,ray,intr=intr)
        # render with main MLP

        pred = edict(origins=center, viewdirs=ray)

        # opt.nerf.sample_stratified means randomness in depth vlaues sampled
        depth_samples = self.sample_depth(opt,batch_size,num_rays=ray.shape[1],
                                            n_samples=opt.nerf.sample_intvs, H=H, W=W,
                                            depth_range=depth_range, mode=mode) # [B,HW,N,1] or [B, N_rays, N, 1]
        pred_coarse = self.nerf.forward_samples(opt,center,ray,depth_samples,
                                                embedder_pts=self.embedder_pts,
                                                embedder_view=self.embedder_view, mode=mode)
        pred_coarse['t'] = depth_samples
        # contains rgb_samples [B, N_rays, N, 3] and density_samples [B, N_rays, N]
        pred_coarse = self.nerf.composite(opt,ray,pred_coarse,depth_samples)
        # new elements are rgb,depth,depth_var, opacity,weights,loss_weight_sparsity
        # [B,HW,K] or [B, N_rays, K], loss_weight_sparsity is [B]
        weights = pred_coarse['weights']

        pred.update(pred_coarse)

        # render with fine MLP from coarse MLP
        compute_fine_sampling = True
        if hasattr(opt.nerf, 'ratio_start_fine_sampling_at_x') and opt.nerf.ratio_start_fine_sampling_at_x is not None \
            and iter is not None and iter < opt.max_iter * opt.nerf.ratio_start_fine_sampling_at_x:
            compute_fine_sampling = False

        if opt.nerf.fine_sampling and compute_fine_sampling:
            with torch.no_grad():
                # resample depth acoording to coarse empirical distribution
                # weights are [B, num_rays, Nf, 1]
                det = mode not in ['train', 'test-optim'] or (not opt.nerf.sample_stratified)
                depth_samples_fine = self.sample_depth_from_pdf(opt, weights=weights[...,0],
                                                                n_samples_coarse=opt.nerf.sample_intvs,
                                                                n_samples_fine=opt.nerf.sample_intvs_fine,
                                                                depth_range=depth_range, det=det) # [B,HW,Nf,1]
                # print(depth_samples_fine.min(), depth_samples_fine.max())
                depth_samples_fine = depth_samples_fine.detach()

            depth_samples = torch.cat([depth_samples,depth_samples_fine],dim=2) # [B,HW,N+Nf,1]

            depth_samples = depth_samples.sort(dim=2).values
            pred_fine = self.nerf_fine.forward_samples(opt,center,ray,depth_samples,
                                                       embedder_pts=self.embedder_pts,
                                                       embedder_view=self.embedder_view, mode=mode)

            pred_fine['t'] = depth_samples
            pred_fine = self.nerf_fine.composite(opt,ray,pred_fine,depth_samples)
            pred_fine = edict({k+'_fine': v for k, v in pred_fine.items()})
            pred.update(pred_fine)
        return pred

    def render_by_slices(self, opt: Dict[str, Any], pose: torch.Tensor, H: int, W: int,
                         intr: torch.Tensor, depth_range: List[float], iter: int, mode: str = None) -> Dict[str, Any]:
        """Main rendering function for the entire image of size (H, W). By slice because
        two many rays to render in a single forward pass.

        Args:
            opt (edict): settings
            pose (torch.Tensor): w2c poses at which to render (L, 3, 4)
            intr (torch.Tensor): corresponding intr matrices (L, 3, 3)
            H, W (int, int): size of rendered image
            depth_range (list): Min and max value for the depth sampling along the ray.
                                The same for all rays here.
            iter (int): iteration
            mode (str, optional): Defaults to None.
        """
        ret_all = edict(rgb=[],rgb_var=[],depth=[],depth_var=[], opacity=[],
                        normal=[], all_cumulated=[])
        if opt.nerf.fine_sampling:
            if (hasattr(opt.nerf, 'ratio_start_fine_sampling_at_x') and opt.nerf.ratio_start_fine_sampling_at_x is not None \
                and iter is not None and iter < opt.max_iter * opt.nerf.ratio_start_fine_sampling_at_x):
                # we skip the fine sampling
                ret_all.update({})
            else:
                ret_all.update(rgb_fine=[],rgb_var_fine=[], depth_fine=[],
                            depth_var_fine=[], opacity_fine=[], normal_fine=[], all_cumulated_fine=[])
        # render the image by slices for memory considerations
        for c in range(0,H*W,opt.nerf.rand_rays):
            ray_idx = torch.arange(c,min(c+opt.nerf.rand_rays,H*W),device=self.device)
            ret = self.render(opt,pose,H=H, W=W, intr=intr,ray_idx=ray_idx,depth_range=depth_range, iter=iter, mode=mode) # [B,R,3],[B,R,1]
            for k in ret_all:
                if k in ret.keys(): ret_all[k].append(ret[k])
            torch.cuda.empty_cache()
        # group all slices of images
        for k in ret_all: ret_all[k] = torch.cat(ret_all[k],dim=1) if len(ret_all[k]) > 0 else None
        return ret_all

    def sample_depth(self, opt: Dict[str, Any], batch_size: int, n_samples: int, H: int, W: int, depth_range: List[float], \
                     num_rays: int = None, mode: str = None) -> torch.Tensor:
        """Sample depths along ray. The same depth range is applied for all the rays.

        Args:
            opt (edict): settings
            batch_size (int):
            n_samples (int): Number of depth samples along each ray
            H (int): img size
            W (int): img size
            depth_range (list)
            num_rays (int, optional): Defaults to None.
            mode (str, optional): Defaults to None.

        Returns:
            depth_samples (torch.Tesnor), shape # [B,num_rays,n_samples,1]
        """

        depth_min, depth_max = depth_range
        num_rays = num_rays or H*W

        if opt.nerf.sample_stratified and mode not in ['val', 'eval', 'test']:
            # random samples
            rand_samples = torch.rand(batch_size,num_rays,n_samples,1,device=self.device)  # [B,HW,N,1]
        else:
            rand_samples = 0.5 * torch.ones(batch_size,num_rays,n_samples,1,device=self.device)

        rand_samples += torch.arange(n_samples,device=self.device)[None,None,:,None].float() # the added part is [1, 1, N, 1] ==> [B,HW,N,1]
        depth_samples = rand_samples/n_samples*(depth_max-depth_min)+depth_min # [B,HW,N,1]

        depth_samples = dict(
            metric=depth_samples,
            inverse=1/(depth_samples+1e-8),
        )[opt.nerf.depth.param]
        # when inverse depth, Depth range is [1, 0] but we actually take the inverse,
        # so the depth samples are between 1 and a very large number.
        return depth_samples

    def sample_depth_from_pdf(self, opt: Dict[str, Any], weights: torch.Tensor, n_samples_coarse: int,
                              n_samples_fine: int, depth_range: List[float], det: bool) -> torch.Tensor:
        """
        Args:
            weights [B, num_rays, N]: Weights assigned to each sampled color.
        """
        depth_min, depth_max = depth_range

        pdf = weights / (weights.sum(dim=-1, keepdims=True) + 1e-6)
        # get CDF from PDF (along last dimension)
        cdf = pdf.cumsum(dim=-1) # [B,HW,N]
        cdf = torch.cat([torch.zeros_like(cdf[...,:1]),cdf],dim=-1) # [B,HW,N+1]

        # here add either uniform or random
        if det:
            # take uniform samples
            grid = torch.linspace(0,1,n_samples_fine+1,device=self.device) # [Nf+1]
        else:
            grid = torch.rand(n_samples_fine+1).to(self.device)

        # unif corresponds to interval mid points
        unif = 0.5*(grid[:-1]+grid[1:]).repeat(*cdf.shape[:-1],1) # [B,HW,Nf]
        idx = torch.searchsorted(cdf,unif,right=True) # [B,HW,Nf] \in {1...N}

        # inverse transform sampling from CDF
        depth_bin = torch.linspace(depth_min,depth_max,n_samples_coarse+1,device=self.device) # [N+1]
        depth_bin = depth_bin.repeat(*cdf.shape[:-1],1) # [B,HW,N+1]
        depth_low = depth_bin.gather(dim=2,index=(idx-1).clamp(min=0)) # [B,HW,Nf]
        depth_high = depth_bin.gather(dim=2,index=idx.clamp(max=n_samples_coarse)) # [B,HW,Nf]
        cdf_low = cdf.gather(dim=2,index=(idx-1).clamp(min=0)) # [B,HW,Nf]
        cdf_high = cdf.gather(dim=2,index=idx.clamp(max=n_samples_coarse)) # [B,HW,Nf]
        # linear interpolation
        t = (unif-cdf_low)/(cdf_high-cdf_low+1e-8) # [B,HW,Nf]
        depth_samples = depth_low+t*(depth_high-depth_low) # [B,HW,Nf]
        # print(depth_low, depth_high)
        return depth_samples[...,None] # [B,HW,Nf,1]


    #------------ SPECIFIC RENDERING where each ray is rendered up to a max depth (different for each ray) -------
    def render_up_to_maxdepth_at_specific_pose_and_rays\
        (self, opt: Dict[str, Any], data_dict: Dict[str, Any], pose: torch.Tensor, intr: torch.Tensor,
         H: int, W: int, depth_max: torch.Tensor, iter: int, pixels: torch.Tensor = None,
         ray_idx: torch.Tensor = None, mode: str = 'train') -> Dict[str, Any]:
        """Rendering of a specified set of pixels at a predefined pose, up to a specific depth.

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
            pose: w2c poses at which to render (L, 3, 4) or (3, 4)
            intr: corresponding intr matrices (L, 3, 3) or (3, 3)
            H, W: size of rendered image
            depth_max: Max depth to sample for each ray (L, N)
            iter (int): iteration
            pixels, ray_idx: if any of the two is specified, will render only at these locations.
                             (L, N, 2) or (N, 2)  / (L, N) or (N)
            mode (str, optional): Defaults to None.
        """
        if len(pose.shape) == 2:
            pose = pose.unsqueeze(0)
        if len(intr.shape) == 2:
            intr = intr.unsqueeze(0)

        if opt.nerf.depth.param == 'inverse':
            depth_range = opt.nerf.depth.range
        else:
            # use the one from the dataset
            depth_range = data_dict.depth_range[0]

        # max_depth should be [B, N]
        ret = self.render_to_max(opt,pose,intr=intr,pixels=pixels,ray_idx=ray_idx,mode=mode,
                                 H=H, W=W, depth_min=depth_range[0], depth_max=depth_max, iter=iter) # [B,N,3],[B,N,1]
        ret.ray_idx = ray_idx
        # ret.update(rgb_fine=rgb_fine,depth_fine=depth_fine,opacity_fine=opacity_fine) # [1, N_rays, K] or [N, N_rays, K]
        return ret

    def render_to_max(self, opt: Dict[str, Any], pose: torch.Tensor, H: int, W: int, intr: torch.Tensor,
                      pixels: torch.Tensor = None, ray_idx: torch.Tensor = None,
                      depth_max: torch.Tensor = None, depth_min: float = None,
                      iter: int = None, mode: str = None) -> Dict[str, Any]:
        """Rendering function for a specified set of pixels at a predefined pose, up to a specific depth.

        Args:
            opt (edict): settings
            pose: w2c poses at which to render (L, 3, 4) or (3, 4)
            intr: corresponding intr matrices (L, 3, 3) or (3, 3)
            H, W: size of rendered image
            depth_max: A different depth max for each pixels, (L, N) or (N)
            depth_min: Min depth
            iter (int): iteration
            pixels, ray_idx: if any of the two is specified, will render only at these locations.
                             (L, N, 2) or (N, 2)  / (L, N) or (N)
            mode (str, optional): Defaults to None.
        """
        batch_size = len(pose)

        if pixels is not None:
            center, ray = camera.get_center_and_ray_at_pixels(pose, pixels, intr=intr)
            # [B, N_rays, 3] and [B, N_rays, 3]
        else:
            # from ray_idx
            center,ray = camera.get_center_and_ray(pose, H, W, intr=intr) # [B,HW,3]
            while ray.isnan().any(): # TODO: weird bug, ray becomes NaN arbitrarily if batch_size>1, not deterministic reproducible
                center,ray = camera.get_center_and_ray(pose, H, W,intr=intr) # [B,HW,3]

            if ray_idx is not None:
                if len(ray_idx.shape) == 2 and ray_idx.shape[0] == batch_size:
                    n_rays = ray_idx.shape[-1]
                    #ray_idx is B, N-rays, ie different ray indices for each image in batch
                    batch_idx_flattened = torch.arange(start=0, end=batch_size).unsqueeze(-1).repeat(1, n_rays).long().view(-1)
                    ray_idx_flattened = ray_idx.reshape(-1).long()

                    # consider only subset of rays
                    center,ray = center[batch_idx_flattened, ray_idx_flattened],ray[batch_idx_flattened, ray_idx_flattened]  # [B*N_rays, 3]
                    center = center.reshape(batch_size, n_rays, 3)
                    ray = ray.reshape(batch_size, n_rays, 3)
                else:
                    # consider only subset of rays
                    # ray_idx is (-1)
                    center,ray = center[:,ray_idx],ray[:,ray_idx]  # [B, N_rays, 3]

        if opt.camera.ndc:
            # convert center/ray representations to NDC
            center,ray = camera.convert_NDC(opt,center,ray,intr=intr)
        # render with main MLP

        pred = edict(origins=center, viewdirs=ray)

        # can only be in metric depth
        depth_samples = self.sample_depth_diff_max_range_per_ray(opt,batch_size,num_rays=ray.shape[1],
                                                                 n_samples=opt.nerf.sample_intvs, H=H, W=W,
                                                                 depth_max=depth_max, depth_min=depth_min,
                                                                 mode=mode) # [B,HW,N,1] or [B, N_rays, N, 1]

        pred_coarse = self.nerf.forward_samples(opt,center,ray,depth_samples,
                                                embedder_pts=self.embedder_pts,
                                                embedder_view=self.embedder_view, mode=mode)
        pred_coarse['t'] = depth_samples
        # contains rgb_samples [B, N_rays, N, 3] and density_samples [B, N_rays, N]
        pred_coarse = self.nerf.composite(opt,ray,pred_coarse,depth_samples)
        # new elements are rgb,depth,depth_var, opacity,weights,loss_weight_sparsity
        # [B,HW,K] or [B, N_rays, K], loss_weight_sparsity is [B]
        weights = pred_coarse['weights']

        pred.update(pred_coarse)

        compute_fine_sampling = True
        # render with fine MLP from coarse MLP
        if hasattr(opt.nerf, 'ratio_start_fine_sampling_at_x') and opt.nerf.ratio_start_fine_sampling_at_x is not None \
            and iter is not None and iter < opt.max_iter * opt.nerf.ratio_start_fine_sampling_at_x:
            compute_fine_sampling = False
        elif hasattr(opt.nerf, 'start_fine_sampling_at_x') and opt.nerf.start_fine_sampling_at_x is not None \
            and iter is not None and iter < opt.nerf.start_fine_sampling_at_x:
            compute_fine_sampling = False

        if opt.nerf.fine_sampling and compute_fine_sampling:
            # we use the same samples to compute opacity
            pred_fine = self.nerf_fine.forward_samples(opt,center,ray,depth_samples,
                                                       embedder_pts=self.embedder_pts,
                                                       embedder_view=self.embedder_view, mode=mode)

            pred_fine['t'] = depth_samples
            pred_fine = self.nerf_fine.composite(opt,ray,pred_fine,depth_samples)
            pred_fine = edict({k+'_fine': v for k, v in pred_fine.items()})
            pred.update(pred_fine)
        return pred

    def to_cuda(self, x):
        return to_cuda(x, self.device)

    def sample_depth_diff_max_range_per_ray(self, opt: Dict[str, Any], batch_size: int, n_samples: int, H: int, W: int,
                                            depth_min: float, depth_max: float,
                                            num_rays: int = None, mode: str = None) -> torch.Tensor:
        """Sample depths along ray. A different depth max is applied for each ray.

        Args:
            opt (edict): settings
            batch_size (int):
            n_samples (int): Number of depth samples along each ray
            H (int): img size
            W (int): img size
            depth_min (float)
            depth_max (torch.Tensor): Max depth to sample for each ray, (batch_size, num_rays)
            num_rays (int, optional): Defaults to None.
            mode (str, optional): Defaults to None.

        Returns:
            depth_samples (torch.Tesnor), shape # [B,num_rays,n_samples,1]
        """
        num_rays = num_rays or H*W

        rand_samples = torch.ones(batch_size,num_rays,n_samples,1,device=self.device)

        rand_samples += torch.arange(n_samples,device=self.device)[None,None,:,None].float() # the added part is [1, 1, N, 1] ==> [B,HW,N,1]

        # depth_max is (B, HW)
        depth_samples = rand_samples/n_samples*(depth_max.unsqueeze(-1).unsqueeze(-1)-depth_min)+depth_min # [B,HW,N,1]

        # can only be in metric here!
        return depth_samples

    def get_w2c_pose(self, opt: Dict[str, Any], data_dict: Dict[str, Any],
                     mode: str = None) -> torch.Tensor:
        if mode == "train":
            pose = data_dict.poses_w2c  # get the current estimates of the camera poses, which are optimized
        elif mode in ["val", "eval", "test-optim", "test"]:
            # val is on the validation set
            # eval is during test/actual evaluation at the end
            # align test pose to refined coordinate system (up to sim3)
            assert hasattr(self, 'sim3_est_to_gt_c2w')
            pose_GT_w2c = data_dict.pose
            ssim_est_gt_c2w = self.sim3_est_to_gt_c2w
            if ssim_est_gt_c2w.type == 'align_to_first':
                pose = backtrack_from_aligning_and_scaling_to_first_cam(pose_GT_w2c, ssim_est_gt_c2w)
            elif ssim_est_gt_c2w.type == 'traj_align':
                pose = backtrack_from_aligning_the_trajectory(pose_GT_w2c, ssim_est_gt_c2w)
            else:
                raise ValueError
            # Here, we align the test pose to the poses found during the optimization (otherwise wont be valid)
            # that's pose. And can learn an extra alignement on top
            # additionally factorize the remaining pose imperfection
            if opt.optim.test_photo and mode != "val":
                pose = camera.pose.compose([data_dict.pose_refine_test, pose])
        else:
            raise ValueError
        return pose

    @torch.no_grad()
    def inference(self, pose, pose_GT, val_data, iteration):
        if pose.shape[0] > 9:
            # alignment of the trajectory
            _, self.sim3_est_to_gt_c2w = prealign_w2c_large_camera_systems(pose, pose_GT)
        else:
            # alignment of the first cameras
            _, self.sim3_est_to_gt_c2w = prealign_w2c_small_camera_systems(pose, pose_GT)
        self.eval()
        torch.set_grad_enabled(False)
        torch.backends.cudnn.enabled = False
        summary_board = SummaryBoard(adaptive=True)
        timer = Timer()
        plotting_dict_total = {}
        self.inner_iteration = iteration + 1
        for i in range(val_data['image'].size(0)):
            data_dict = {key: tensor[i:i + 1] for key, tensor in val_data.items()}
            data_dict = self.to_cuda(data_dict)
            timer.add_prepare_time()
            output_dict, result_dict, plotting_dict = self.val_step(i, data_dict)
            timer.add_process_time()
            result_dict = release_cuda(result_dict)
            plotting_dict_total.update(plotting_dict)  # store the plotting at multiple iterations
            summary_board.update_from_result_dict(result_dict)
            torch.cuda.empty_cache()
            if i > 5:
                break
        self.train()
        torch.set_grad_enabled(True)
        torch.backends.cudnn.enabled = True
        summary_dict = summary_board.summary()
        message = '[Val] '+ get_log_string(summary_dict, iteration=self.iteration, timer=timer)
        print(message)
        self.current_best_val = summary_dict['best_value']
        return plotting_dict_total
    @torch.no_grad()
    def val_step(self, iteration: int, data_dict: Dict[str, Any]
                 ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        data_dict = edict(data_dict)
        H, W = data_dict.image.shape[-2:]
        plotting_dict = {}

        output_dict = self.forward(self.settings, data_dict, mode="val", iter=self.iteration)
        # will render the full image
        output_dict['mse'], output_dict['mse_fine'] = compute_mse_on_rays(data_dict, output_dict)

        # to compute the loss:
        # poses_w2c = self.net.get_w2c_pose(self.settings, data_dict, mode='val')
        # data_dict.poses_w2c = poses_w2c
        # loss_dict, stats_dict, plotting_dict = self.loss_module.compute_loss\
        #     (self.settings, data_dict, output_dict, iteration=iteration, mode="val")

        results_dict = self.make_result_dict(self.settings, data_dict, output_dict, loss={}, split='val')
        # results_dict.update(stats_dict)
        # results_dict['loss'] = loss_dict['all']

        results_dict['best_value'] = - results_dict['PSNR_fine'] if 'PSNR_fine' in results_dict.keys() \
            else - results_dict['PSNR']

        # run some evaluations
        gt_image = data_dict.image.reshape(-1, 3, H, W)

        # coarse prediction
        pred_rgb_map = output_dict.rgb.reshape(-1, H, W, 3).permute(0, 3, 1, 2)  # (B, 3, H, W)
        ssim = ssim_loss(pred_rgb_map, gt_image).item()
        lpips = self.lpips_loss(pred_rgb_map * 2 - 1, gt_image * 2 - 1).item()

        results_dict['ssim'] = ssim
        results_dict['lpips'] = lpips

        if 'fg_mask' in data_dict.keys():
            results_dict.update(compute_metrics_masked(data_dict, pred_rgb_map, gt_image,
                                                       self.lpips_loss, suffix=''))

        if 'rgb_fine' in output_dict.keys():
            pred_rgb_map = output_dict.rgb_fine.reshape(-1, H, W, 3).permute(0, 3, 1, 2)
            ssim = ssim_loss(pred_rgb_map, gt_image).item()
            lpips = self.lpips_loss(pred_rgb_map * 2 - 1, gt_image * 2 - 1).item()

            results_dict['ssim_fine'] = ssim
            results_dict['lpips_fine'] = lpips

            if 'fg_mask' in data_dict.keys():
                results_dict.update(compute_metrics_masked(data_dict, pred_rgb_map, gt_image,
                                                           self.lpips_loss, suffix='_fine'))

        if iteration < 5 or (iteration % 4 == 0):
            plotting_dict_ = self.visualize(self.settings, data_dict, output_dict, step=iteration, split="val")
            plotting_dict.update(plotting_dict_)
        return output_dict, results_dict, plotting_dict

    def save_snapshot(self, filename, directory, optimizer, scheduler):
        """Saves a checkpoint of the network and other variables."""
        if hasattr(self, 'return_model_dict'):
            model_state_dict = self.return_model_dict()
        else:
            model_state_dict = self.state_dict()
        # save model
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        filename = osp.join(directory, filename)
        state_dict = {
            'current_pose': self.data_dict.poses_w2c,
            'epoch': self.epoch,
            'iteration': self.iteration,
            'iteration_nerf': self.iteration_nerf,
            'state_dict':  model_state_dict,
            'best_val': self.best_val,
            'epoch_of_best_val': self.epoch_of_best_val
        }
        state_dict['optimizer'] = optimizer.state_dict()
        state_dict['scheduler'] = scheduler.state_dict()
        # save snapshot
        torch.save(state_dict, filename)
        return

    @torch.no_grad()
    def evaluate_full(self, pose, pose_GT, val_data, opt,  plot= False,
                      save_ind_files= True, out_scene_dir= ''):
        """
        Does the actual evaluation here on the test set. Important that opt is given as variable to change
        the test time optimization input.
        Args:
            opt (edict): settings
            plot (bool, optional): Defaults to False.
            save_ind_files (bool, optional): Defaults to False
            out_scene_dir (str, optional): Path to dir, to save visualization if plot is True. Defaults to ''.
        Returns: dict with evaluation results
        """
        self.eval()
        if pose.shape[0] > 9:
            # alignment of the trajectory
            pose_aligned, self.sim3_est_to_gt_c2w = prealign_w2c_large_camera_systems(pose, pose_GT)
        else:
            # alignment of the first cameras
            pose_aligned, self.sim3_est_to_gt_c2w = prealign_w2c_small_camera_systems(pose, pose_GT)

        error = evaluate_camera_alignment(pose_aligned, pose_GT)

        res = []
        if plot:
            os.makedirs(out_scene_dir, exist_ok=True)

        results_dict = {'single': {}}
        torch.backends.cudnn.enabled = False
        for i in range(val_data['image'].size(0)):
            data_dict = {key: tensor[i:i + 1] for key, tensor in val_data.items()}
            data_dict = edict(data_dict)
            data_dict = self.to_cuda(data_dict)
            file_id = str(data_dict['idx'].item())

            total_img_coarse, total_img_fine = [], []

            if opt.model in ["joint_pose_nerf_training", 'nerf_fixed_noisy_poses'] and opt.optim.test_photo:
                # run test-time optimization to factorize imperfection in optimized poses from view synthesis evaluation
                data_dict = self.evaluate_test_time_photometric_optim(opt, data_dict)
                # important is data_dict.pose_refine_test
            H, W = data_dict.image.shape[-2:]
            opt.H, opt.W = H, W

            output_dict = self.forward(opt, data_dict, mode="eval", iter=None)

            # evaluate view synthesis, coarse
            scaling_factor_for_pred_depth = 1.
            if self.settings.model == 'joint_pose_nerf_training' and hasattr(self, 'sim3_est_to_gt_c2w'):
                # adjust the rendered depth, since the optimized scene geometry and poses are valid up to a 3D
                # similarity, compared to the ground-truth.
                scaling_factor_for_pred_depth = (
                            self.sim3_est_to_gt_c2w.trans_scaling_after * self.sim3_est_to_gt_c2w.s) \
                    if self.sim3_est_to_gt_c2w.type == 'align_to_first' else self.sim3_est_to_gt_c2w.s

            # gt image
            gt_rgb_map = data_dict.image  # (B, 3, H, W)

            # rendered image and depth map
            pred_rgb_map = output_dict.rgb.view(-1, opt.H, opt.W, 3).permute(0, 3, 1, 2)  # [B,3,H,W]
            pred_depth = output_dict.depth  # [B, -1, 1]

            results = compute_metrics(data_dict, output_dict, pred_rgb_map, pred_depth, gt_rgb_map,
                                      lpips_loss=self.lpips_loss,
                                      scaling_factor_for_pred_depth=scaling_factor_for_pred_depth, suffix='_c')

            if 'depth_fine' in output_dict.keys():
                pred_rgb_map_fine = output_dict.rgb_fine.view(-1, opt.H, opt.W, 3).permute(0, 3, 1, 2)  # [B,3,H,W]
                pred_depth_fine = output_dict.depth_fine
                results_fine = compute_metrics(data_dict, output_dict, pred_rgb_map_fine,
                                               pred_depth_fine, gt_rgb_map,
                                               scaling_factor_for_pred_depth=scaling_factor_for_pred_depth,
                                               lpips_loss=self.lpips_loss, suffix='_f')
                results.update(results_fine)

            res.append(results)

            message = "==================\n"
            message += "{}, curr_id: {}, shape {}x{} \n".format(self.settings.scene, file_id, H, W)
            for k, v in results.items():
                message += 'current {}: {:.2f}\n'.format(k, v)
            print(message)

            results_dict['single'][file_id] = results

            # plotting
            depth_range = None
            if plot:
                # invdepth = (1-var.depth)/var.opacity if opt.camera.ndc else 1/(var.depth/var.opacity+eps)
                # invdepth_map = invdepth.view(-1,opt.H,opt.W,1).permute(0,3,1,2) # [B,1,H,W]
                depth = output_dict.depth.view(-1, opt.H, opt.W, 1).permute(0, 3, 1,
                                                                            2) * scaling_factor_for_pred_depth  # [B,1,H,W]
                depth_var = output_dict.depth_var.view(-1, opt.H, opt.W, 1).permute(0, 3, 1, 2)  # [B,1,H,W]

                if hasattr(data_dict, 'depth_range'):
                    depth_range = data_dict.depth_range[0].cpu().numpy().tolist()

                self.visualize_eval(total_img_coarse, data_dict.image, pred_rgb_map, depth,
                                    depth_var, depth_range=depth_range)
                if 'depth_gt' in data_dict.keys():
                    depth_gt = data_dict.depth_gt[0]
                    depth_gt_colored = (255 * colorize_np(depth_gt.cpu().squeeze().numpy(),
                                                          range=depth_range, append_cbar=False)).astype(np.uint8)
                    total_img_coarse += [depth_gt_colored]

                if 'depth_fine' in output_dict.keys():
                    # invdepth_fine = (1-var.depth_fine)/var.opacity_fine if opt.camera.ndc else 1/(var.depth_fine/var.opacity_fine+eps)
                    # invdepth_map_fine = invdepth.view(-1,opt.H,opt.W,1).permute(0,3,1,2) # [B,1,H,W]
                    depth = output_dict.depth_fine.view(-1, opt.H, opt.W, 1).permute(0, 3, 1,
                                                                                     2) * scaling_factor_for_pred_depth  # [B,1,H,W]
                    depth_var = output_dict.depth_var_fine.view(-1, opt.H, opt.W, 1).permute(0, 3, 1, 2)  # [B,1,H,W]
                    self.visualize_eval(total_img_fine, data_dict.image, pred_rgb_map_fine, depth,
                                        depth_var, depth_range=depth_range)
                    if 'depth_gt' in data_dict.keys():
                        depth_gt = data_dict.depth_gt[0]
                        depth_gt_colored = (255 * colorize_np(depth_gt.cpu().squeeze().numpy(),
                                                              range=depth_range, append_cbar=False)).astype(np.uint8)
                        total_img_fine += [depth_gt_colored]

                # save the final image
                total_img_coarse = np.concatenate(total_img_coarse, axis=1)
                if len(total_img_fine) > 2:
                    total_img_fine = np.concatenate(total_img_fine, axis=1)
                    total_img = np.concatenate((total_img_coarse, total_img_fine), axis=0)
                else:
                    total_img = total_img_coarse
                if 'depth_gt' in data_dict.keys():
                    name = '{}_gt_rgb_depthc_depthvarc_depthgt.png'.format(file_id)
                else:
                    name = '{}_gt_rgb_depthc_depthvarc.png'.format(file_id)
                imageio.imwrite(os.path.join(out_scene_dir, name), total_img)

            if save_ind_files:
                pred_img_plot = pred_rgb_map_fine if 'depth_fine' in output_dict.keys() else pred_rgb_map
                pred_depth_plot = pred_depth_fine.view(-1, opt.H, opt.W, 1).permute(0, 3, 1,
                                                                                    2) if 'depth_fine' in output_dict.keys() \
                    else pred_depth.view(-1, opt.H, opt.W, 1).permute(0, 3, 1, 2)

                depth_gt_image = data_dict.depth_gt if 'depth_gt' in data_dict.keys() else None
                self.save_ind_files(out_scene_dir, file_id, data_dict.image, pred_img_plot,
                                    pred_depth_plot * scaling_factor_for_pred_depth,
                                    depth_range=depth_range, depth_gt=depth_gt_image)

        # compute average results over the full test set
        avg_results = {}
        keys = res[0].keys()
        for key in keys:
            avg_results[key] = np.mean([r[key] for r in res])
        results_dict['mean'] = avg_results

        # log results
        message = "------avg over {}-------\n".format(self.settings.scene)
        for k, v in avg_results.items():
            message += 'current {}: {:.3f}\n'.format(k, v)
        print(message)
        results_dict['rot_error'] = error.R.mean().item()
        results_dict['trans_error'] = error.t.mean().item()
        return results_dict

    @torch.no_grad()
    def generate_videos_synthesis(self, pose, pose_GT, opt, dataset, out_scene_dir= ''):
        """
        Will generate a video by sampling poses and rendering the corresponding images.
        """
        opt.output_path = out_scene_dir
        self.eval()

        poses = pose if opt.model == "joint_pose_nerf_training" else pose_GT
        poses_c2w = camera.pose.invert(poses)

        intr = self.data_dict.intr[:1].to(self.device)  # grab intrinsics

        # val_dataset = edict()
        # val_dataset.idx = torch.tensor(self.data_dict['train'])
        # val_dataset.image = self.images[self.data_dict['train']].permute(0, 3, 1, 2)
        # val_dataset.depth_range = torch.tensor([[self.near, self.far]]).repeat(
        #     len(self.data_dict['train']), 1)
        # val_dataset.intr = self.Ks[self.data_dict['train']]
        # val_dataset.pose = self.pose_GT[self.data_dict['train']]


        depth_range = self.data_dict['depth_range'][0].tolist()

        # alternative 1
        if 'llff' in self.settings.dataset:
            # render the novel views
            novel_path = "{}/novel_view".format(opt.output_path)
            os.makedirs(novel_path, exist_ok=True)

            pose_novel_c2w = generate_spiral_path(poses_c2w, bounds=np.array(depth_range),
                                                  n_frames=60).to(self.device)

            pose_novel = camera.pose.invert(pose_novel_c2w)

        elif 'dtu' in self.settings.dataset:
            # dtu
            n_frame = 60
            novel_path = "{}/novel_view".format(opt.output_path)
            os.makedirs(novel_path, exist_ok=True)

            pose_novel_c2w = generate_spiral_path_dtu(poses_c2w, n_rots=1, n_frames=n_frame).to(self.device)
            pose_novel_c2w = pose_novel_c2w
            pose_novel_1 = camera.pose.invert(pose_novel_c2w)

            # rotate novel views around the "center" camera of all poses
            scale = 1
            test_poses_w2c = poses
            idx_center = (test_poses_w2c - test_poses_w2c.mean(dim=0, keepdim=True))[..., 3].norm(dim=-1).argmin()
            pose_novel_2 = camera.get_novel_view_poses(opt, test_poses_w2c[idx_center], N=n_frame, scale=scale).to(
                self.device)

            pose_novel = torch.cat((pose_novel_1, pose_novel_2), dim=0)
            # render the novel views
        else:
            n_frame = 60
            novel_path = "{}/novel_view".format(opt.output_path)
            os.makedirs(novel_path, exist_ok=True)

            scale = 1
            test_poses_w2c = poses
            idx_center = (test_poses_w2c - test_poses_w2c.mean(dim=0, keepdim=True))[..., 3].norm(dim=-1).argmin()
            pose_novel = camera.get_novel_view_poses(opt, test_poses_w2c[idx_center], N=n_frame, scale=scale).to(
                self.device)

        pose_novel_tqdm = tqdm.tqdm(pose_novel, desc="rendering novel views", leave=False)

        H, W = self.data_dict.image.shape[-2:]
        for i, pose_aligned in enumerate(pose_novel_tqdm):
            ret = self.render_image_at_specific_pose_and_rays \
                (opt, self.data_dict, pose_aligned, intr, H, W, mode='val', iter=None)
            if 'rgb_fine' in ret:
                depth = ret.depth_fine
                rgb_map = ret.rgb_fine.view(-1, H, W, 3).permute(0, 3, 1, 2)  # [B,3,H,W]
            else:
                depth = ret.depth
                rgb_map = ret.rgb.view(-1, H, W, 3).permute(0, 3, 1, 2)  # [B,3,H,W]
            depth_map = depth.view(-1, H, W, 1).permute(0, 3, 1, 2)  # [B,1,H,W]
            color_depth_map = colorize_np(depth_map.cpu().squeeze().numpy(), range=depth_range, append_cbar=False)
            torchvision_F.to_pil_image(rgb_map.cpu()[0]).save("{}/rgb_{}.png".format(novel_path, i))
            imageio.imwrite("{}/depth_{}.png".format(novel_path, i), (255 * color_depth_map).astype(np.uint8))

        # write videos

        rgb_vid_fname = "{}/novel_view_{}_{}_rgb".format(opt.output_path, opt.dataset, opt.scene)
        depth_vid_fname = "{}/novel_view_{}_{}_depth".format(opt.output_path, opt.dataset, opt.scene)
        os.system(
            "ffmpeg -y -framerate 10 -i {0}/rgb_%d.png -pix_fmt yuv420p {1}.mp4 >/dev/null 2>&1".format(novel_path,
                                                                                                        rgb_vid_fname))
        os.system(
            "ffmpeg -y -framerate 10 -i {0}/depth_%d.png -pix_fmt yuv420p {1}.mp4 >/dev/null 2>&1".format(novel_path,
                                                                                                          depth_vid_fname))

        os.system("ffmpeg -f image2 -framerate 10 -i {0}/rgb_%d.png -loop -1 {1}.gif >/dev/null 2>&1".format(novel_path,
                                                                                                             rgb_vid_fname))
        os.system(
            "ffmpeg -f image2 -framerate 10 -i {0}/depth_%d.png -loop -1 {1}.gif >/dev/null 2>&1".format(novel_path,
                                                                                                         depth_vid_fname))
        return

    @torch.enable_grad()
    def evaluate_test_time_photometric_optim(self, opt: Dict[str, Any],
                                             data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Run test-time optimization. Optimizes over data_dict.se3_refine_test"""
        # only optimizes for the test pose here
        data_dict.se3_refine_test = torch.nn.Parameter(torch.zeros(1, 6, device=self.device))
        optimizer = getattr(torch.optim, opt.optim.algo_pose)
        optim_pose = optimizer([dict(params=[data_dict.se3_refine_test], lr=opt.optim.lr_pose)])
        # iterator = tqdm.trange(opt.optim.test_iter,desc="test-time optim.",leave=False,position=1)
        for it in range(opt.optim.test_iter):
            optim_pose.zero_grad()

            data_dict.pose_refine_test = camera.lie.se3_to_SE3(data_dict.se3_refine_test)
            output_dict = self.forward(opt, data_dict, mode="test-optim", iter=None)

            # current estimate of the pose
            poses_w2c = self.get_pose(self.settings, data_dict, mode='test-optim')  # is it world to camera
            data_dict.poses_w2c = poses_w2c

            # iteration needs to reflect the overall training
            loss, stats_dict, plotting_dict = self.loss_module.compute_loss \
                (opt, data_dict, output_dict, iteration=self.iteration, mode='test-optim')
            loss.all.backward()
            optim_pose.step()
            # iterator.set_postfix(loss="{:.3f}".format(loss.all))
        return data_dict

    @torch.no_grad()
    def visualize_eval(self, to_plot: List[Any], image: torch.Tensor, rendered_img: torch.Tensor,
                       rendered_depth: torch.Tensor, rendered_depth_var: torch.Tensor, depth_range: List[float] = None):
        """Visualization for the test set"""

        image = (image.permute(0, 2, 3, 1)[0].cpu().numpy() * 255.).astype(np.uint8)  # (B, H, W, 3), B is 1
        H, W = image.shape[1:3]
        # cannot visualize if it is not rendering the full image!
        depth = rendered_depth[0].squeeze().cpu().numpy()  # [B,H,W, 1] and then (H, W, 1)
        depth_var = rendered_depth_var[0].squeeze().cpu().numpy()  # [B,H,W, 1] and then (H, W, 1)
        rgb_map = rendered_img.permute(0, 2, 3, 1)[0].cpu().numpy()  # [B,3, H,W] and then (H, W, 3)

        fine_pred_rgb_np_uint8 = (255 * np.clip(rgb_map, a_min=0, a_max=1.)).astype(np.uint8)

        fine_pred_depth_colored = colorize_np(depth, range=depth_range, append_cbar=False)
        fine_pred_depth_colored = (255 * fine_pred_depth_colored).astype(np.uint8)

        fine_pred_depth_var_colored = colorize_np(depth_var, append_cbar=False)
        fine_pred_depth_var_colored = (255 * fine_pred_depth_var_colored).astype(np.uint8)

        to_plot += [image, fine_pred_rgb_np_uint8, fine_pred_depth_colored, fine_pred_depth_var_colored]
        return

    @torch.no_grad()
    def save_ind_files(self, save_dir: str, name: str, image: torch.Tensor,
                       rendered_img: torch.Tensor, rendered_depth: torch.Tensor,
                       depth_range: List[float] = None, depth_gt: torch.Tensor = None):
        """Save rendering and ground-truth data as individual files.

        Args:
            save_dir (str): dir to save the images
            name (str): name of image (without extension .png)
            image (torch.Tensor): gt image of shape [1, 3, H, W]
            rendered_img (torch.Tensor): rendered image of shape [1, 3, H, W]
            rendered_depth (torch.Tensor): rendered depth of shape [1, H, W, 1]
            depth_range (list of floats): depth range for depth visualization
            depth_gt (torch.Tensor): gt depth of shape [1, H, W, 1]
        """
        rend_img_dir = os.path.join(save_dir, 'rendered_imgs')
        rend_depth_dir = os.path.join(save_dir, 'rendered_depths')
        gt_img_dir = os.path.join(save_dir, 'gt_imgs')
        gt_depth_dir = os.path.join(save_dir, 'gt_depths')
        if not os.path.exists(rend_img_dir):
            os.makedirs(rend_img_dir, exist_ok=True)
        if not os.path.exists(gt_img_dir):
            os.makedirs(gt_img_dir, exist_ok=True)
        if not os.path.exists(rend_depth_dir):
            os.makedirs(rend_depth_dir, exist_ok=True)
        if not os.path.exists(gt_depth_dir):
            os.makedirs(gt_depth_dir, exist_ok=True)

        image = (image.permute(0, 2, 3, 1)[0].cpu().numpy() * 255.).astype(np.uint8)  # (B, H, W, 3), B is 1
        imageio.imwrite(os.path.join(gt_img_dir, name + '.png'), image)
        H, W = image.shape[1:3]
        # cannot visualize if it is not rendering the full image!

        rgb_map = rendered_img.permute(0, 2, 3, 1)[0].cpu().numpy()  # [B,3, H,W] and then (H, W, 3)
        fine_pred_rgb_np_uint8 = (255 * np.clip(rgb_map, a_min=0, a_max=1.)).astype(np.uint8)
        imageio.imwrite(os.path.join(rend_img_dir, name + '.png'), fine_pred_rgb_np_uint8)

        depth = rendered_depth[0].squeeze().cpu().numpy()  # [B,H,W, 1] and then (H, W)
        fine_pred_depth_colored = colorize_np(depth, range=depth_range, append_cbar=False)
        fine_pred_depth_colored = (255 * fine_pred_depth_colored).astype(np.uint8)
        imageio.imwrite(os.path.join(rend_depth_dir, name + '.png'), fine_pred_depth_colored)

        if depth_gt is not None:
            depth = depth_gt[0].squeeze().cpu().numpy()  # [B,H,W, 1] and then (H, W, 1)
            fine_pred_depth_colored = colorize_np(depth, range=depth_range, append_cbar=False)
            fine_pred_depth_colored = (255 * fine_pred_depth_colored).astype(np.uint8)
            imageio.imwrite(os.path.join(gt_depth_dir, name + '.png'), fine_pred_depth_colored)
        return





