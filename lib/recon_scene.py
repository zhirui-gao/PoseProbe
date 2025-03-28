import copy
import importlib
import os
import random
import time

import cv2

from eval import prealign_w2c_small_camera_systems

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import einops
import json
from torchvision.utils import save_image
from lib import utils_vis
from lib import common
from lib.nvs_fun import visualize_val_image
from lib.losses import compute_diff_loss
from tqdm import tqdm
import lib.voxurf_coarse as Model
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import trimesh
import torch
from datetime import datetime
from lib import utils
from torch.utils.tensorboard import SummaryWriter
from lib import camera
from easydict import EasyDict as edict
from lib.losses import object_losses
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
from lib.bg_nerf.source.models.renderer import Graph
import lib.bg_nerf.source.admin.settings as ws_settings
from lib.bg_nerf.source.training.define_trainer import define_trainer
@torch.no_grad()
def get_all_training_poses(model, poses, device):
    # get ground-truth (canonical) camera poses
    pose_GT = poses.to(device)
    pose = pose_GT.clone()
    # add synthetic pose perturbation to all training data
    if model.camera_noise:
        pose[model.i_train] = camera.pose.compose([model.pose_noise, pose[model.i_train]])

    # add learned pose correction to all training data
    pose_refine = camera.lie.se3_to_SE3(model.se3_refine)

    pose[model.i_train] = camera.pose.compose([pose_refine, pose[model.i_train]])
    return pose, pose_GT

def get_current_pose(model, poses_gt, optimize_align=False,ids=None):
    s_ids = model.i_train
    if ids is not None:
        s_ids = ids
    pose = poses_gt.clone()
    pose_refine = camera.lie.se3_to_SE3(model.se3_refine)
    pose[s_ids] = camera.pose.compose([model.pose_noise[s_ids], pose[s_ids]])
    pose[s_ids] = camera.pose.compose([pose_refine[s_ids], pose[s_ids]])
    return pose

def get_current_pose_pnp(model, pose_pnp, optimize_align=False,ids=None):
    pose = torch.stack(pose_pnp, dim=0).float()
    s_ids = model.i_train
    if ids is not None:
        s_ids = ids
    # add learned pose correction to all training data
    pose_refine = camera.lie.se3_to_SE3(model.se3_refine)
    s_ids = s_ids[s_ids != 0] # fix the first pose
    pose[s_ids] = camera.pose.compose([pose_refine[s_ids], pose[s_ids]])

    if optimize_align:
        pose_align_refine = camera.lie.se3_to_SE3(model.se3_align_refine)
        pose[s_ids] = camera.pose.compose([pose_align_refine[0], pose[s_ids]])
    return pose




@torch.no_grad()
def evaluate_camera_alignment(pose_aligned,pose_GT):
    # measure errors in rotation and translation
    pose_aligned_c2w = camera.pose.invert(pose_aligned)
    pose_GT_c2w = camera.pose.invert(pose_GT)
    R_aligned,t_aligned = pose_aligned_c2w.split([3,1],dim=-1)
    R_GT,t_GT = pose_GT_c2w.split([3,1],dim=-1)
    R_error = camera.rotation_distance(R_aligned,R_GT)
    t_error = (t_aligned-t_GT)[...,0].norm(dim=-1)
    R_error = R_error*180. / np.pi
    t_error = t_error*100
    error = edict(R=R_error, t=t_error)
    return error

def get_ray_dir(points, K, c2w, inverse_y, flip_x, flip_y,mode='center'):
    if mode == 'center':
        points = points + 0.5
    if flip_x:
        points[:,:, 0]  = points[:,:, 0] .flip((1,))
    if flip_y:
        points[:,:, 1] = points[:,:, 1].flip((0,))
    if inverse_y:
        dirs = torch.stack(
            [(points[:, :, 0] - K[:, 0, [2]]) / K[:, 0, [0]], (points[:, :, 1] - K[:, 1, [2]]) / K[:, 1, [1]],
             torch.ones_like(points[:, :, 0])], -1)
    else:
        dirs = torch.stack(
        [(points[:,:, 0] - K[:,0,[2]]) / K[:,0,[0]], -(points[:,:, 1] - K[:,1,[2]]) / K[:,1,[1]],
         -torch.ones_like(points[:,:, 0])], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:,np.newaxis, :3, :3],
                       -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)
    rays_o = c2w[:,np.newaxis,:3, 3].expand(rays_d.shape)
    return rays_o, rays_d


class scene_rep_reconstruction(torch.nn.Module):
    def __init__(self, args, cfg, logger, cfg_model, cfg_train, xyz_min, xyz_max, data_dict, stage):
        super(scene_rep_reconstruction, self).__init__()
        # init
        self.args, self.cfg_train, self.cfg_model, self.cfg, self.logger, self.data_dict = \
            args, cfg_train, cfg_model, cfg, logger, data_dict

        # Logging setup
        log_dir = os.path.join(cfg.basedir, cfg.expname, 'log')
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir, filename_suffix=datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # World Bound Scaling
        cfg_train.world_bound_scale = getattr(cfg_train, 'world_bound_scale', 1.5)
        if abs(cfg_train.world_bound_scale - 1) > 1e-9:
            xyz_shift = (xyz_max - xyz_min) * (cfg_train.world_bound_scale - 1) / 2
            xyz_min -= xyz_shift
            xyz_max += xyz_shift

        keys = ['HW', 'Ks', 'near', 'far', 'i_train', 'i_val', 'i_test','matcher_infos', 'poses', 'render_poses',
                'images', 'images_gray', 'masks', 'samplers', 'align_pose']
        self.HW, self.Ks, self.near, self.far, self.i_train, self.i_val, self.i_test, self.matcher_result,self.poses, \
            self.render_poses, self.images, self.images_gray, self.masks, self.samplers, self.align_pose = [data_dict[k] for k in keys]
        self.rect_size = ((xyz_max - xyz_min) / (cfg_train.world_bound_scale * 1.05)).tolist()
        self.range_shape = (xyz_max - xyz_min) / (cfg_train.world_bound_scale * 1.05)
        sdf_grid_path, sdf0 = None, None

        self.last_ckpt_path = os.path.join(cfg.basedir, cfg.expname, f'{stage}_pose_last.tar')
        model_kwargs = copy.deepcopy(cfg_model)
        num_voxels = model_kwargs.pop('num_voxels')
        num_voxels_bg = model_kwargs.pop('num_voxels_bg', num_voxels)

        self.model_pose = Model.pose_model(i_train=self.i_train, camera_noise=cfg.camera.noise)
        self.model = Model.Voxurf(
            xyz_min=xyz_min, xyz_max=xyz_max,
            num_voxels=num_voxels,
            num_voxels_bg=num_voxels_bg,
            mask_cache_path=sdf_grid_path,
            exppath=os.path.join(cfg.basedir, cfg.expname),
            camera_noise=cfg.camera.noise,
            barf_c2f=cfg.camera.barf_c2f,
            i_train=self.i_train,
            N_iters=cfg_train.N_iters,
            HW=self.HW,
            range_shape=self.range_shape,
            rect_size=self.rect_size,
            **model_kwargs
        )
        self.model.maskout_near_cam_vox(self.poses[self.i_train, :3, 3], self.near)
        self.model = self.model.to(self.device)
        if sdf0 is not None:
            self.model.init_sdf_from_sdf(sdf0, smooth=False)

        self.model_pose = self.model_pose.to(self.device)
        # Pose Initialization
        self.poses_raw, self.pose_GT = get_all_training_poses(model=self.model_pose, poses=self.poses,
                                                              device=self.device)

        # Optimizer Initialization
        self.optimizer = utils.create_optimizer_or_freeze_model(self.model, cfg_train, global_step=0)

        self.render_kwargs = {
            'near': self.near,
            'far': self.far,
            'bg': 1 if cfg.data.white_bkgd else 0,
            'stepsize': cfg_model.stepsize,
            'inverse_y': cfg.data.inverse_y,
            'flip_x': cfg.data.flip_x,
            'flip_y': cfg.data.flip_y,
        }
        self.nl = 0.05

        # PnP Initialization
        self.initialize_pnp()
        if self.cfg.pnp.use_pnp:
            self.initialize_pnp()

        # Evaluation
        self.evaluate_initial_pose()

        # Matcher Setup
        self.setup_matcher_results()

    def initialize_pnp(self):
        """Initialize camera poses using PnP."""
        self.poses_pnp = [self.pose_GT[0].detach()]
        for i in self.i_train[1:]:
            if getattr(self.cfg.pnp, 'use_identical', False):
                camera_pose = self.poses_pnp[i - 1]
            else:
                camera_pose = self.opencv_pnp_ransac(self.matcher_result[i], i, self.Ks,
                                                     self.poses_pnp[i - 1].unsqueeze(0), self.render_kwargs)
            self.poses_pnp.append(camera_pose)
        initial_pose = torch.stack(self.poses_pnp, dim=0)
        np.save(self.cfg.data.datadir + '/' + str(len(self.i_train)) + '_initial_pose_new.npy',
                initial_pose.detach().cpu())

    def evaluate_initial_pose(self):
        """Evaluate initial pose alignment."""
        initial_pose = get_current_pose(model=self.model_pose, poses_gt=self.poses)[self.i_train]
        pose_aligned, pose_ref = initial_pose.detach().cpu(), self.pose_GT[self.i_train].detach().cpu()
        pose_aligned, _ = prealign_w2c_small_camera_systems(pose_aligned, pose_ref)
        error = evaluate_camera_alignment(pose_aligned, pose_ref)
        print('Initialized by PnP, the pose error is:', error.R.mean(), error.t.mean())
        fig = plt.figure(figsize=(10, 10))
        utils_vis.plot_save_poses_blender(fig, pose_aligned, pose_ref=pose_ref, path=self.cfg.data.datadir, ep='-1')
        plt.close()

    def setup_matcher_results(self):
        """Setup matcher results for training."""
        coord0, coord1, mconf, i_index, j_index = [], [], [], [], []
        for i in self.i_train:
            num_camera = min(1, len(self.i_train) - 1)
            j_train = list(range(i - 1, -1, -1)) + list(range(i + 1, self.i_train.shape[0], 1))
            mconf_h = torch.stack(self.matcher_result[i], dim=0)[:num_camera, :, -1]
            coord0_h = torch.stack(self.matcher_result[i], dim=0)[:num_camera, :, 0:2]
            coord1_h = torch.stack(self.matcher_result[i], dim=0)[:num_camera, :, 2:4]
            i_index.append(np.array(i).repeat(num_camera))
            j_index.append(j_train[:num_camera])
            coord0.append(coord0_h)
            coord1.append(coord1_h)
            mconf.append(mconf_h)
        self.mconf = torch.concat(mconf, dim=0)
        self.coord0 = torch.concat(coord0, dim=0)
        self.coord1 = torch.concat(coord1, dim=0)
        self.i_index = np.concatenate(i_index, axis=0)
        self.j_index = np.concatenate(j_index, axis=0)


        coord0, coord1, mconf = [], [], []
        for i in self.i_train:
            mconf_h = torch.stack(self.matcher_result[i + len(self.i_train)], dim=0)[:1, :, -1]
            coord0_h = torch.stack(self.matcher_result[i + len(self.i_train)], dim=0)[:1, :, 0:2]
            coord1_h = torch.stack(self.matcher_result[i + len(self.i_train)], dim=0)[:1, :, 2:4]
            coord0.append(coord0_h)
            coord1.append(coord1_h)
            mconf.append(mconf_h)
        self.mconf_scene = torch.concat(mconf, dim=0)
        self.coord0_scene = torch.concat(coord0, dim=0)
        self.coord1_scene = torch.concat(coord1, dim=0)

        if self.cfg.camera.incremental:
            self.selected_i_train = [0, 1]
        else:
            self.selected_i_train = list(range(0, len(self.i_train), 1))

    @torch.no_grad()
    def log_scalars(self, loss, loss_weight, metric=None, step=0, split="train"):
        for key, value in loss.items():
            if key == "all":
                continue
            if loss_weight[key] is not None:
                self.writer.add_scalar(f"{split}/loss_{key}", value, step)
        if metric is not None:
            for key, value in metric.items():
                self.writer.add_scalar(f"{split}/{key}", value, step)

    def opencv_pnp_ransac(self, matcher_result_list, img_id, Ks, current_pose, render_kwargs):
        if isinstance(img_id, int) or isinstance(img_id, np.int64):
            img_id = [img_id]
        matcher_result = matcher_result_list[0]
        coord0_h = matcher_result[:, 0:2][None]  # others
        x2d = matcher_result[:, 2:4][None]
        weights = matcher_result[:, -1][None]
        rays_o_0, rays_d_0 = get_ray_dir(
            coord0_h, Ks[img_id], c2w=camera.pose.invert(current_pose),
            inverse_y=self.cfg.data.inverse_y,
            flip_x=self.cfg.data.flip_x, flip_y=self.cfg.data.flip_y, mode='no_center'
        )
        rays_o_0 = einops.rearrange(rays_o_0, 'b n c -> (b n) c', c=3)
        rays_d_0 = einops.rearrange(rays_d_0, 'b n c -> (b n) c', c=3)
        query_points, mask_valid, sdf_ray_step = self.model.query_sdf_point_wocuda(
            rays_o_0, rays_d_0, global_step=None, keep_dim=True, **render_kwargs
        )
        world_points = einops.rearrange(query_points, '(b n) c -> b n c', b=len(img_id), c=3)
        mask_valid = einops.rearrange(mask_valid, '(b n) -> b n', b=len(img_id))
        weights = weights * mask_valid
        world_points = world_points.detach().cpu().numpy().squeeze(0)
        img_points = x2d.detach().cpu().numpy().squeeze(0)
        weights = weights.detach().cpu().numpy().squeeze(0)
        Ks = Ks.detach().cpu().numpy()
        dist_coeffs = np.zeros((4, 1))
        mask = weights > 0
        world_points = world_points[mask]
        img_points = img_points[mask]
        _, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(
            world_points, img_points, Ks[0], dist_coeffs
        )
        rotation_vector, translation_vector = torch.tensor(rotation_vector), torch.tensor(translation_vector)
        R = camera.lie.so3_to_SO3(rotation_vector.T)
        pose = torch.cat([R.squeeze(0), translation_vector], dim=-1).float()
        return pose  # the inverse of extrinsic matrices

    def point_to_ray_distance(self, ray_origins, ray_directions, point):
        v = point - ray_origins
        t = torch.sum(v * ray_directions, dim=1)
        distances = torch.norm(v, dim=1)
        p_proj = ray_origins + t.unsqueeze(1) * ray_directions
        distances = torch.where(t < 0, distances, torch.norm(point - p_proj, dim=1))
        return distances


    def get_project_error(self,global_step,current_pose, coord0, coord1, i_train, j_train,
                          mconf, use_deform=True, pixel_thre=None, **render_kwargs):
        coord = torch.concat([coord0, coord1], dim=0)
        index = np.concatenate([i_train, j_train], axis=0)
        mconf = torch.concat([mconf, mconf], dim=0)
        rays_o_p, rays_d_p = get_ray_dir(coord, self.Ks[index], c2w=camera.pose.invert(current_pose[index]),
                                         inverse_y=self.cfg.data.inverse_y,
                                         flip_x=self.cfg.data.flip_x, flip_y=self.cfg.data.flip_y, mode='no_center')
        rays_o_p = einops.rearrange(rays_o_p, 'b n c ->(b n) c', c=3)
        rays_d_p = einops.rearrange(rays_d_p, 'b n c ->(b n) c', c=3)
        if use_deform:
            query_points, mask_valid, _ = self.model.query_sdf_point_wocuda_render(rays_o_p, rays_d_p,
                                                                                                global_step=global_step,
                                                                                                use_deform=use_deform,
                                                                                                keep_dim=True,
                                                                                                **render_kwargs)
        else:
            query_points, mask_valid, _ = self.model.query_sdf_point_wocuda_wodeform(rays_o_p, rays_d_p,
                                                                              global_step=global_step,
                                                                              keep_dim=True, **render_kwargs)

        # distance to center:
        dis2center = self.point_to_ray_distance(rays_o_p, rays_d_p, point=self.model.xyz_min + self.model.xyz_max)
        near_surface_loss = (torch.clamp(dis2center - self.model.diagonal_length/2., min=0.0)* (mconf.flatten() > 0)).sum()


        query_points = einops.rearrange(query_points, '(b n) c ->b n c', b=len(index), c=3)
        mask_valid = einops.rearrange(mask_valid, '(b n)->b n', b=len(index))

        index = np.concatenate([j_train, i_train], axis=0)
        camera_pose_j = current_pose[index]
        pc_camera = camera.world2cam(query_points, camera_pose_j)
        if self.cfg.data.inverse_y:
            mask_pc_invalid = (pc_camera[..., 2:] < self.nl).expand_as(pc_camera)
            pc_camera[mask_pc_invalid] = self.nl
            p_reprojected = common.project_to_cam_real(pc_camera, self.Ks[index, ...], self.HW)
        else:
            mask_pc_invalid = (-pc_camera[..., 2:] < self.nl).expand_as(pc_camera)
            pc_camera[mask_pc_invalid] = self.nl
            p_reprojected = common.project_to_cam_real(pc_camera, self.Ks[index, ...], self.HW)
            p_reprojected[..., 0] = self.HW[0, 1] - p_reprojected[..., 0]

        diff = torch.norm(p_reprojected - torch.concat([coord1, coord0], dim=0), p=2, dim=-1)
        valid = (~mask_pc_invalid[:, :, 0]) * mask_valid
        if pixel_thre is not None:
            valid_corr = diff.detach().le(pixel_thre)
            valid = valid & valid_corr
        projection_dis_error = compute_diff_loss('huber', diff, weights=mconf, mask=valid, delta=1.)
        return projection_dis_error, near_surface_loss

    def get_project_feature_loss(self, global_step,use_deform, current_pose, imsz,target_tr,rays_o_tr,rays_d_tr, i_list,j_list):
        num_min = min(256, imsz)
        indices = [(torch.randperm(imsz, device=target_tr.device)[:num_min])]
        rays_o_tr = einops.rearrange(rays_o_tr[indices], '(b n) c ->b n c', b=len(i_list),c=3)
        rays_d_tr = einops.rearrange(rays_d_tr[indices], '(b n) c ->b n c', b=len(i_list),c=3)

        loss_surface_projection = 0
        rays_o = rays_o_tr.reshape(-1, 3)
        rays_d = rays_d_tr.reshape(-1, 3)
        camera_pose_0 = current_pose[i_list]
        camera_pose_1 = current_pose[j_list]
        query_points, mask_valid, _ = self.model.query_sdf_point_wocuda_render(rays_o, rays_d,
                                                                        global_step=global_step,
                                                                        use_deform=use_deform,
                                                                        keep_dim=True, **self.render_kwargs)
        query_points = einops.rearrange(query_points, '(b n) c ->b n c', b=len(i_list), c=3)
        mask_valid = einops.rearrange(mask_valid, '(b n)->b n', b=len(i_list))

        pc_camera_1 = camera.world2cam(query_points, camera_pose_1)
        if self.cfg.data.inverse_y:
            mask_pc_invalid = (pc_camera_1[..., 2:] < self.nl).expand_as(pc_camera_1)
            pc_camera_1[mask_pc_invalid] = self.nl
        else:
            mask_pc_invalid = (-pc_camera_1[..., 2:] < self.nl).expand_as(pc_camera_1)
            pc_camera_1[mask_pc_invalid] = self.nl
        p_reprojected_1 = common.project_to_cam_real(pc_camera_1, self.Ks[j_list, ...], self.HW)
        rays_o_ref, rays_d_ref = get_ray_dir(p_reprojected_1,  self.Ks[j_list, ...],
                                             c2w=camera.pose.invert(current_pose[j_list]),
                                             inverse_y= self.cfg.data.inverse_y, flip_x= self.cfg.data.flip_x,
                                             flip_y= self.cfg.data.flip_y, mode='no_center')
        rays_o_ref = rays_o_ref.reshape(-1, 3)
        rays_d_ref = rays_d_ref.reshape(-1, 3)
        query_points_ref, valid_point_ref, _ = self.model.query_sdf_point_wocuda_render(rays_o_ref, rays_d_ref,
                                                                                 global_step=global_step,
                                                                                 use_deform=use_deform,
                                                                                 keep_dim=True,
                                                                                 **self.render_kwargs)
        query_points_ref = einops.rearrange(query_points_ref, '(b n) c ->b n c', b=len(j_list), c=3)
        valid_point_ref = einops.rearrange(valid_point_ref, '(b n)->b n', b=len(j_list))
        valid_depth_ray = torch.linalg.norm(query_points - query_points_ref, dim=-1) < self.model.voxel_size * 2
        valid_depth_ray = valid_depth_ray * valid_point_ref * mask_valid

        pc_camera_0 = camera.world2cam(query_points, camera_pose_0)
        pc_camera_1 = camera.world2cam(query_points, camera_pose_1)
        if self.cfg.data.inverse_y:
            mask_pc_invalid = (pc_camera_0[..., 2:] < self.nl).expand_as(pc_camera_0)
            pc_camera_0[mask_pc_invalid] = self.nl
            mask_pc_invalid = (pc_camera_1[..., 2:] < self.nl).expand_as(pc_camera_1)
            pc_camera_1[mask_pc_invalid] = self.nl
        else:
            mask_pc_invalid = (-pc_camera_0[..., 2:] < self.nl).expand_as(pc_camera_0)
            pc_camera_0[mask_pc_invalid] = self.nl
            mask_pc_invalid = (-pc_camera_1[..., 2:] < self.nl).expand_as(pc_camera_1)
            pc_camera_1[mask_pc_invalid] = self.nl
        pc_camera = torch.concat([pc_camera_0, pc_camera_1], dim=0)
        ij_list = np.concatenate((i_list, j_list))
        p_reprojected, valid_mask = common.project_to_cam(pc_camera, self.Ks[ij_list, ...], self.HW)
        valid_mask = (valid_mask[0:len(i_list), ] * valid_mask[len(i_list):, ...]).squeeze(-1)
        valid_mask = valid_mask * valid_depth_ray
        for vgg_features_layer in self.data_dict['vgg_features']:
            rgb_pc_proj_feature = common.get_tensor_values(vgg_features_layer[ij_list, ...],
                                                           p_reprojected,
                                                           mode='bilinear', scale=False, detach=False,
                                                           detach_p=False, align_corners=True)
            feature0 = rgb_pc_proj_feature[0:len(i_list), ...][valid_mask]
            feature1 = rgb_pc_proj_feature[len(i_list):, ...][valid_mask]
            loss_surface_projection += (1 - (torch.cosine_similarity(feature0, feature1, dim=-1).mean())) * len(
                i_list)
        return loss_surface_projection



    def gather_training_rays(self, current_pose, imgs_indice,ray_sampler=None):
        c2w = camera.pose.invert(current_pose)
        if  self.data_dict['irregular_shape']:
            rgb_tr_ori = [self.images[i].to('cpu' if self.cfg.data.load2gpu_on_the_fly else
                                            self.device) for i in self.i_train]
            mask_tr_ori = [self.masks[i].to('cpu' if self.cfg.data.load2gpu_on_the_fly else
                                            self.device) for i in self.i_train]
        else:
            rgb_tr_ori = self.images[imgs_indice].to('cpu' if self.cfg.data.load2gpu_on_the_fly else self.device)
            mask_tr_ori = self.masks[imgs_indice].to('cpu' if self.cfg.data.load2gpu_on_the_fly else self.device)
        if ray_sampler is None:
            ray_sampler = self.cfg_train.ray_sampler
        if ray_sampler == 'in_maskcache':
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = Model.get_training_rays_in_maskcache_sampling_grad(
                rgb_tr_ori=rgb_tr_ori,
                train_poses=c2w[self.i_train],
                HW=self.HW[self.i_train], Ks=self.Ks[self.i_train],
                ndc=self.cfg.data.ndc, inverse_y=self.cfg.data.inverse_y,
                flip_x=self.cfg.data.flip_x, flip_y=self.cfg.data.flip_y,
                model=self.model, render_kwargs=self.render_kwargs,
            )
            indices = torch.randperm(len(rgb_tr), device=rgb_tr.device)[:self.cfg_train.N_rand]
            rgb_tr = rgb_tr[indices]
            rays_o_tr = rays_o_tr[indices]
            rays_d_tr = rays_d_tr[indices]
            viewdirs_tr = viewdirs_tr[indices]
        elif ray_sampler == 'flatten':
            rgb_tr, mask_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = Model.get_training_rays_flatten(
                rgb_tr_ori=rgb_tr_ori,
                mask_tr_ori=mask_tr_ori,
                train_poses=c2w[imgs_indice],
                HW=self.HW[imgs_indice], Ks=self.Ks[imgs_indice], ndc=self.cfg.data.ndc, inverse_y=self.cfg.data.inverse_y,
                flip_x=self.cfg.data.flip_x, flip_y=self.cfg.data.flip_y)
            indices = torch.randperm(len(rgb_tr), device=rgb_tr.device)[:self.cfg_train.N_rand]
            rgb_tr = rgb_tr[indices]
            mask_tr = mask_tr[indices]
            rays_o_tr = rays_o_tr[indices]
            rays_d_tr = rays_d_tr[indices]
            viewdirs_tr = viewdirs_tr[indices]
        elif ray_sampler=='semantic':
            rgb_tr, mask_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = Model.get_training_rays_semantic(
                rgb_tr_ori=rgb_tr_ori,
                mask_tr_ori=mask_tr_ori,
                train_poses=c2w[imgs_indice],
                HW=self.HW[imgs_indice], Ks=self.Ks[imgs_indice], ndc=self.cfg.data.ndc, inverse_y=self.cfg.data.inverse_y,
                flip_x=self.cfg.data.flip_x, flip_y=self.cfg.data.flip_y,samplers=self.samplers)

        elif ray_sampler=='semantic_split':
            rgb_tr, mask_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = Model.get_training_rays_semantic(
                rgb_tr_ori=rgb_tr_ori,
                mask_tr_ori=mask_tr_ori,
                train_poses=c2w[imgs_indice],
                HW=self.HW[imgs_indice], Ks=self.Ks[imgs_indice], ndc=self.cfg.data.ndc, inverse_y=self.cfg.data.inverse_y,
                flip_x=self.cfg.data.flip_x, flip_y=self.cfg.data.flip_y, samplers=self.samplers)
        else:
            rgb_tr,mask_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = Model.get_training_rays(
                rgb_tr=rgb_tr_ori,
                mask_tr = mask_tr_ori,
                train_poses=c2w[imgs_indice],
                HW=self.HW[imgs_indice], Ks=self.Ks[imgs_indice], ndc=self.cfg.data.ndc, inverse_y=self.cfg.data.inverse_y,
                flip_x=self.cfg.data.flip_x, flip_y=self.cfg.data.flip_y)
        return rgb_tr, mask_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz

    def freeze_rgb_net(self):
        for name, param in self.model.named_parameters():
            if name.split('.')[0] in ['k0', 'rgbnet']:
                param.requires_grad = False

    def unfreeze_rgb_net(self):
        for name, param in self.model.named_parameters():
            if name.split('.')[0] in ['k0', 'rgbnet']:
                param.requires_grad = True

    def freeze_deform_net(self):
        for name, param in self.model.named_parameters():
            if name.split('.')[0] in ['warp_network']:
                param.requires_grad = False

    def unfreeze_deform_net(self):
        for name, param in self.model.named_parameters():
            if name.split('.')[0] in ['warp_network']:
                param.requires_grad = True

    def create_freeze_grad_hook(self,num_fixed_rows):
        def freeze_grad_hook(grad):
            grad_clone = grad.clone()
            grad_clone[:num_fixed_rows, :] = 0
            return grad_clone

        return freeze_grad_hook

    def optimize_increamental(self, end_step, start_object, opt=None):
        opt.H, opt.W = self.HW[0]
        opt.device = self.device

        # Initialize tracking lists
        psnr_lst, psnr_bg_lst, weight_lst, ep_list = [], [], [], []
        mask_lst, bg_mask_lst, weight_sum_lst, weight_nonzero_lst, s_val_lst = [], [], [], [], []

        # Settings
        time0 = time.time()
        selected_i, global_step = 1, 0
        incremental_step = [0] + [self.cfg.camera.incremental_step] * (len(self.i_train) - 2)

        pbar = tqdm(range(self.model_bg.settings.max_iter))
        step_pose_optim = self.model_bg.settings.max_iter * \
                             self.model_bg.settings.ratio_end_joint_nerf_pose_refinement
        self.optim_pose, self.sched_pose = utils.create_optimizer_pose(self.model_pose, self.cfg_train,
                                                                       step_pose_optim, False)
        while global_step <= end_step:
            pbar.update(1)
            IsOptimizePose = global_step < step_pose_optim
            if self.cfg.camera.incremental and len(self.selected_i_train) < len(self.i_train) and \
                    global_step % incremental_step[selected_i]==0 and global_step:
                selected_i += 1
                self.selected_i_train.append(selected_i)
                if getattr(self.cfg.pnp, 'use_identical', False):
                    self.poses_pnp[selected_i] = self.current_pose[selected_i - 1].detach()
                elif self.cfg.pnp.use_pnp:
                    self.poses_pnp[selected_i] = self.opencv_pnp_ransac(self.matcher_result[selected_i], selected_i,self.Ks,
                                                                        self.current_pose[selected_i - 1].detach().unsqueeze(0),
                                                                        self.render_kwargs)
                self.model_bg.load_dataset(idx=torch.tensor(self.selected_i_train),
                                           image=self.images[self.selected_i_train].permute(0, 3, 1, 2),
                                           depth_range=torch.tensor([[self.near, self.far]]).repeat(len(self.selected_i_train), 1),
                                           intr=self.Ks[self.selected_i_train])

            train_idx = self.i_train[self.selected_i_train]
            if self.cfg.pnp.use_pnp:
                self.current_pose = get_current_pose_pnp(model=self.model_pose, pose_pnp=self.poses_pnp,
                                                         ids=train_idx)
            else:
                self.current_pose = get_current_pose(model=self.model_pose, poses_gt=self.poses)

            optimize_object_nerf = global_step <= self.cfg_train.N_iters and global_step >= start_object

            if global_step == self.cfg_train.N_iters+1:
                self.save_checkpoints(global_step)
                del self.model
                torch.cuda.empty_cache()

            pose_use_deform = optimize_object_nerf and len(self.selected_i_train) > 2

            loss_scalars, loss_weight = edict(), edict()

            self.optimizer.zero_grad(set_to_none=True)
            self.optim_pose.zero_grad()
            self.optimizer_bg_nerf.zero_grad()
            # Initialize losses and PSNR values
            loss, loss_bg, psnr, psnr_bg = 0., 0., 0., 0.

            if optimize_object_nerf:
                # Gather training rays
                target_tr, mask_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = self.gather_training_rays(
                    self.current_pose, train_idx)
                indices = torch.randperm(len(target_tr), device=target_tr.device)[:self.cfg_train.N_rand]
                target, mask = target_tr[indices], mask_tr[indices]
                rays_o, rays_d, viewdirs = rays_o_tr[indices], rays_d_tr[indices], viewdirs_tr[indices]

                # Render and compute object losses
                render_result = self.model(rays_o, rays_d, viewdirs, use_deform=True,
                                           global_step=global_step, **self.render_kwargs)
                loss_scalars, loss_weight, loss = object_losses(render_result, self.cfg_train, target, mask,
                                                                global_step, self.model.N_iters, True)
                psnr = utils.mse2psnr(loss_scalars.img_render.detach()).item()
                optimize_deform_net = True
                # Surface-based perceptual loss
                if self.cfg_train.weight_surface_projection > 0 and IsOptimizePose:
                    rand_id = np.random.randint(len(train_idx))
                    self_id, other_id = self.i_index[rand_id], self.j_index[rand_id]
                    loss_surface_projection = self.get_project_feature_loss(global_step,optimize_deform_net,
                                                                            self.current_pose,
                                                                            imsz[rand_id], target_tr, rays_o_tr,
                                                                            rays_d_tr, [self_id], [other_id])
                    loss_scalars.overlap_pc = loss_surface_projection
                    loss_weight.overlap_pc = self.cfg_train.weight_surface_projection
                    loss += self.cfg_train.weight_surface_projection * loss_surface_projection

                # Projection and near-surface loss
                if self.cfg_train.projection_dis_error > 0:
                    rand_id = np.random.randint(len(train_idx))
                    self_id, other_id = [self.i_index[rand_id]], [self.j_index[rand_id]]
                    mconf_, coord_self, coord_other = self.mconf[self_id], self.coord0[self_id], self.coord1[self_id]
                    projection_dis_error, near_surface_loss = self.get_project_error(global_step, self.current_pose,
                                                                                     coord_self, coord_other, other_id,
                                                                                     self_id, mconf_, pixel_thre=200,
                                                                                     use_deform=pose_use_deform,
                                                                                     **self.render_kwargs)
                    loss_scalars.loss_near_surface = near_surface_loss
                    loss_weight.loss_near_surface = self.cfg_train.weight_near_surface
                    loss += loss_weight.loss_near_surface * near_surface_loss

                    loss_scalars.projection_dis_error = projection_dis_error
                    loss_weight.projection_dis_error = self.cfg_train.projection_dis_error
                    loss += loss_weight.projection_dis_error * projection_dis_error

            # Forward pass for background model and compute MSE loss
            self.model_bg.data_dict.poses_w2c = self.current_pose[train_idx]
            self.model_bg.data_dict.iter = global_step
            self.model_bg.iteration = global_step
            id = random.randrange(len(train_idx))
            self.model_bg.data_dict.corrs_id = [self.i_index[id], self.j_index[id]]
            output_dict, result_dict, plotting_dict = self.model_bg.train_iteration(global_step)
            loss_bg = result_dict["loss"]
            psnr_bg = result_dict['PSNR'].detach().cpu().numpy()
            loss = loss * 0.1 + loss_bg
            loss.backward()
            psnr_bg_lst.append(psnr_bg)
            psnr_lst.append(psnr)
            self.writer.add_scalar('train/lr_pose', self.optim_pose.state_dict()['param_groups'][0]['lr'], global_step)
            self.writer.add_scalar('train/psnr_bg', psnr_bg, global_step)
            if global_step % self.args.i_print == 0:
                with torch.no_grad():
                    split = 'train'
                    _, pose_GT = get_all_training_poses(model=self.model_pose, poses=self.pose_GT, device=device)
                    if self.cfg.pnp.use_pnp:
                        current_pose = get_current_pose_pnp(model=self.model_pose, pose_pnp=self.poses_pnp,
                                                            ids=train_idx, )
                    else:
                        current_pose = get_current_pose(model=self.model_pose, poses_gt=self.pose_GT, ids=train_idx, )

                    pose_aligned, pose_ref = current_pose[train_idx].detach().cpu(), pose_GT[train_idx].detach().cpu()
                    pose_aligned, _ = prealign_w2c_small_camera_systems(pose_aligned, pose_ref)
                    error = evaluate_camera_alignment(pose_aligned, pose_ref)
                    self.writer.add_scalar("{0}/error_R".format(split), error.R.mean(), global_step)
                    self.writer.add_scalar("{0}/error_t".format(split), error.t.mean(), global_step)
                    self.log_scalars(loss=loss_scalars, loss_weight=loss_weight, step=global_step)

                    fig = plt.figure(figsize=(10, 10))

                    output_path = os.path.join(self.cfg.basedir, self.cfg.expname)
                    cam_path = "{}/poses".format(output_path)
                    os.makedirs(cam_path, exist_ok=True)
                    utils_vis.plot_save_poses_blender(fig, pose_aligned, pose_ref, path=cam_path,
                                                      ep=str(len(train_idx))+'_'+str(global_step))
                    plt.close()
                    ep_list.append(global_step)
                eps_time = time.time() - time0
                eps_time_str = f'{eps_time // 3600:02.0f}:{eps_time // 60 % 60:02.0f}:{eps_time % 60:02.0f}'
                self.logger.info(f'Optimize cameras: iter{global_step:3d} / '
                            f'(R/t): {error.R.mean():.3f} / {error.t.mean():.3f} / '
                            f'Loss: {loss:.9f} / PSNR: {np.mean(psnr_lst):5.2f} / PSNR_bg: {np.mean(psnr_bg_lst):5.2f} / '
                            f'Eps: {eps_time_str}')
                psnr_lst, psnr_bg_lst, weight_lst, weight_sum_lst, weight_nonzero_lst, mask_lst, bg_mask_lst, s_val_lst \
                    = [], [], [], [], [], [], [], []
            if global_step % self.args.i_validate == 0: #
                val_dataset = edict()
                val_dataset.idx =  torch.tensor(self.data_dict['i_test'])
                val_dataset.image= self.images[self.data_dict['i_test']].permute(0, 3, 1, 2)
                val_dataset.depth_range= torch.tensor([[self.near, self.far]]).repeat(
                                               len(self.data_dict['i_test']), 1)
                val_dataset.intr= self.Ks[self.data_dict['i_test']]
                val_dataset.pose = self.pose_GT[self.data_dict['i_test']]
                plotting_dict_total = self.model_bg.inference(current_pose[train_idx], pose_GT[train_idx], val_dataset, global_step)

                message = 'VALIDATION IMPROVED ! From current value = {} at iteration {} to current value = {} ' \
                          'at current iteration {}'.format(self.model_bg.best_val, self.model_bg.epoch_of_best_val,
                                                           self.model_bg.current_best_val, self.model_bg.iteration)
                self.logger.critical(message)
                if self.model_bg.current_best_val is not None and self.model_bg.current_best_val < self.model_bg.best_val:
                    message = 'VALIDATION IMPROVED ! From best value = {} at iteration {} to best value = {} ' \
                              'at current iteration {}'.format(self.model_bg.best_val, self.model_bg.epoch_of_best_val,
                                                               self.model_bg.current_best_val, self.model_bg.iteration)
                    self.logger.critical(message)
                    self.model_bg.best_val = self.model_bg.current_best_val  # update best_val
                    self.model_bg.epoch_of_best_val = self.model_bg.iteration  # update_epoch_of_best_val
                    checkpoint_path = os.path.join(self.cfg.basedir, self.cfg.expname)
                    self.model_bg.save_snapshot('model_best.pth.tar', checkpoint_path,
                                               self.optimizer_bg_nerf, self.sched_bg_nerf)
                save_img_path = os.path.join(self.cfg.basedir, self.cfg.expname, 'training_imgs/bg_nerf')

                save_dir = os.path.join(save_img_path, f'{global_step}')
                os.makedirs(save_dir, exist_ok=True)
                for key, value in plotting_dict_total.items():
                    if len(value.shape) == 4:
                        save_path = os.path.join(save_dir, f'{key}.png')
                        save_image(value, save_path, nrow=8, normalize=True)
                    else:
                        save_path = os.path.join(save_dir, f'{key}.png')
                        save_image(value, save_path, normalize=True)

            if optimize_object_nerf:
                wm = render_result['weights'].max(-1)[0]
                ws = render_result['weights'].sum(-1)
                if (wm > 0).float().mean() > 0:
                    psnr_lst.append(psnr)
                    weight_lst.append(wm[wm > 0].mean().detach().cpu().numpy())
                    weight_sum_lst.append(ws[ws > 0].mean().detach().cpu().numpy())
                    weight_nonzero_lst.append((ws > 0).float().mean().detach().cpu().numpy())
                    mask_lst.append(render_result['mask'].float().mean().detach().cpu().numpy())
                    if 'bg_mask' in render_result:
                        bg_mask_lst.append(render_result['bg_mask'].float().mean().detach().cpu().numpy())
                s_val = render_result["s_val"] if "s_val" in render_result else 0
                s_val_lst.append(s_val)
                self.writer.add_scalar('train/psnr', psnr, global_step)
                self.writer.add_scalar('train/sdf_alpha', self.model.sdf_alpha, global_step)
                self.writer.add_scalar('train/sdf_beta', self.model.sdf_beta, global_step)
                self.writer.add_scalar('train/mask', mask_lst[-1], global_step)
                self.writer.add_scalar('train/s_val', s_val, global_step)
                if not getattr(self.cfg_train, 'cosine_lr', ''):
                    decay_steps = self.cfg_train.lrate_decay * 1000
                    decay_factor = 0.1 ** (1 / decay_steps)
                    for i_opt_g, param_group in enumerate(self.optimizer.param_groups):
                        param_group['lr'] = param_group['lr'] * decay_factor
                        self.writer.add_scalar('train/lr_' + param_group['name'], param_group['lr'], global_step)
                if global_step % self.args.i_validate_mesh == 0 or global_step== self.cfg_train.N_iters:
                    self.validate_deform_mesh(128, threshold=0.0, prefix="{}final".format(global_step),
                                         world_space=True,
                                         scale_mats_np=self.data_dict['scale_mats_np'], gt_eval='dtu' in self.cfg.basedir)

                if global_step % self.args.i_validate == 0:
                    render_id = random.randint(0, len(train_idx) - 1)
                    stack_image, psnr = visualize_val_image(self.model, None, self.images[train_idx],
                                                            self.current_pose[train_idx], self.cfg,
                                                            self.Ks[train_idx], self.HW[train_idx],
                                                            global_step, None, self.cfg.data.ndc,
                                                            self.render_kwargs, id=render_id)
                    save_img_path = os.path.join(self.cfg.basedir, self.cfg.expname, 'training_imgs/object_nerf')
                    os.makedirs(save_img_path, exist_ok=True)
                    img_name = 'step-' + str(global_step) + '_id-' + str(render_id) + '.png'
                    save_image(stack_image.permute(2, 0, 1), os.path.join(save_img_path, img_name))

            self.optimizer_bg_nerf.step()
            self.sched_bg_nerf.step()
            if optimize_object_nerf:
                self.optimizer.step()
            if IsOptimizePose:
                self.optim_pose.step()
                self.sched_pose.step()

            if global_step == end_step:
                checkpoint_path = os.path.join(self.cfg.basedir, self.cfg.expname)
                self.model_bg.save_snapshot('model_last.pth.tar', checkpoint_path,
                                            self.optimizer_bg_nerf, self.sched_bg_nerf)
            global_step +=1

    def save_checkpoints(self, global_step):
        """Save model checkpoints."""
        last_ckpt_path = os.path.join(self.cfg.basedir, self.cfg.expname, 'last_ckpt.tar')
        torch.save({
            'global_step': global_step,
            'current_pose': self.current_pose,
            'model_kwargs': self.model.get_kwargs(),
            'MaskCache_kwargs': self.model.get_MaskCache_kwargs(),
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'sched_pose_state_dict': self.sched_pose.state_dict(),
            'optimizer_pose_state_dict': self.optim_pose.state_dict(),
        }, last_ckpt_path)



    def get_bg_model(self, load_latest=False):
        self.logger.info('Creating NerF model for joint pose-NeRF training')
        torch.backends.cudnn.benchmark = True
        data_root = './lib/bg_nerf'
        settings = ws_settings.Settings(data_root)
        settings.data_root = data_root

        train_module_for_launching = 'joint_pose_nerf_training.' + self.cfg.data.dataset_type
        # get the config file
        expr_module = importlib.import_module('lib.bg_nerf.train_settings.{}.{}'.format(train_module_for_launching.replace('/', '.'),
                                          'sparf'))
        expr_func = getattr(expr_module, 'get_config')

        settings.distributed = False
        settings.local_rank = 0
        settings = edict(settings.__dict__)

        # get the config and define the trainer
        model_config = expr_func()
        model_config.train_sub = len(self.i_train)
        model_config.scene = self.cfg.expname
        opt = define_trainer(args=settings, settings_model=model_config, save_option=False)
        self.model_bg = Graph(opt, self.device)
        self.optimizer_bg_nerf, self.sched_bg_nerf = utils.setup_optimizer(self.model_bg, opt)
        self.model_bg.load_dataset(idx=torch.tensor(self.selected_i_train),
                                   image=self.images[self.selected_i_train].permute(0,3,1,2),
                                   depth_range=torch.tensor([[self.near, self.far]]).repeat(len(self.selected_i_train),1),
                                   intr=self.Ks[self.selected_i_train])


        self.model_bg.define_loss_module(self.coord0_scene, self.coord1_scene, self.mconf_scene)
        self.model_bg.settings = opt
        checkpoint_path = os.path.join(self.cfg.basedir, self.cfg.expname, 'model_last.pth.tar')
        latest_step = 0
        if load_latest and os.path.exists(checkpoint_path):
            checkpoint_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            # Load model dict
            self.model_bg.load_state_dict(checkpoint_dict['state_dict'], strict=True)
            self.optimizer_bg_nerf.load_state_dict(checkpoint_dict['optimizer'])
            self.sched_bg_nerf.load_state_dict(checkpoint_dict['scheduler'])
            self.current_pose = checkpoint_dict['current_pose']
            self.model_bg.iteration = checkpoint_dict['iteration']
            self.model_bg.iteration_nerf = checkpoint_dict['iteration_nerf']
            latest_step = checkpoint_dict['iteration'] - 1
        return latest_step, opt

    def forward(self):
        torch.set_grad_enabled(True)
        latest_step, opt = self.get_bg_model()
        self.optimize_increamental(end_step=self.model_bg.settings.max_iter, start_object=0, opt=opt)

    def validate_deform_mesh(self, resolution=128, threshold=0.0, prefix="", world_space=False,
                             scale_mats_np=None, gt_eval=False, smooth=True,
                             extract_color=False):
        os.makedirs(os.path.join(self.cfg.basedir, self.cfg.expname, 'meshes'), exist_ok=True)
        bound_min = self.model.xyz_min.clone().detach().float()
        bound_max = self.model.xyz_max.clone().detach().float()

        gt_path = os.path.join(self.cfg.data.datadir, "stl_total.ply") if gt_eval else ''
        vertices0, triangles = self.model.extract_deform_geometry(bound_min, bound_max, resolution=resolution,
                                                             threshold=threshold, scale_mats_np=scale_mats_np,
                                                             gt_path=gt_path, smooth=smooth,
                                                             )

        if world_space and scale_mats_np is not None:
            vertices = vertices0 * scale_mats_np[0, 0] + scale_mats_np[:3, 3][None]
        else:
            vertices = vertices0

        if extract_color:
            # use normal direction as the viewdir
            ray_pts = torch.from_numpy(vertices0).cuda().float().split(8192 * 32, 0)
            vertex_colors = [self.model.mesh_color_forward(pts) for pts in ray_pts]
            vertex_colors = (torch.concat(vertex_colors).cpu().detach().numpy() * 255.).astype(np.uint8)
            mesh = trimesh.Trimesh(vertices, triangles, vertex_colors=vertex_colors)
        else:
            mesh = trimesh.Trimesh(vertices, triangles)
        mesh_path = os.path.join(self.cfg.basedir, self.cfg.expname, 'meshes', "deform"+ prefix + '.ply')
        mesh.export(mesh_path)
        self.logger.info("deform mesh saved at " + mesh_path)
        return 0

    def run_eval(self, settings, out_dir, expname, plot=True, save_ind_files= True, split='i_test'):
        """ Run final evaluation on the test set. Computes novel-view synthesis performance.
        When the poses were optimized, also computes the pose registration error. Optionally, one can run
        test-time pose optimization to factor out the pose error from the novel-view synthesis performance.
        """
        model_name = settings.model
        args = settings
        # the loss is redefined here as only the photometric one, in case the
        # test-time photometric optimization is used
        args.loss_type = 'photometric'
        args.loss_weight.render = 0.

        # load the test step
        args.val_sub = None
        val_dataset = edict()
        val_dataset.idx = torch.tensor(self.data_dict[split])
        val_dataset.image = self.images[self.data_dict[split]].permute(0, 3, 1, 2)
        val_dataset.depth_range = torch.tensor([[self.near, self.far]]).repeat(
            len(self.data_dict[split]), 1)
        val_dataset.intr = self.Ks[self.data_dict[split]]
        val_dataset.pose = self.pose_GT[self.data_dict[split]]
        pose_GT = self.pose_GT[self.data_dict['i_train']]
        pose = self.current_pose[self.data_dict['i_train']].cuda()

        print("saving results to {}...".format(out_dir))
        os.makedirs(out_dir, exist_ok=True)



        save_all = {}
        test_optim_options = [True] if model_name in ["joint_pose_nerf_training", 'nerf_fixed_noisy_poses'] else [
            False]  # , False
        for test_optim in test_optim_options:
            print('test pose optim : {}'.format(test_optim))
            args.optim.test_photo = test_optim

            possible_to_plot = True
            if test_optim is False and model_name in ["joint_pose_nerf_training", 'nerf_fixed_noisy_poses']:
                possible_to_plot = False
            results_dict = self.model_bg.evaluate_full(pose, pose_GT, val_dataset, args, plot=plot and possible_to_plot,
                                                 save_ind_files=save_ind_files and possible_to_plot,
                                                 out_scene_dir=out_dir)
            self.logger.info("--------------------------")
            self.logger.info("rot:   {:8.3f}".format(results_dict['rot_error']))
            self.logger.info("trans: {:10.5f}".format(results_dict['trans_error']))
            self.logger.info("--------------------------")
            if test_optim:
                save_all['w_test_optim'] = results_dict
            elif model_name in ["joint_pose_nerf_training", 'nerf_fixed_noisy_poses']:
                save_all['without_test_optim'] = results_dict
            else:
                # nerf
                save_all = results_dict

        save_all['iteration'] = self.model_bg.iteration

        # name_file = '{}.txt'.format(expname)
        name_file = '{}.json'.format(expname)

        print('Saving json file to {}/{}'.format(out_dir, name_file))
        with open("{}/{}".format(out_dir, name_file), "w") as f:
            json.dump(save_all, f, indent=4)
        return


    def novel_view(self, settings, out_dir, split='i_train'):
        args = settings
        args.val_sub = None
        dataset = edict()
        dataset.idx = torch.tensor(self.data_dict[split])
        dataset.image = self.images[self.data_dict[split]].permute(0, 3, 1, 2)
        dataset.depth_range = torch.tensor([[self.near, self.far]]).repeat(
            len(self.data_dict[split]), 1)
        dataset.intr = self.Ks[self.data_dict[split]]
        dataset.pose = self.pose_GT[self.data_dict[split]]
        pose_GT = self.pose_GT[self.data_dict['i_train']]
        pose = self.current_pose[self.data_dict['i_train']].cuda()
        self.model_bg.generate_videos_synthesis(pose, pose_GT, args,dataset, out_scene_dir=out_dir)