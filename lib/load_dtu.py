import os
from glob import glob

import cv2 as cv
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use('Agg')
import imageio
import cv2
from lib.utils_vis import load_matching_network, matching_pair
# This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose



def load_dtu_data(basedir, normalize=True, reso_level=2, mask=True, white_bg=True,
                  matching_config=None, selected_id=None):


    i_train = selected_id
    exclude_idx = [3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 36, 37, 38, 39]
    i_test = [i for i in np.arange(49) if i not in i_train + exclude_idx]
    rgb_paths = sorted(glob(os.path.join(basedir, 'image', '*png')))
    if len(rgb_paths) == 0:
        rgb_paths = sorted(glob(os.path.join(basedir, 'image', '*jpg')))
    if len(rgb_paths) == 0:
        rgb_paths = sorted(glob(os.path.join(basedir, 'rgb', '*png')))

    splits = ['train', 'val', 'test']

    mask_paths = sorted(glob(os.path.join(basedir.replace('DTU', 'idrmasks'), 'mask', '*png')))
    if len(mask_paths) == 0:
        mask_paths = sorted(glob(os.path.join(basedir.replace('DTU', 'idrmasks'), 'mask', '*jpg')))

    render_cameras_name = 'cameras.npz' #'cameras_sphere.npz' if normalize else 'cameras_large.npz'
    camera_dict = np.load(os.path.join(basedir, render_cameras_name))
    world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(len(rgb_paths))]
    if normalize:
        scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(len(rgb_paths))]
    else:
        scale_mats_np = None
    all_intrinsics = []
    all_poses = []
    all_imgs = []
    imgs_gray = []
    all_masks = []
    counts = [0]
    for s in splits:
        imgs = []
        poses = []
        masks = []
        if s=='train':
            exclude_idx = i_train
        else:
            exclude_idx = i_test
        for i in exclude_idx:
            world_mat = world_mats_np[i]
            im_name = rgb_paths[i]
            if i not in exclude_idx:
                continue
            if normalize:
                P = world_mat @ scale_mats_np[i]
            else:
                P = world_mat
            P = P[:3, :4]
            intrinsics, pose_c2w = load_K_Rt_from_P(None, P)
            all_intrinsics.append(intrinsics[:3,:3])
            pose = np.linalg.inv(pose_c2w)[:3,:]
            poses.append(pose)
            if len(mask_paths) > 0:
                mask_ = (imageio.imread(mask_paths[i]) / 255.).astype(np.float32)
                if mask_.ndim == 3:
                    masks.append(mask_[...,:3])
                else:
                    masks.append(np.repeat(mask_[...,None], 3, axis=-1))
            imgs.append((imageio.imread(im_name) / 255.).astype(np.float32))
        imgs = np.array(imgs).astype(np.float32)  # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        masks = np.array(masks).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_masks.append(masks)
        all_poses.append(poses)

    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)]
    i_train, i_val, i_test = i_split
    imgs = np.concatenate(all_imgs, 0)
    masks = np.concatenate(all_masks, 0)
    poses = np.concatenate(all_poses, 0)
    H, W = imgs[0].shape[:2]
    K = all_intrinsics[0]
    focal = all_intrinsics[0][0,0]
    print("Date original shape: ", H, W)
    masks = F.interpolate(torch.from_numpy(masks).permute(0, 3, 1, 2), size=(H, W)).permute(0, 2, 3, 1).numpy()
    # if mask:
    #     assert len(mask_paths) > 0
    #     bg = 1. #if white_bg else 0.
    #     imgs = imgs * masks + bg * (1 - masks)

    render_poses = poses[i_split[-1]]

    imgs_matching = np.zeros_like(imgs)
    imgs_matching[masks > 0] = imgs[masks > 0]
    max_matcher = matching_config.max_matcher
    matching_outdoor, sg_config = load_matching_network(matching_config)
    matching_config.superglue = 'indoor'
    matching_indoor, sg_config = load_matching_network(matching_config)
    images_object = torch.tensor(imgs_matching[i_train][..., 0:3]).permute(0, 3, 1, 2)
    def matching_batch(imgs_matching, matching_model, type=''):
        with torch.no_grad():
            matcher_infos_list = []
            resize_scale = 1
            for i in range(i_train.shape[0]):
                matcher_infos = []
                candidate = list(range(i - 1, -1, -1)) + list(range(i + 1, i_train.shape[0], 1))
                candidate = candidate[0:1]
                for j in candidate:
                    image0 = cv2.resize(imgs_matching[i][..., 0:3], dsize=None, fx=resize_scale, fy=resize_scale)
                    image1 = cv2.resize(imgs_matching[j][..., 0:3], dsize=None, fx=resize_scale, fy=resize_scale)
                    imgs_gray.append(image0[...,0])
                    mask_img = cv2.resize(imgs_matching[j][..., -1], dsize=None, fx=resize_scale, fy=resize_scale)
                    mask_img = torch.from_numpy(mask_img).to(matching_config.device)
                    matcher_info = matching_pair(matching_model, matching_config, image0, image1, mask_img,
                                                 reso_level, resize_scale, i, j, max_matcher, type)
                    matcher_infos.append(matcher_info)
                matcher_infos_list.append(matcher_infos)
            return matcher_infos_list

    matcher_infos_object = matching_batch(imgs_matching, matching_indoor)
    matcher_infos_scene = matching_batch(imgs, matching_indoor, type='scene')
    matcher_infos_list = matcher_infos_object + matcher_infos_scene

    if reso_level > 1:
        H, W = int(H / reso_level), int(W / reso_level)
        imgs =  F.interpolate(torch.from_numpy(imgs).permute(0,3,1,2), size=(H, W)).permute(0,2,3,1).numpy()
        if masks is not None:
            masks =  F.interpolate(torch.from_numpy(masks).permute(0,3,1,2), size=(H, W)).permute(0,2,3,1).numpy()
        gray_half_res = np.zeros((imgs_gray.shape[0], H, W)).astype(np.float32)
        for i, gray in enumerate(imgs_gray):
            gray_half_res[i] = cv2.resize(gray, (W, H), interpolation=cv2.INTER_AREA)
        imgs_gray = gray_half_res
        K[:2] /= reso_level
        focal /= reso_level

    del matching_outdoor ,matching_indoor
    torch.cuda.empty_cache()
    return imgs,imgs_gray, poses, render_poses, [H, W, focal], K, i_split, \
        scale_mats_np[0], masks[...,[0]],matcher_infos_list, sg_config, images_object,np.eye(4)

class Dataset:
    def __init__(self, conf):
        super(Dataset, self).__init__()
        print('Load data: Begin')
        self.device = torch.device('cuda')
        self.conf = conf

        self.data_dir = conf.get_string('data_dir')
        self.render_cameras_name = conf.get_string('render_cameras_name')
        self.object_cameras_name = conf.get_string('object_cameras_name')

        self.camera_outside_sphere = conf.get_bool('camera_outside_sphere', default=True)
        self.scale_mat_scale = conf.get_float('scale_mat_scale', default=1.1)

        camera_dict = np.load(os.path.join(self.data_dir, self.render_cameras_name))
        self.camera_dict = camera_dict
        self.images_lis = sorted(glob(os.path.join(self.data_dir, 'image/*.png')))
        self.n_images = len(self.images_lis)
        self.images_np = np.stack([cv.imread(im_name) for im_name in self.images_lis]) / 256.0
        self.masks_lis = sorted(glob(os.path.join(self.data_dir, 'mask/*.png')))
        self.masks_np = np.stack([cv.imread(im_name) for im_name in self.masks_lis]) / 256.0

        # world_mat is a projection matrix from world to image
        self.world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.scale_mats_np = []

        # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
        self.scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.intrinsics_all = []
        self.pose_all = []

        for scale_mat, world_mat in zip(self.scale_mats_np, self.world_mats_np):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())

        self.images = torch.from_numpy(self.images_np.astype(np.float32)).cpu()  # [n_images, H, W, 3]
        self.masks  = torch.from_numpy(self.masks_np.astype(np.float32)).cpu()   # [n_images, H, W, 3]
        self.intrinsics_all = torch.stack(self.intrinsics_all).to(self.device)   # [n_images, 4, 4]
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)  # [n_images, 4, 4]
        self.focal = self.intrinsics_all[0][0, 0]
        self.pose_all = torch.stack(self.pose_all).to(self.device)  # [n_images, 4, 4]
        self.H, self.W = self.images.shape[1], self.images.shape[2]
        self.image_pixels = self.H * self.W

        object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
        object_bbox_max = np.array([ 1.01,  1.01,  1.01, 1.0])
        # Object scale mat: region of interest to **extract mesh**
        object_scale_mat = np.load(os.path.join(self.data_dir, self.object_cameras_name))['scale_mat_0']
        object_bbox_min = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_min[:, None]
        object_bbox_max = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_max[:, None]
        self.object_bbox_min = object_bbox_min[:3, 0]
        self.object_bbox_max = object_bbox_max[:3, 0]

        print('Load data: End')

    def near_far_from_sphere(self, rays_o, rays_d):
        a = torch.sum(rays_d**2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        return near, far

    def image_at(self, idx, resolution_level):
        img = cv.imread(self.images_lis[idx])
        return (cv.resize(img, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255)