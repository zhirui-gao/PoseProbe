import glob
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as torch_F
from einops import rearrange
from typing import List, Any, Dict, Tuple
from external.SuperGlue.models.matching import Matching
import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm
from lib.bg_nerf.source.datasets.base import Dataset
from lib.bg_nerf.source.utils import camera as camera
import matplotlib.pyplot as plt

def matching_pair(matching, matching_config, image0, image1,
                  mask_img,factor,resize_scale,i,j, max_matcher,scene=''):

    pred = matching({'image0': image1, 'image1': image0})
    pred = {k: v[0] for k, v in pred.items()}  # .cpu().numpy()
    matches, conf = pred['matches0'], pred['matching_scores0']
    valid = mask_img[pred['keypoints0'][:, 1].long(), pred['keypoints0'][:, 0].long()] > 0
    kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
    valid = (matches > -1) * valid
    mkpts0 = kpts0[valid] / (factor * resize_scale)
    mkpts1 = kpts1[matches[valid]] / (factor * resize_scale)
    mconf = conf[valid]
    if matching_config.use_kornia:
        text = [
            'loftr',
            'Matches: {}'.format(len(mkpts0)),
        ]
    else:
        text = [
            'SuperGlue',
            'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
            'Matches: {}'.format(len(mkpts0)),
        ]
    # padding to fixed number
    print(i, '_', j, 'correspondences:', mconf.shape[0], 'confidenct:', mconf.mean())
    mkpts0 = torch_F.pad(mkpts0, (0, 0, 0, max_matcher - mkpts0.shape[0]))
    mkpts1 = torch_F.pad(mkpts1, (0, 0, 0, max_matcher - mkpts1.shape[0]))
    mconf = torch_F.pad(mconf, (0, max_matcher - mconf.shape[0]))
    mconf = rearrange(mconf, 'c->c 1')
    matcher_info = torch.cat((mkpts0, mkpts1, mconf), dim=1)  # [1024,5]

    color = cm.jet(mconf.detach().cpu().numpy())
    image0 = image0.detach().cpu().numpy() * 255
    image1 = image1.detach().cpu().numpy() * 255
    make_matching_plot(
        image1[0][0], image0[0][0], factor * mkpts0.detach().cpu().numpy(), factor * mkpts1.detach().cpu().numpy(),
                                    factor * mkpts0.detach().cpu().numpy(),
                                    factor * mkpts1.detach().cpu().numpy(), color,
        text, './matching_imgs/' + str(i) + '_' + str(j) + scene + '.png', False,
        True, True, 'Matches')
    return matcher_info

def grayscale(colors):
    """Return grayscale of given color."""
    gray = lambda rgb: np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
    gray_img = gray(colors)
    # plt.imshow(gray_img, cmap=plt.get_cmap(name='gray'))
    # plt.show()
    return gray_img

def plot_image_pair(imgs, dpi=100, size=6, pad=.5):
    n = len(imgs)
    assert n == 2, 'number of images must be two'
    figsize = (size*n, size*3/4) if size is not None else None
    _, ax = plt.subplots(1, n, figsize=figsize, dpi=dpi)
    for i in range(n):
        ax[i].imshow(imgs[i], cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        for spine in ax[i].spines.values():  # remove frame
            spine.set_visible(False)
    plt.tight_layout(pad=pad)

def plot_keypoints(kpts0, kpts1, color='w', ps=2):
    ax = plt.gcf().axes
    ax[0].scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps)
    ax[1].scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps)


def plot_matches(kpts0, kpts1, color, lw=1.5, ps=4):
    fig = plt.gcf()
    ax = fig.axes
    fig.canvas.draw()

    transFigure = fig.transFigure.inverted()
    fkpts0 = transFigure.transform(ax[0].transData.transform(kpts0))
    fkpts1 = transFigure.transform(ax[1].transData.transform(kpts1))

    fig.lines = [matplotlib.lines.Line2D(
        (fkpts0[i, 0], fkpts1[i, 0]), (fkpts0[i, 1], fkpts1[i, 1]), zorder=1,
        transform=fig.transFigure, c=color[i], linewidth=lw)
                 for i in range(len(kpts0))]
    ax[0].scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps)
    ax[1].scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps)


def make_matching_plot(image0, image1, kpts0, kpts1, mkpts0, mkpts1,
                       color, text, path, show_keypoints=False,
                       fast_viz=False, opencv_display=False,
                       opencv_title='matches', small_text=[]):
    plot_image_pair([image0, image1])
    if show_keypoints:
        plot_keypoints(kpts0, kpts1, color='k', ps=4)
        plot_keypoints(kpts0, kpts1, color='w', ps=2)
    plot_matches(mkpts0, mkpts1, color)

    fig = plt.gcf()
    txt_color = 'k' if image0[:100, :150].mean() > 200 else 'w'
    fig.text(
        0.01, 0.99, '\n'.join(text), transform=fig.axes[0].transAxes,
        fontsize=15, va='top', ha='left', color=txt_color)

    txt_color = 'k' if image0[-100:, :150].mean() > 200 else 'w'
    fig.text(
        0.01, 0.01, '\n'.join(small_text), transform=fig.axes[0].transAxes,
        fontsize=5, va='bottom', ha='left', color=txt_color)

    plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
    plt.close()


@torch.no_grad()
def load_matching_network(opt):
    config = {
        'superpoint': {
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints
        },
        'superglue': {
            'weights': opt.superglue,
            'max_matcher': opt.max_matcher,
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
        }
    }
    matching = Matching(config).eval().to(opt.device)
    return matching, config

def as_intrinsics_matrix(intrinsics: np.ndarray) -> np.ndarray:
    """
    Get matrix representation of intrinsics.
    """
    K = np.eye(3)
    K[0, 0] = intrinsics[0]
    K[1, 1] = intrinsics[1]
    K[0, 2] = intrinsics[2]
    K[1, 2] = intrinsics[3]
    return K


class BaseRGBDDataset(Dataset):
    def __init__(self, args: Dict[str, Any], split: str, scale: float = 1.):
        super().__init__(args, split)
        self.scale = scale
        self.png_depth_scale = None
        self.distortion = None

    def compute_3d_bounds(self, opt: Dict[str, Any], H: int, W: int, intrinsics: np.ndarray,
                          poses_w2c: np.ndarray, depth_range: List[float]) -> np.ndarray:
        """Computes the center of the 3D points corresponding to the far range. This
        will be used to re-center the cameras, such as the scene is more or less
        centered at the origin. """
        poses_w2c = torch.from_numpy(poses_w2c).float()
        intrinsics = torch.from_numpy(intrinsics).float()
        # Compute boundaries of 3D space
        max_xyz = torch.full((3,), -1e6).cpu()
        min_xyz = torch.full((3,), 1e6).cpu()
        near, far = depth_range

        rays_o, rays_d = camera.get_center_and_ray(poses_w2c[:, :3], H, W, intr=intrinsics)
        # (B, HW, 3), (B, HW, 3)

        points_3D_max = rays_o + rays_d * far  # [H, W, 3]
        points_3D = points_3D_max

        max_xyz = torch.max(points_3D.view(-1, 3).amax(0), max_xyz)
        min_xyz = torch.min(points_3D.view(-1, 3).amin(0), min_xyz)
        bb_center = (max_xyz + min_xyz).detach() / 2.
        return bb_center.numpy()

    def get_all_camera_poses(self, args: Dict[str, Any]):
        # of the current split
        if isinstance(self.render_poses_c2w, list):
            return torch.from_numpy(np.linalg.inv(np.stack(self.render_poses_c2w, axis=0)))[:, :3].float()  # (B, 3, 4)
        else:
            # numpy array
            return torch.from_numpy(np.linalg.inv(self.render_poses_c2w))[:, :3].float()  # (B, 3, 4)

    def read_image_and_depth(self, color_path: str, depth_path: str, K: np.ndarray
                             ) -> Tuple[np.ndarray, np.ndarray]:
        color_data = cv2.imread(color_path)
        if '.png' in depth_path:
            depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        else:
            raise ValueError

        if self.distortion is not None:
            # undistortion is only applied on color image, not depth!
            color_data = cv2.undistort(color_data, K, self.distortion)

        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        depth_data = depth_data.astype(np.float32) / self.png_depth_scale
        H, W = depth_data.shape
        # the intrinsics correspond to the depth size, so the image need to be resized to depth size.
        color_data = cv2.resize(color_data, (W, H))
        return color_data, depth_data

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Args:
            idx (int)

        Returns:
            a dictionary for each image index containing the following elements:
                * idx: the index of the image
                * rgb_path: the path to the RGB image. Will be used to save the renderings with a name.
                * image: the corresponding image, a torch Tensor of shape [3, H, W]. The RGB values are
                            normalized to [0, 1] (not [0, 255]).
                * intr: intrinsics parameters, numpy array of shape [3, 3]
                * pose:  world-to-camera transformation matrix in OpenCV format, numpy array of shaoe [3, 4]
                * depth_range: depth_range, numpy array of shape [1, 2]
                * scene: self.scenes[render_scene_id]

                * depth_gt: ground-truth depth map, numpy array of shape [H, W]
                * valid_depth_gt: mask indicating where the depth map is valid, bool numpy array of shape [H, W]
        """

        rgb_file = self.render_rgb_files[idx]
        depth_file = self.render_depth_files[idx]
        scene = self.scene
        render_pose_c2w = self.render_poses_c2w[idx].copy()
        render_intrinsics = self.intrinsics.copy()

        rgb, depth = self.read_image_and_depth(rgb_file, depth_file, render_intrinsics)
        depth = depth * self.scale

        render_pose_c2w[:3, 3] *= self.scale
        render_pose_w2c = np.linalg.inv(render_pose_c2w)

        edge_h = self.crop_edge_h
        edge_w = self.crop_edge_w
        if edge_h > 0 or edge_w > 0:
            # crop image edge, there are invalid value on the edge of the color image
            rgb = rgb[edge_h:-edge_h, edge_w:-edge_w]
            depth = depth[edge_h:-edge_h, edge_w:-edge_w]
            # need to adapt the intrinsics accordingly
            render_intrinsics[0, 2] -= edge_w
            render_intrinsics[1, 2] -= edge_h

        # process by resizing and cropping
        rgb, render_intrinsics, depth = self.preprocess_image_and_intrinsics(
            rgb, intr=render_intrinsics, depth=depth, channel_first=False)

        ret = {
            'idx': idx,
            'rgb_path': rgb_file,
            'image': rgb.permute(2, 0, 1),  # torch tensor (3, self.H, self.W)
            'depth_gt': depth,  # (H, W)
            'valid_depth_gt': depth > 0.,
            'intr': render_intrinsics[:3, :3].astype(np.float32),  # 3x3
            'pose': render_pose_w2c[:3].astype(np.float32),  # # 3x4, world to camera
            'scene': self.scene
        }

        depth_range = torch.tensor([self.near * (1 - self.args.increase_depth_range_by_x_percent),
                                    self.far * (1 + self.args.increase_depth_range_by_x_percent)])
        ret['depth_range'] = depth_range
        return ret


class ReplicaPerScene(BaseRGBDDataset):
    def __init__(self, args: Dict[str, Any], split: str, scenes: str, scale: float = 1.):
        super(ReplicaPerScene, self).__init__(args, split, scale)
        self.args.resize=[340, 600]

        self.scene = scenes
        self.input_folder = args.datadir
        self.color_paths = np.array(sorted(
            glob.glob(f'{self.input_folder}/results/frame*.jpg')))
        self.depth_paths = np.array(sorted(
            glob.glob(f'{self.input_folder}/results/depth*.png')))
        self.n_img = len(self.color_paths)
        self.load_poses(f'{self.input_folder}/traj.txt')  # camera to world in self.poses_c2w

        # fixed for the whole dataset
        self.H, self.W, fx, fy, cx, cy = 680, 1200, 600.0, 600.0, 599.5, 339.5
        self.intrinsics = as_intrinsics_matrix((fx, fy, cx, cy))
        self.png_depth_scale = 6553.5  # for depth image in png format
        self.crop_edge_w = self.crop_edge_h = 0  # does not crop any edges

        self.scale = 1.  # for the translation pose and the depth

        # recenter all the cameras
        translation_recenter = True
        if translation_recenter:
            avg_trans = self.poses_c2w[:, :3, -1].mean(0).reshape(1, -1)
            self.poses_c2w[:, :3, -1] -= avg_trans



    def load_replica_data(self, matching_config=None, seleted_id_arg=None):
        if self.scene == 'room1' or self.scene == 'office1' or self.scene == 'office0':
            self.near, self.far = 0.1, 4.5
        else:
            self.near, self.far = 0.1, 6.5
        start = 0
        n_train_img = len(self.poses_c2w)
        if self.scene == 'office0':
            test_interval = 10
        elif self.scene == 'office1':
            test_interval = 50
        elif self.scene == 'office2':
            test_interval = 10
        elif self.scene == 'office3':
            test_interval = 30
        elif self.scene == 'office4':
            test_interval = 30
        elif self.scene == 'room0':
            test_interval = 10
        elif self.scene == 'room1':
            test_interval = 10
        else:
            test_interval = 10
        i_train = np.arange(start, n_train_img)
        if seleted_id_arg is not None:
            i_train = i_train[seleted_id_arg].astype(np.int_)
            start = seleted_id_arg[0]
        end_test = i_train[-1] + test_interval
        i_test = np.array([int(j) for j in np.arange(start, end_test) if (j not in i_train)])[::test_interval]
        c2w_poses = self.poses_c2w.copy()
        depth_range = [self.near, self.far]
        train_poses_c2w = c2w_poses[i_train]
        # this is very IMPORTANT
        # need to adjust the pose such as the scene is centered around 0 for the
        # selected training images
        # otherwise, the range of 3D points covered by the scenes is very far off.
        avg_trans = self.compute_3d_bounds(None, self.H, self.W, self.intrinsics,
                                           np.linalg.inv(train_poses_c2w), depth_range)
        # rescale all the poses
        c2w_poses[:, :3, -1] -= avg_trans.reshape(1, -1)
        self.poses_c2w[:, :3, -1] -= avg_trans.reshape(1, -1)


        all_imgs = []
        all_masks = []
        imgs_gray = []
        all_poses = []
        all_intrs = []
        counts = [0]
        splits = ['train', 'val', 'test']
        for s in splits:
            imgs = []
            poses = []
            masks = []
            intrs = []
            if s == 'train':
                exclude_idx = i_train
            else:
                exclude_idx = i_test

            for idx in exclude_idx:
                rgb_file = self.color_paths[idx]
                depth_file = self.depth_paths[idx]
                render_pose_c2w = c2w_poses[idx].copy()
                render_intrinsics = self.intrinsics.copy()
                rgb, depth = self.read_image_and_depth(rgb_file, depth_file, render_intrinsics)

                depth = depth * self.scale
                render_pose_c2w[:3, 3] *= self.scale
                render_pose_w2c = np.linalg.inv(render_pose_c2w)
                edge_h = self.crop_edge_h
                edge_w = self.crop_edge_w
                if edge_h > 0 or edge_w > 0:
                    # crop image edge, there are invalid value on the edge of the color image
                    rgb = rgb[edge_h:-edge_h, edge_w:-edge_w]
                    depth = depth[edge_h:-edge_h, edge_w:-edge_w]
                    # need to adapt the intrinsics accordingly
                    render_intrinsics[0, 2] -= edge_w
                    render_intrinsics[1, 2] -= edge_h

                # process by resizing and cropping
                rgb, render_intrinsics, depth = self.preprocess_image_and_intrinsics(
                    rgb, intr=render_intrinsics, depth=depth, channel_first=False)


                poses.append(render_pose_w2c)
                imgs.append(rgb)
                masks.append(depth > 0)
                intrs.append(render_intrinsics[:3, :3].astype(np.float32))

            imgs = np.stack(imgs, axis=0)
            poses = np.array(poses).astype(np.float32)
            intrs = np.array(intrs).astype(np.float32)
            masks = (np.array(masks)).astype(np.float32)



            counts.append(counts[-1] + imgs.shape[0])
            all_imgs.append(imgs)
            all_masks.append(masks)
            all_poses.append(poses)
            all_intrs.append(intrs)
        i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)]
        i_train, i_val, i_test = i_split
        imgs = np.concatenate(all_imgs, 0)
        masks = np.concatenate(all_masks, 0)
        poses = np.concatenate(all_poses, 0)
        all_intrs = np.concatenate(all_intrs, 0)

        H, W = imgs[0].shape[:2]

        align_pose = np.eye(4)


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
                        image0 = torch.from_numpy(grayscale(image0)[None, None]).to(
                            matching_config.device).float()
                        image1 = torch.from_numpy(grayscale(image1)[None, None]).to(
                            matching_config.device).float()

                        imgs_gray.append(image0.squeeze(0))
                        mask_img = cv2.resize(imgs_matching[j][..., -1], dsize=None, fx=resize_scale, fy=resize_scale)
                        mask_img = torch.from_numpy(mask_img).to(matching_config.device)
                        matcher_info = matching_pair(matching_model, matching_config, image0, image1, mask_img,
                                                     1., resize_scale, i, j, max_matcher, type)
                        matcher_infos.append(matcher_info)
                    matcher_infos_list.append(matcher_infos)
                return matcher_infos_list

        matcher_infos_object = matching_batch(imgs_matching, matching_outdoor)
        matcher_infos_scene = matching_batch(imgs, matching_indoor, type='scene')
        matcher_infos_list = matcher_infos_object + matcher_infos_scene

        del matching_outdoor, matching_indoor
        torch.cuda.empty_cache()
        focal = all_intrs[0][0, 0]
        poses = poses[...,:3,:]
        render_poses= poses
        return imgs, masks[..., None], imgs_gray, poses, render_poses, [H, W,focal], all_intrs, i_split, \
            matcher_infos_list, sg_config, images_object, align_pose


    def __len__(self):
        return len(self.render_rgb_files)

    def define_train_and_test_splits(self, color_paths: List[str], depth_paths: List[str], c2w_poses: np.ndarray):

        # these were compute with fps.
        # define depth ranges
        if self.scene == 'room1' or self.scene == 'office1' or self.scene == 'office0':
            self.near, self.far = 0.1, 4.5
        else:
            self.near, self.far = 0.1, 6.5

        start = 0
        n_train_img = len(self.poses_c2w)
        if self.scene == 'office0':
            if self.args.train_sub is not None and self.args.train_sub > 3:
                train_interval = 50
            else:
                train_interval = 80
            test_interval = 10
        elif self.scene == 'office1':
            if self.args.train_sub is not None and self.args.train_sub > 6:
                train_interval = 80
            elif self.args.train_sub is not None and self.args.train_sub > 3:
                train_interval = 100
            else:
                train_interval = 200
            test_interval = 50
        elif self.scene == 'office2':
            if self.args.train_sub is not None and self.args.train_sub > 6:
                train_interval = 80
            elif self.args.train_sub is not None and self.args.train_sub > 3:
                train_interval = 100
            else:
                train_interval = 150
            test_interval = 10
        elif self.scene == 'office3':
            if self.args.train_sub is not None and self.args.train_sub > 3:
                train_interval = 200
            else:
                train_interval = 350
            test_interval = 30
        elif self.scene == 'office4':
            start = 850
            train_interval = 100
            test_interval = 30
        elif self.scene == 'room0':
            if self.args.train_sub is not None and self.args.train_sub > 3:
                train_interval = 100
            else:
                train_interval = 250
            test_interval = 10
        elif self.scene == 'room1':
            if self.args.train_sub is not None and self.args.train_sub > 3:
                start = 300
                train_interval = 100
            else:
                train_interval = 50
            test_interval = 10
        else:
            train_interval = 80
            test_interval = 10
        i_train = np.arange(start, n_train_img)[::train_interval].astype(np.int_)

        if self.args.train_sub is not None:
            i_train = i_train[:self.args.train_sub]

        end_test = i_train[-1] + test_interval
        i_test = np.array([int(j) for j in np.arange(start, end_test) if (j not in i_train)])[::test_interval]

        train_rgb_files = color_paths[i_train]
        train_depth_files = depth_paths[i_train]
        train_poses_c2w = c2w_poses[i_train]

        depth_range = [self.near, self.far]

        # this is very IMPORTANT
        # need to adjust the pose such as the scene is centered around 0 for the
        # selected training images
        # otherwise, the range of 3D points covered by the scenes is very far off.
        avg_trans = self.compute_3d_bounds(self.args, self.H, self.W, self.intrinsics,
                                           np.linalg.inv(train_poses_c2w), depth_range)
        # rescale all the poses
        c2w_poses[:, :3, -1] -= avg_trans.reshape(1, -1)
        self.poses_c2w[:, :3, -1] -= avg_trans.reshape(1, -1)
        train_poses_c2w = c2w_poses[i_train]

        test_rgb_files = color_paths[i_test]
        test_depth_files = depth_paths[i_test]
        test_poses_c2w = c2w_poses[i_test]

        if self.split == 'train':
            self.render_rgb_files = train_rgb_files
            self.render_poses_c2w = train_poses_c2w
            self.render_depth_files = train_depth_files
        else:
            self.render_rgb_files = test_rgb_files
            self.render_poses_c2w = test_poses_c2w
            self.render_depth_files = test_depth_files

    def load_poses(self, path: str):
        poses_c2w = []
        with open(path, "r") as f:
            lines = f.readlines()
        for i in range(self.n_img):
            line = lines[i]
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4).astype(np.float32)
            poses_c2w.append(c2w)
        self.poses_c2w = np.stack(poses_c2w, axis=0)  # (B, 4, 4)
        self.valid_poses = np.arange(self.poses_c2w.shape[0]).tolist()
        # here, all the poses are valid
        return True

