import os
import torch
import numpy as np
import imageio
import json
import torch.nn.functional as F
import cv2
from external.SuperGlue.models.matching import Matching
from einops import rearrange
import torch.nn.functional as torch_F
import kornia
import matplotlib
import tqdm
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from lib import camera

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


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

def grayscale(colors):
    """Return grayscale of given color."""
    gray = lambda rgb: np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
    gray_img = gray(colors)
    # plt.imshow(gray_img, cmap=plt.get_cmap(name='gray'))
    # plt.show()
    return gray_img


def parse_raw_camera(pose_raw):
    pose_flip = camera.pose(R=torch.diag(torch.tensor([1,-1,-1]))) # right, up, backward -> right, down, forward
    pose = camera.pose.compose([pose_flip,pose_raw[:3]])
    pose = camera.pose.invert(pose) # c2w -> w2c
    return pose

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



def load_custom_data(basedir, factor=2, testskip=1, trainskip=1,matching_config=None,seleted_id_arg=None):
    splits = ['train', 'val', 'test']
    all_imgs = []
    all_masks = []
    imgs_gray = []
    all_poses = []
    counts = [0]

    pose_w2c = np.load(os.path.join(basedir,'w2c.npy'))


    for s in splits:
        imgs = []
        poses = []
        masks = []
        if s == 'train':
            selected_id = seleted_id_arg  # [30,34,38,42,46,50]
            if selected_id is not None:
                trainskip = 1
                top_n = -1
            skip = trainskip  # 1
        else:
            skip = testskip
            seleted_id_arg = list(range(0, max(seleted_id_arg)+1, 1))
        for idx,frame in enumerate(seleted_id_arg):
            fname = os.path.join(basedir, 'scene_image',"%05d" % frame + '.jpg')
            imgs.append(imageio.imread(fname))
            mask_img = imageio.imread(fname.replace('_image','_mask').replace('.jpg','.jpg'))
            masks.append(mask_img>0)
            if s=='train':
                # pose_blender = camera.pose(R=torch.diag(torch.tensor([1, 1, 1])))
                # pose_opencv = pose_blender.detach().cpu().numpy()
                # poses.append(pose_opencv)
                pose_opencv = pose_w2c[frame]
                poses.append(pose_opencv)
            else:
                pose_opencv = pose_w2c[frame]
                poses.append(pose_opencv)

        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        masks = (np.array(masks)).astype(np.float32)


        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_masks.append(masks)
        all_poses.append(poses)

    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]

    imgs = np.concatenate(all_imgs, 0)
    masks = np.concatenate(all_masks, 0)
    poses = np.concatenate(all_poses, 0)

    H, W = imgs[0].shape[:2]

    intrinsics = np.loadtxt(os.path.join(basedir,'intrinsics.txt'))
    focal = intrinsics[0][0]
    align_pose =  np.loadtxt(os.path.join(basedir,'cube_align_pose.txt'))

    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)

    imgs_matching = np.zeros_like(imgs)
    imgs_matching[masks > 0] = imgs[masks > 0, :]
    i_train, i_val, i_test = i_split
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

                    mask_img = cv2.resize(imgs_matching[j][..., -1], dsize=None, fx=resize_scale, fy=resize_scale)
                    mask_img = torch.from_numpy(mask_img).to(matching_config.device)
                    matcher_info = matching_pair(matching_model, matching_config, image0, image1, mask_img,
                                                 factor, resize_scale, i, j, max_matcher, type)
                    matcher_infos.append(matcher_info)
                matcher_infos_list.append(matcher_infos)
            return matcher_infos_list

    matcher_infos_object = matching_batch(imgs_matching, matching_outdoor)
    matcher_infos_scene = matching_batch(imgs, matching_indoor, type='scene')
    matcher_infos_list = matcher_infos_object + matcher_infos_scene
    if factor > 1:
        H = H // factor
        W = W // factor
        focal = focal / factor
        imgs_half_res = np.zeros((imgs.shape[0], H, W, 3)).astype(np.float32)
        masks_half_res = np.zeros((masks.shape[0], H, W)).astype(np.float32)

        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        for i, mask in enumerate(masks):
            masks_half_res[i] = cv2.resize(mask, (W, H), interpolation=cv2.INTER_AREA)

        imgs = imgs_half_res
        masks = masks_half_res


    del matching_indoor, matching_outdoor
    torch.cuda.empty_cache()

    return imgs,masks[...,None], None, poses, render_poses, \
        [H, W, focal], i_split, matcher_infos_list, sg_config, images_object,align_pose


def get_R_T_from_matchers(matcher_infos, hwf):
    #import korniamatcher_result

    H, W, focal = hwf
    K = torch.tensor([
        [focal, 0, 0.5 * W],
        [0, focal, 0.5 * H],
        [0, 0, 1]
    ]).float()
    matcher_infos = matcher_infos[1:5,...]
    points1 = matcher_infos[:, :, 0:2]
    points2 = matcher_infos[:, :, 2:4]
    weights = matcher_infos[:, :, -1]
    Ks = K[None,...].repeat(matcher_infos.shape[0], 1,1)
    Ks1, Ks2 = Ks,Ks
    F_mat = kornia.geometry.epipolar.find_fundamental(points1, points2, weights)
    E_mat = kornia.geometry.epipolar.essential_from_fundamental(F_mat, Ks1, Ks2)
    R_mat, T_mat, points3d = kornia.geometry.epipolar.motion_from_essential_choose_solution\
        (E_mat, Ks, Ks, points1, points2, mask=weights>0.5)
    import open3d
    for i in range(1,matcher_infos.shape[0]):
        point_cloud1 = open3d.geometry.PointCloud()
        point_cloud1.points = open3d.utility.Vector3dVector(points3d[i].detach().cpu().numpy())
        open3d.visualization.draw_geometries([point_cloud1])

    return R_mat, T_mat, points3d

