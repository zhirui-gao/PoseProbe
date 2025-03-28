import os, sys, copy, time, random, argparse,cv2
from eval import prealign_w2c_small_camera_systems
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import einops
import yaml
from torchvision.utils import save_image
from lib import utils_vis
from lib import common
from shutil import copyfile
from lib.losses import compute_diff_loss
from tqdm import tqdm, trange
from mmengine import config
import imageio
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import trimesh
import logging
import torch
import torch.nn.functional as F
from datetime import datetime
from lib import utils, dtu_eval
from torch.utils.tensorboard import SummaryWriter
from lib.load_data import load_data
from lib.utils import  get_root_logger
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
from lib import camera, vgg_loss
from easydict import EasyDict as edict
from lib.mipnerf_.utils.vis import visualize_depth
from lib.losses import object_losses
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

@torch.no_grad()
def cal_leading_eigenvector(M, method='power'):
    """
    Calculate the leading eigenvector using power iteration algorithm or torch.symeig
    Input:
        - M:      [bs, num_corr, num_corr] the compatibility matrix
        - method: select different method for calculating the learding eigenvector.
    Output:
        - solution: [bs, num_corr] leading eigenvector
    """
    if method == 'power':
        # power iteration algorithm
        leading_eig = torch.ones_like(M[:, :, 0:1])
        leading_eig_last = leading_eig
        for i in range(10):
            leading_eig = torch.bmm(M, leading_eig)
            leading_eig = leading_eig / (torch.norm(leading_eig, dim=1, keepdim=True) + 1e-6)
            if torch.allclose(leading_eig, leading_eig_last):
                break
            leading_eig_last = leading_eig
        leading_eig = leading_eig.squeeze(-1)
        return leading_eig
    elif method == 'eig':  # cause NaN during back-prop
        e, v = torch.symeig(M, eigenvectors=True)
        leading_eig = v[:, :, -1]
        return leading_eig
    else:
        exit(-1)

def config_parser():
    '''Define command line arguments
    '''

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', required=True,
                        help='config file path')
    parser.add_argument("--seed", type=int, default=777,
                        help='Random seed')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--no_reload_optimizer", action='store_true',
                        help='do not reload optimizer state from saved ckpt')
    parser.add_argument("--ft_path", type=str, default='',
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--export_bbox_and_cams_only", type=str, default='',
                        help='export scene bbox and camera poses for debugging and 3d visualization')
    parser.add_argument("--export_coarse_only", type=str, default='')
    parser.add_argument("--export_fine_only", type=str, default='')
    parser.add_argument("--mesh_from_sdf", action='store_true')

    # testing options
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true')
    parser.add_argument("--render_train", action='store_true')
    parser.add_argument("--render_video", action='store_true')
    parser.add_argument("--render_video_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    parser.add_argument("--eval_ssim", default=True)
    parser.add_argument("--eval_lpips_alex", default=True)
    parser.add_argument("--eval_lpips_vgg", default=True)

    # logging/saving options
    parser.add_argument("--i_print", type=int, default=100,
                        help='frequency of console printout and metric loggin')

    parser.add_argument("--i_validate_mesh", type=int, default=2000,
                        help='step of validating mesh')

    parser.add_argument("--i_validate", type=int, default=2000)
    parser.add_argument("--i_weights", type=int, default=10000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("-s", "--suffix", type=str, default="",
                        help='suffix for exp name')
    parser.add_argument("-p", "--prefix", type=str, default="",
                        help='prefix for exp name')
    parser.add_argument("--load_density_only", type=int, default=1)
    parser.add_argument("--load_expname", type=str, default="")
    parser.add_argument("--inst_seg_tag", type=int, default=3)
    parser.add_argument("--scene", type=str, default=0)
    parser.add_argument("--smooth_depth", type=float, default=0.1)
    parser.add_argument("--no_dvgo_init", action='store_true')
    parser.add_argument("--run_dvgo_init", action='store_true')
    parser.add_argument("--interpolate", default='0_1')
    parser.add_argument("--extract_color", action='store_true')
    parser.add_argument("--barf_opt", type=str, default='./lib/barf_model/barf_acc.yaml')
    parser.add_argument("--out_dir", help="Output directory.", type=str, default='./out')
    parser.add_argument("--dataset_name", help="Single or multi data.", type=str, default="blender")
    parser.add_argument("--mip_config", help="Path to config file.", required=False, default='./lib/mipnerf/configs/')
    parser.add_argument("--opts", nargs=argparse.REMAINDER,default=None,
                        help="Modify hparams. Example: train.py resume out_dir TRAIN.BATCH_SIZE 2")

    return parser

def render_viewpoints(model, render_poses, HW, Ks, ndc, render_kwargs,
                      gt_imgs=None, masks=None, savedir=None, render_factor=0, idx=None,
                      eval_ssim=True, eval_lpips_alex=True, eval_lpips_vgg=True,
                      use_bar=True, step=0, rgb_only=False):
    '''Render images for the given viewpoints; run evaluation if gt given.
    '''
    if render_poses.shape[1]==4:
        render_poses = render_poses[:, :3,:]
    assert len(render_poses) == len(HW) and len(HW) == len(Ks)
    render_poses = camera.pose.invert(render_poses) # gzr
    if render_factor!=0:
        HW = np.copy(HW)
        Ks = np.copy(Ks)
        HW //= render_factor
        Ks[:, :2, :3] //= render_factor

    rgbs = []
    normals = []
    ins = []
    outs = []
    disps = []
    psnrs = []
    fore_psnrs = []
    bg_psnrs = []
    ssims = []
    lpips_alex = []
    lpips_vgg = []
    render_normal = True
    split_bg = getattr(model, "bg_density", False)
    for i, c2w in enumerate((render_poses)):
        H, W = HW[i]
        K = Ks[i]
        rays_o, rays_d, viewdirs = Model.get_rays_of_a_view(
            H, W, K, c2w, ndc, inverse_y=render_kwargs['inverse_y'],
            flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        keys = ['rgb_marched', 'disp', 'alphainv_cum']
        if render_normal:
            keys.append('normal_marched')
        if split_bg:
            keys.extend(['in_marched', 'out_marched'])
        rays_o = rays_o.flatten(0, -2)
        rays_d = rays_d.flatten(0, -2)
        viewdirs = viewdirs.flatten(0, -2)
        render_result_chunks = [
            {k: v for k, v in model.inference(ro, rd, vd,training=False, **render_kwargs).items() if k in keys}
            for ro, rd, vd in zip(rays_o.split(4096, 0), rays_d.split(4096, 0), viewdirs.split(4096, 0))
        ]
        render_result = {
            k: torch.cat([ret[k] for ret in render_result_chunks]).reshape(H,W,-1)
            for k in render_result_chunks[0].keys()
        }
        rgb = render_result['rgb_marched'].cpu().numpy()
        rgbs.append(rgb)
        if rgb_only and savedir is not None:
            imageio.imwrite(os.path.join(savedir, '{:03d}.png'.format(i)), utils.to8b(rgb))
            continue

        disp = render_result['disp'].cpu().numpy()
        disps.append(disp)

        if render_normal:
            normal = render_result['normal_marched'].cpu().numpy()
            normals.append(normal)

        if split_bg:
            inside = render_result['in_marched'].cpu().numpy()
            ins.append(inside)
            outside = render_result['out_marched'].cpu().numpy()
            outs.append(outside)

        if masks is not None:
            if isinstance(masks[i], torch.Tensor):
                mask = masks[i].cpu().numpy() #.reshape(H, W, 1)
            else:
                mask = masks[i] #.reshape(H, W, 1)
            if mask.ndim == 2:
                mask = mask.reshape(H, W, 1)
            bg_rgb = rgb * (1 - mask)
            bg_gt = gt_imgs[i] * (1 - mask)
        else:
            mask, bg_rgb, bg_gt = np.ones(rgb.shape[:2]), np.ones(rgb.shape), np.ones(rgb.shape)

        if i==0:
            logger.info('Testing {} {}'.format(rgb.shape, disp.shape))
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb - gt_imgs[i])))
            back_p, fore_p = 0., 0.
            if  masks is not None:
                back_p = -10. * np.log10(np.sum(np.square(bg_rgb - bg_gt))/np.sum(1-mask))
                fore_p = -10. * np.log10(np.sum(np.square(rgb*mask - gt_imgs[i]*mask))/np.sum(mask))
            error = 1 - np.exp(-20 * np.square(rgb - gt_imgs[i]).sum(-1))[...,None].repeat(3,-1)
            logging.info("{} | full-image psnr {:.2f} | foreground psnr {:.2f} | background psnr: {:.2f} ".format(i, p, fore_p, back_p))
            psnrs.append(p)
            fore_psnrs.append(fore_p)
            bg_psnrs.append(back_p)
            if eval_ssim:
                ssims.append(utils.rgb_ssim(rgb, gt_imgs[i], max_val=1))
            if eval_lpips_alex:
                lpips_alex.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='alex', device='cpu'))
            if eval_lpips_vgg:
                lpips_vgg.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='vgg', device='cpu'))
        if savedir is not None:
            rgb8 = utils.to8b(rgbs[-1])
            id = idx if idx is not None else i
            step_pre = str(step) + '_' if step > 0 else ''
            filename = os.path.join(savedir, step_pre+'{:03d}.png'.format(id))
            rendername = os.path.join(savedir, step_pre + 'render_{:03d}.png'.format(id))
            gtname = os.path.join(savedir, step_pre + 'gt_{:03d}.png'.format(id))

            img8 = rgb8
            if gt_imgs is not None:
                error8 = utils.to8b(error)
                gt8 = utils.to8b(gt_imgs[i])
                imageio.imwrite(gtname, gt8)
                img8 = np.concatenate([error8, rgb8, gt8], axis=0)

            if split_bg and gt_imgs is not None:
                in8 = utils.to8b(ins[-1])
                out8 = utils.to8b(outs[-1])
                img8_2 = np.concatenate([in8, out8], axis=1)
                img8 = np.concatenate([rgb8, gt8], axis=1)
                img8 = np.concatenate([img8, img8_2], axis=0)

            imageio.imwrite(rendername, rgb8)
            imageio.imwrite(filename, img8)

            if render_normal:
                rot = c2w[:3, :3].permute(1, 0).cpu().numpy()
                normal = (rot @ normals[-1][..., None])[...,0]
                normal = 0.5 - 0.5 * normal
                if masks is not None:
                    normal = normal * mask.mean(-1)[...,None] + (1 - mask)
                normal8 = utils.to8b(normal)
                step_pre = str(step) + '_' if step > 0 else ''
                filename = os.path.join(savedir, step_pre+'{:03d}_normal.png'.format(id))
                imageio.imwrite(filename, normal8)

    rgbs = np.array(rgbs)
    disps = np.array(disps)
    if len(psnrs):
        logger.info('Testing psnr {:.2f} (avg) | foreground {:.2f} | background {:.2f}'.format(
            np.mean(psnrs), np.mean(fore_psnrs), np.mean(bg_psnrs)))
        if eval_ssim: logger.info('Testing ssim {} (avg)'.format(np.mean(ssims)))
        if eval_lpips_vgg: logger.info('Testing lpips (vgg) {} (avg)'.format(np.mean(lpips_vgg)))
        if eval_lpips_alex: logger.info('Testing lpips (alex) {} (avg)'.format(np.mean(lpips_alex)))

    return rgbs, disps

@torch.no_grad()
def visualize_val_image(model,model_bg, images_gt, render_poses,intr,HW, global_step,
                        opt,ndc,render_kwargs,id=None, render_only=False):
    if id is None:
        rand_idx = random.randint(0, len(render_poses) - 1)
    else:
        rand_idx = id

    render_poses = render_poses[rand_idx][None]
    intr = intr[rand_idx][None]
    HW = HW[rand_idx][None]
    image_gt = images_gt[rand_idx][None].cpu()
    if render_only is False:
        if model is None:
            rgbs, disps = torch.zeros_like(images_gt[0][None]).cpu(), torch.zeros_like(image_gt[0][None]).cpu()
        else:
            rgbs, disps = render_viewpoints(model, render_poses, HW, intr, ndc, render_kwargs)
            rgbs = torch.tensor(rgbs).cpu()
            disps = torch.tensor(disps).cpu()
            disps = disps.repeat(1, 1, 1, 3).cpu()
    if model_bg is None or opt is None:
        depth_bg,fine_rgb,coarse_rgb = torch.zeros_like(image_gt[0]),torch.zeros_like(image_gt[0]),torch.zeros_like(image_gt[0])
    else:
        rays, _ = model_bg.forward_graph(opt, render_poses, intr=intr, mode='val',global_step=global_step)
        stack = model_bg.validation_step(batch=[rays, image_gt],global_step=global_step) # (4, 3, H, W)
        coarse_rgb = einops.rearrange(stack[1,...], 'b h w -> h w b')
        fine_rgb = einops.rearrange(stack[2, ...], 'b h w -> h w b')

        depth_bg = visualize_depth(stack[3,0,...])
        depth_bg = torch.tensor(einops.rearrange(depth_bg, 'b h w -> h w b'),device=fine_rgb.device)
    if render_only:
        return fine_rgb, depth_bg
    else:
        stack_image = torch.cat([image_gt[0], coarse_rgb, fine_rgb, depth_bg, rgbs[0], disps[0]],dim=1)  # ( H, W,3)
        loss_mse_render = F.mse_loss(image_gt[0], fine_rgb)
        psnr = utils.mse2psnr(loss_mse_render.detach()).item()
        return stack_image, psnr


@torch.no_grad()
def visualize_bg_image(model_bg, images_gt, render_poses,intr, global_step,opt,id=None):
    if id is None:
        rand_idx = random.randint(0, len(render_poses) - 1)
    else:
        rand_idx = id
    render_poses = render_poses[rand_idx][None]
    intr = intr[rand_idx][None]
    image_gt = images_gt[rand_idx][None].cpu()

    ret, _ = model_bg.forward_graph(opt, render_poses, intr=intr, mode='val', global_step=global_step)
    rgb_map_bg, depth_map_bg, opacity = model_bg.visualize(opt, ret, split="val")
    stack_image = torch.cat([image_gt[0], rgb_map_bg, depth_map_bg, opacity], dim=1)  # ( H, W,3)
    loss_mse_render = F.mse_loss(image_gt[0], rgb_map_bg)
    psnr = utils.mse2psnr(loss_mse_render.detach()).item()
    return stack_image, psnr


@torch.no_grad()
def visualize_object_image(model,images_gt, render_poses,intr,HW, global_step,
                        opt,ndc,render_kwargs,id=None):
    if id is None:
        rand_idx = random.randint(0, len(render_poses) - 1)
    else:
        rand_idx = id
    render_poses = render_poses[rand_idx][None]
    intr = intr[rand_idx][None]
    HW = HW[rand_idx][None]
    image_gt = images_gt[rand_idx][None].cpu()
    rgbs, disps = render_viewpoints(model, render_poses, HW, intr, ndc, render_kwargs)
    rgbs = torch.tensor(rgbs).cpu()
    disps = torch.tensor(disps).cpu()
    disps = disps.repeat(1, 1, 1, 3).cpu()
    stack_image = torch.cat([image_gt[0], rgbs[0], disps[0]],dim=1)  # ( H, W,3)
    loss_mse_render = F.mse_loss(image_gt[0], rgbs)
    psnr = utils.mse2psnr(loss_mse_render.detach()).item()
    return stack_image, psnr

@torch.no_grad()
def visualize_test_image(model,model_bg, images_gt, render_poses,intr,HW, global_step,
                        opt,ndc,render_kwargs,id=None):
    logger.info('image id {}'.format(id))
    if id is None:
        rand_idx = random.randint(0, len(render_poses) - 1)
    else:
        rand_idx = id
    render_poses = render_poses[rand_idx][None]
    intr = intr[rand_idx][None]
    HW = HW[rand_idx][None]
    image_gt = images_gt[rand_idx][None].cpu()

    if model is None:
        rgbs, disps = torch.zeros_like(images_gt[0][None]).cpu(), torch.zeros_like(image_gt[0][None]).cpu()
    else:
        rgbs, disps = render_viewpoints(model, render_poses, HW,intr, ndc,render_kwargs)
        rgbs = torch.tensor(rgbs).cpu()

    if model_bg is None or opt is None:
        depth_bg,fine_rgb,coarse_rgb = torch.zeros_like(image_gt[0]),torch.zeros_like(image_gt[0]),torch.zeros_like(image_gt[0])
    else:
        ret, _ = model_bg.forward_graph(opt, render_poses, intr=intr, mode='val',global_step=global_step)
        fine_rgb, depth_bg, opacity = model_bg.visualize(opt, ret, split="val")# (4, 3, H, W)


    loss_mse_render = F.mse_loss(image_gt[0], fine_rgb)
    psnr = utils.mse2psnr(loss_mse_render.detach()).item()
    ssims = utils.rgb_ssim(fine_rgb, image_gt[0], max_val=1)
    lpips_alex = utils.rgb_lpips(fine_rgb.cpu().numpy(), image_gt[0].cpu().numpy(), net_name='alex', device='cpu')
    lpips_vgg =(utils.rgb_lpips(fine_rgb.cpu().numpy(), image_gt[0].cpu().numpy(), net_name='vgg', device='cpu'))

    logger.info('Testing psnr {} ssim {} lpips (alex) {} lpips (vgg) {}'.format(psnr,ssims,lpips_alex, lpips_vgg))
    result = torch.tensor([psnr,ssims,lpips_alex, lpips_vgg])
    return image_gt[0], fine_rgb, depth_bg, rgbs[0], result


def gen_poses_between(pose_0, pose_1, ratio):
    if pose_0.shape[0]==3:
        pose_0 = np.concatenate((pose_0,np.array([[0,0,0,1]])),axis=0)
        pose_1 = np.concatenate((pose_1,np.array([[0,0,0,1]])),axis=0)
    pose_0 = np.linalg.inv(pose_0)
    pose_1 = np.linalg.inv(pose_1)
    rot_0 = pose_0[:3, :3]
    rot_1 = pose_1[:3, :3]
    rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
    key_times = [0, 1]
    slerp = Slerp(key_times, rots)
    rot = slerp(ratio)
    pose = np.diag([1.0, 1.0, 1.0, 1.0])
    pose = pose.astype(np.float32)
    pose[:3, :3] = rot.as_matrix()
    pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
    pose = np.linalg.inv(pose)
    return pose

@torch.no_grad()
def interpolate_view(model_bg, savedir, img_idx_0, img_idx_1, render_poses, HW, Ks, **render_kwargs):
    render_poses = render_poses.cpu().numpy()
    pose_0, pose_1 = render_poses[img_idx_0], render_poses[img_idx_1]
    images = []
    n_frames = 60
    image_dir = os.path.join(savedir, 'images_full')
    os.makedirs(image_dir, exist_ok=True)
    poses = []
    for i in range(n_frames):
        new_pose = gen_poses_between(pose_0, pose_1, np.sin(((i / n_frames) - 0.5) * np.pi) * 0.5 + 0.5)
        poses.append(new_pose)

    render_kwargs.update(dict(
        savedir=image_dir,
        eval_ssim=False, eval_lpips_alex=False, eval_lpips_vgg=False,
        rgb_only=True,
    ))
    HW = HW[:1].repeat(len(poses),0)
    Ks = Ks[:1].repeat(len(poses),1,1)
    rgbs = []
    render_poses = torch.from_numpy(np.asarray(poses)[:,:3,:]).cuda()
    for i in tqdm(range(n_frames)):
        fine_rgb, depth_bg = visualize_val_image(model, model_bg, torch.ones(n_frames,HW[0][0],HW[0][1],3), render_poses,
                                                Ks, HW,latest_step, opt_bg, cfg.data.ndc,
                                                render_viewpoints_kwargs['render_kwargs'], id=i,render_only=True)
        rgbs.append(fine_rgb.numpy())
        stack_image = torch.cat([fine_rgb,depth_bg],dim=1)
        filename = os.path.join(image_dir, '{:03d}.png'.format(i))
        imageio.imwrite(filename, np.uint8(stack_image*255))

    for i in range(n_frames):
        images.append(rgbs[i])
    for i in range(n_frames):
        images.append(rgbs[n_frames - i - 1])
    h, w, _ = images[0].shape
    imageio.mimwrite(os.path.join(savedir, 'render_{}_{}.mp4'.format(img_idx_0, img_idx_1)),
                     utils.to8b(images), fps=30, quality=8)


def seed_everything():
    '''Seed everything for better reproducibility.
    (some pytorch operation is non-deterministic like the backprop of grid_samples)
    '''
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


def load_everything(args, cfg, device):
    '''Load images / poses / camera settings / data split.
    '''
    vgg_model = vgg_loss.VGGLoss()
    mode = getattr(cfg.data, 'mode', dict())
    if 'train_all' in cfg:
        mode.update(train_all=cfg.train_all)
        print(" * * * Train with all the images: {} * * * ".format(cfg.train_all))
    if 'reso_level' in cfg:
        mode.update(reso_level=cfg.reso_level)
    data_dict = load_data(cfg.data, **mode, white_bg=cfg.data.white_bkgd)

    # remove useless field
    kept_keys = {
        'hwf', 'HW', 'Ks', 'near', 'far',
        'i_train', 'i_val', 'i_test', 'irregular_shape', 'align_pose','depths',
         'poses','render_poses', 'images','images_gray', 'scale_mats_np', 'masks','matcher_infos'}  #
    for k in list(data_dict.keys()):
        if k not in kept_keys:
            data_dict.pop(k)
    vgg_model.to(device)
    vgg_features = vgg_model.get_multi_features(torch.tensor(data_dict['images'][data_dict['i_train']]))
    data_dict['vgg_features'] = vgg_features
    # construct data tensor
    if data_dict['irregular_shape']:
        data_dict['images'] = [torch.FloatTensor(im, device='cpu').cuda() for im in data_dict['images']]
        data_dict['masks'] = [torch.FloatTensor(im, device='cpu').cuda() for im in data_dict['masks']]
    else:
        data_dict['images'] = torch.FloatTensor(data_dict['images'], device='cpu').cuda()
        data_dict['masks'] = torch.FloatTensor(data_dict['masks'], device='cpu').cuda()
    data_dict['poses'] = torch.Tensor(data_dict['poses'])
    data_dict['align_pose'] = torch.Tensor(data_dict['align_pose'])
    data_dict['Ks'] = torch.Tensor(data_dict['Ks'])
    transform = transforms.GaussianBlur(kernel_size=15)
    semantics = transform(data_dict['masks'].permute(0,3,1,2)).permute(0,2,3,1)# # [n_images, H, W, 1]

    semantics[data_dict['masks'] > 1.0 - 1e-5] = 2
    semantics[(semantics>0)*(semantics<=1.0-1e-5)] = 1

    background_samplers = []
    boundary_samplers = []
    object_samplers = []
    for idx in range(len(data_dict['i_train'])):
        background_samplers.append([(data_dict['masks'][idx] == 0).nonzero(as_tuple=True)[0].type(torch.int16),
                                    (data_dict['masks'][idx] == 0).nonzero(as_tuple=True)[1].type(torch.int16)])
        boundary_samplers.append([(semantics[idx]==1).nonzero(as_tuple=True)[0].type(torch.int16),
                              (semantics[idx]==1).nonzero(as_tuple=True)[1].type(torch.int16)])
        object_samplers.append([(semantics[idx]==2).nonzero(as_tuple=True)[0].type(torch.int16),
                                (semantics[idx]==2).nonzero(as_tuple=True)[1].type(torch.int16)])
        # import ipdb; ipdb.set_trace()
    data_dict['samplers']={}
    data_dict['samplers']['background'] = background_samplers
    data_dict['samplers']['boundary'] = boundary_samplers
    data_dict['samplers']['object'] = object_samplers
    del vgg_model
    torch.cuda.empty_cache()
    return data_dict


def compute_bbox_by_cam_frustrm(args, cfg, HW, Ks, poses, i_train, near, far, **kwargs):
    logger.info('compute_bbox_by_cam_frustrm: start')
    xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
    xyz_max = -xyz_min
    poses = camera.pose.invert(poses)
    for (H, W), K, c2w in zip(HW[i_train], Ks[i_train], poses[i_train]):
        rays_o, rays_d, viewdirs = Model.get_rays_of_a_view(
            H=H, W=W, K=K, c2w=c2w,
            ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
            flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        pts_nf = torch.stack([rays_o+viewdirs*near, rays_o+viewdirs*far])
        xyz_min = torch.minimum(xyz_min, pts_nf.amin((0,1,2)))
        xyz_max = torch.maximum(xyz_max, pts_nf.amax((0,1,2)))
    logger.info('compute_bbox_by_cam_frustrm: xyz_min {}'.format(xyz_min))
    logger.info('compute_bbox_by_cam_frustrm: xyz_max {}'.format(xyz_max))
    logger.info('compute_bbox_by_cam_frustrm: finish')
    return xyz_min, xyz_max

@torch.no_grad()
def compute_bbox_by_coarse_geo(model_class, model_path, thres):
    logger.info('compute_bbox_by_coarse_geo: start')
    eps_time = time.time()
    model,_ = utils.load_model(model_class, model_path, strict=False)
    interp = torch.stack(torch.meshgrid(
        torch.linspace(0, 1, model.density.shape[2]),
        torch.linspace(0, 1, model.density.shape[3]),
        torch.linspace(0, 1, model.density.shape[4]),
    ), -1)
    dense_xyz = model.xyz_min * (1-interp) + model.xyz_max * interp
    density = model.grid_sampler(dense_xyz, model.density)
    alpha = model.activate_density(density)
    mask = (alpha > thres)
    active_xyz = dense_xyz[mask]
    xyz_min = active_xyz.amin(0)
    xyz_max = active_xyz.amax(0)
    logger.info('compute_bbox_by_coarse_geo: xyz_min {}'.format(xyz_min))
    logger.info('compute_bbox_by_coarse_geo: xyz_max {}'.format(xyz_max))
    eps_time = time.time() - eps_time
    logger.info('compute_bbox_by_coarse_geo: finish (eps time: {} secs)'.format(eps_time))
    return xyz_min, xyz_max


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
def log_scalars(loss,loss_weight,metric=None,step=0,split="train",writer=None):
    for key, value in loss.items():
        if key=="all": continue
        if loss_weight[key] is not None:
            writer.add_scalar("{0}/loss_{1}".format(split,key),value,step)
    if metric is not None:
        for key,value in metric.items():
            writer.add_scalar("{0}/{1}".format(split,key),value,step)


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

@torch.no_grad()
def prealign_cameras(pose,pose_GT,device, use_svd=False):
    # compute 3D similarity transform via Procrustes analysis
    center = torch.zeros(1,1,3,device=device)
    center_pred = camera.cam2world(center,pose)[:,0] # [N,3]
    center_GT = camera.cam2world(center,pose_GT)[:,0] # [N,3]
    try:
        if use_svd:
            sim3 = camera.procrustes_analysis(center_GT,center_pred)
        else:
            sim3 = edict(t0=0, t1=0, s0=1, s1=1, R=torch.eye(3, device=device))
    except:
        sim3 = edict(t0=0,t1=0,s0=1,s1=1,R=torch.eye(3,device=device))
    # align the camera poses
    center_aligned = (center_pred-sim3.t1)/sim3.s1@sim3.R.t()*sim3.s0+sim3.t0
    R_aligned = pose[...,:3]@sim3.R.t()
    t_aligned = (-R_aligned@center_aligned[...,None])[...,0]
    pose_aligned = camera.pose(R=R_aligned,t=t_aligned)
    return pose_aligned, sim3

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


def comp_closest_pts_idx_with_split(pts_src, pts_des):
    """
    :param pts_src:     (3, S)
    :param pts_des:     (3, D)
    :param num_split:
    :return:
    """
    pts_src_list = torch.split(pts_src, 500000, dim=1)
    idx_list = []
    for pts_src_sec in pts_src_list:
        diff = pts_src_sec[:, :, np.newaxis] - pts_des[:, np.newaxis, :]  # (3, S, 1) - (3, 1, D) -> (3, S, D)
        dist = torch.linalg.norm(diff, dim=0)  # (S, D)
        closest_idx = torch.argmin(dist, dim=1)  # (S,)
        idx_list.append(closest_idx)
    closest_idx = torch.cat(idx_list)
    return closest_idx

def comp_point_point_error(Xt, Yt):
    closest_idx = comp_closest_pts_idx_with_split(Xt, Yt)
    pt_pt_vec = Xt - Yt[:, closest_idx]  # (3, S) - (3, S) -> (3, S)
    pt_pt_dist = torch.linalg.norm(pt_pt_vec, dim=0)
    eng = torch.mean(pt_pt_dist)
    return eng



def get_overlap_region(points1, points2, tol=1e-5):
    closest_idx = comp_closest_pts_idx_with_split(points1.permute(1,0), points2.permute(1,0))
    diff = points1 - points2[closest_idx]
    dist = torch.linalg.norm(diff, dim=1)
    idx = dist < tol
    return points1[idx]


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



def validate_image(cfg, stage, step, data_dict, render_viewpoints_kwargs, eval_all=True):
    testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_test_{stage}')
    os.makedirs(testsavedir, exist_ok=True)
    rand_idx = random.randint(0, len(data_dict['poses'][data_dict['i_test']])-1)
    logger.info("validating test set idx: {}".format(rand_idx))
    eval_lpips_alex = args.eval_lpips_alex and eval_all
    eval_lpips_vgg = args.eval_lpips_alex and eval_all
    rgbs, disps = render_viewpoints(
        render_poses=data_dict['poses'][data_dict['i_test']][rand_idx][None],
        HW=data_dict['HW'][data_dict['i_test']][rand_idx][None],
        Ks=data_dict['Ks'][data_dict['i_test']][rand_idx][None],
        gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_test']][rand_idx][None],
        masks=[data_dict['masks'][i].cpu().numpy() for i in data_dict['i_test']][rand_idx][None],
        savedir=testsavedir,
        eval_ssim=args.eval_ssim, eval_lpips_alex=eval_lpips_alex, eval_lpips_vgg=eval_lpips_vgg, idx=rand_idx, step=step,
        **render_viewpoints_kwargs)


def validate_deform_mesh(model, resolution=128, threshold=0.0, prefix="", world_space=False,
                  scale_mats_np=None, gt_eval=False, runtime=True, scene=122, smooth=True,
                  extract_color=False):
    os.makedirs(os.path.join(cfg.basedir, cfg.expname, 'meshes'), exist_ok=True)
    bound_min = model.xyz_min.clone().detach().float()
    bound_max = model.xyz_max.clone().detach().float()

    gt_path = os.path.join(cfg.data.datadir, "stl_total.ply") if gt_eval else ''
    vertices0, triangles = model.extract_deform_geometry(bound_min, bound_max, resolution=resolution,
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
        vertex_colors = [model.mesh_color_forward(pts) for pts in ray_pts]
        vertex_colors = (torch.concat(vertex_colors).cpu().detach().numpy() * 255.).astype( np.uint8)
        mesh = trimesh.Trimesh(vertices, triangles, vertex_colors=vertex_colors)
    else:
        mesh = trimesh.Trimesh(vertices, triangles)
    mesh_path = os.path.join(cfg.basedir, cfg.expname, 'meshes', "{}_deform_".format(scene)+prefix+'.ply')
    mesh.export(mesh_path)
    logger.info("deform mesh saved at " + mesh_path)
    return 0

def validate_mesh(model, resolution=128, threshold=0.0, prefix="", world_space=False,
                  scale_mats_np=None, gt_eval=False, runtime=True, scene=122, smooth=True,
                  extract_color=False):
    os.makedirs(os.path.join(cfg.basedir, cfg.expname, 'meshes'), exist_ok=True)
    bound_min = model.xyz_min.clone().detach().float()
    bound_max = model.xyz_max.clone().detach().float()

    gt_path = os.path.join(cfg.data.datadir, "stl_total.ply") if gt_eval else ''
    vertices0, triangles = model.extract_geometry(bound_min, bound_max, resolution=resolution,
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
        vertex_colors = [model.mesh_color_forward(pts) for pts in ray_pts]
        vertex_colors = (torch.concat(vertex_colors).cpu().detach().numpy() * 255.).astype( np.uint8)
        mesh = trimesh.Trimesh(vertices, triangles, vertex_colors=vertex_colors)
    else:
        mesh = trimesh.Trimesh(vertices, triangles)
    mesh_path = os.path.join(cfg.basedir, cfg.expname, 'meshes', "{}_".format(scene)+prefix+'.ply')
    mesh.export(mesh_path)
    logger.info("mesh saved at " + mesh_path)
    if gt_eval:
        mean_d2s, mean_s2d, over_all = dtu_eval.eval(mesh_path, scene=scene, eval_dir=os.path.join(cfg.basedir, cfg.expname, 'meshes'),
                      dataset_dir='data/DTU', suffix=prefix+'eval', use_o3d=False, runtime=runtime)
        res = "standard point cloud sampling" if not runtime else "down sampled point cloud for fast eval (NOT standard!):"
        logger.info("mesh evaluation with {}".format(res))
        logger.info(" [ d2s: {:.3f} | s2d: {:.3f} | mean: {:.3f} ]".format(mean_d2s, mean_s2d, over_all))
        return over_all
    return 0.


class scene_rep_reconstruction(torch.nn.Module):
    def __init__(self, args, cfg, cfg_model, cfg_train, xyz_min, xyz_max, data_dict, stage):
        super(scene_rep_reconstruction, self).__init__()
        logger.info("= " * 10 + "Begin training state [ {} ]".format(stage) + " =" * 10)
        # init
        self.args = args
        self.cfg_train = cfg_train
        self.cfg_model = cfg_model
        self.cfg = cfg

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not hasattr(cfg_train, 'world_bound_scale') :
            cfg_train.world_bound_scale = 1.5

        if abs(cfg_train.world_bound_scale - 1) > 1e-9:
            xyz_shift = (xyz_max - xyz_min) * (cfg_train.world_bound_scale - 1) / 2
            xyz_min -= xyz_shift
            xyz_max += xyz_shift
        self.HW, self.Ks, self.near, self.far, self.i_train, self.i_val, self.i_test, \
            self.poses, self.render_poses, self.images, self.images_gray, self.masks, self.samplers, self.align_pose = [
            data_dict[k] for k in ['HW', 'Ks', 'near', 'far', 'i_train', 'i_val', 'i_test', 'poses', 'render_poses',
                                   'images', 'images_gray', 'masks', 'samplers', 'align_pose'
                                   ]
        ]

        sdf_grid_path = None
        sdf0 = None
        self.rect_size = ((xyz_max - xyz_min) / (cfg_train.world_bound_scale* 1.05)).tolist()
        self.range_shape = (xyz_max - xyz_min) / (cfg_train.world_bound_scale * 1.05)
        if cfg.surf_model_and_render.load_sdf:
            sdf_grid_path = os.path.join(cfg.data.datadir, 'sdf_grid.npy')
            if os.path.exists(sdf_grid_path):
                sdf_dict = np.load(sdf_grid_path, allow_pickle=True).tolist()
                range_shape = (sdf_dict['xyz_max'] - sdf_dict['xyz_min']) * cfg_train.world_bound_scale / 2
                xyz_min = -torch.tensor(range_shape).to(self.device).to(torch.float32)
                xyz_max = torch.tensor(range_shape).to(self.device).to(torch.float32)
                self.rect_size = ((xyz_max - xyz_min) / (cfg_train.world_bound_scale * 1.05)).tolist()
                self.range_shape = (xyz_max - xyz_min) / (cfg_train.world_bound_scale * 1.05)
                xyz_min[:] = xyz_min.min()
                xyz_max[:] = xyz_max.max()
        self.last_ckpt_path = os.path.join(cfg.basedir, cfg.expname, f'{stage}_pose_last.tar')
        # init model
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
            i_train=self.i_train, i_test=self.i_test,
            N_iters=cfg_train.N_iters,
            HW=self.HW,
            range_shape = self.range_shape,
            rect_size = self.rect_size,
            **model_kwargs)
        self.model.maskout_near_cam_vox(self.poses[self.i_train, :3, 3], self.near)
        self.model = self.model.to(self.device)
        if sdf0 is not None:
            self.model.init_sdf_from_sdf(sdf0, smooth=False)

        self.model_pose = self.model_pose.to(self.device)

        self.poses_raw, self.pose_GT = get_all_training_poses(model=self.model_pose, poses=self.poses, device=device)
        # init optimizer
        self.optimizer = utils.create_optimizer_or_freeze_model(self.model, cfg_train, global_step=0)
        self.optim_pose, self.sched_pose = utils.create_optimizer_pose(self.model_pose, cfg_train, False)

        self.optim_pose_align, self.sched_pose_align = utils.create_optimizer_pose(self.model_pose, cfg_train, True)
        # init rendering setup
        self.render_kwargs = {
            'near': data_dict['near'],
            'far': data_dict['far'],
            'bg': 1 if cfg.data.white_bkgd else 0,
            'stepsize': cfg_model.stepsize,
            'inverse_y': cfg.data.inverse_y,
            'flip_x': cfg.data.flip_x,
            'flip_y': cfg.data.flip_y,
        }
        self.nl = 0.05  # nearest_limit
        self.matcher_result = data_dict['matcher_infos']  # N,512,5
        self.j_train = self.i_train - 1
        self.j_train[0] = 1

        coord0, coord1, mconf, i_index, j_index = [], [], [], [], []
        self.poses_pnp = []
        first_pose = self.pose_GT[0]
        self.poses_pnp.append(first_pose.detach())
        for i in self.i_train:
            if i==0:
                continue
            else:
                if getattr(self.cfg.pnp, 'use_identical', False):
                    camera_pose = self.poses_pnp[i - 1]
                else:
                    camera_pose = self.opencv_pnp_ransac(self.matcher_result[i], i, self.Ks,
                                                         self.poses_pnp[i - 1].unsqueeze(0), self.render_kwargs)
            self.poses_pnp.append(camera_pose)
        initial_pose = torch.stack(self.poses_pnp, dim=0)
        np.save(cfg.data.datadir + '/' + str(len(self.i_train)) + '_initial_pose_new.npy', initial_pose.detach().cpu())

        # evaluation
        initial_pose = get_current_pose(model=self.model_pose, poses_gt=self.poses)[self.i_train]
        pose_aligned, pose_ref = initial_pose.detach().cpu(), self.pose_GT[self.i_train].detach().cpu()
        pose_aligned, _ = prealign_w2c_small_camera_systems(pose_aligned, pose_ref)
        error = evaluate_camera_alignment(pose_aligned,pose_ref)
        print('initilized by PnP, the pose error is:',error.R.mean(),error.t.mean() )
        fig = plt.figure(figsize=(10, 10))
        utils_vis.plot_save_poses_blender(fig, pose_aligned, pose_ref=pose_ref, path=cfg.data.datadir,
                                          ep='-1')
        plt.close()

        for i in self.i_train:
            num_camera = min(1, len(self.i_train) - 1)
            j_train = list(range(i - 1, -1, -1)) + list(range(i + 1, self.i_train.shape[0], 1))
            mconf_h = torch.stack(self.matcher_result[i], dim=0)[:num_camera, :, -1]  # (b,n)
            coord0_h = torch.stack(self.matcher_result[i], dim=0)[:num_camera, :, 0:2]  # (b,n,2)
            coord1_h = torch.stack(self.matcher_result[i], dim=0)[:num_camera, :, 2:4]  # (b,n,2)
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
        coord0, coord1, mconf= [], [], []
        for i in self.i_train:
            mconf_h = torch.stack(self.matcher_result[i+len(self.i_train)], dim=0)[:1, :, -1]  # (b,n)
            coord0_h = torch.stack(self.matcher_result[i+len(self.i_train)], dim=0)[:1, :, 0:2]  # (b,n,2)
            coord1_h = torch.stack(self.matcher_result[i+len(self.i_train)], dim=0)[:1, :, 2:4]  # (b,n,2)
            coord0.append(coord0_h)
            coord1.append(coord1_h)
            mconf.append(mconf_h)
        self.mconf_scene = torch.concat(mconf, dim=0)
        self.coord0_scene = torch.concat(coord0, dim=0)
        self.coord1_scene = torch.concat(coord1, dim=0)
        if self.cfg.camera.incremental:
            incremental_i_train = [0,1]
        else:
            incremental_i_train = list(range(0, len(self.i_train), 1))
        self.selected_i_train = incremental_i_train

    def opencv_pnp_ransac(self, matcher_result_list, img_id, Ks, current_pose, render_kwargs):
        if isinstance(img_id, int) or isinstance(img_id, np.int64):
            img_id = [img_id]
        matcher_result = matcher_result_list[0]
        coord0_h = matcher_result[:, 0:2][None] #  others
        x2d = matcher_result[:, 2:4][None]
        weights = matcher_result[:, -1][None]
        rays_o_0, rays_d_0 = get_ray_dir(coord0_h, Ks[img_id], c2w=camera.pose.invert(current_pose),
                                         inverse_y=cfg.data.inverse_y,
                                         flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y, mode='no_center')
        rays_o_0 = einops.rearrange(rays_o_0, 'b n c ->(b n) c', c=3)
        rays_d_0 = einops.rearrange(rays_d_0, 'b n c ->(b n) c', c=3)
        query_points, mask_valid, sdf_ray_step = self.model.query_sdf_point_wocuda(rays_o_0, rays_d_0,
                                                                              global_step=None,
                                                                              keep_dim=True, **render_kwargs)
        world_points = einops.rearrange(query_points, '(b n) c ->b n c', b=len(img_id), c=3)
        mask_valid = einops.rearrange(mask_valid, '(b n)->b n', b=len(img_id))
        weights = weights * mask_valid
        world_points = world_points.detach().cpu().numpy().squeeze(0)
        img_points = x2d.detach().cpu().numpy().squeeze(0)
        weights = weights.detach().cpu().numpy().squeeze(0)
        Ks = Ks.detach().cpu().numpy()
        dist_coeffs = np.zeros((4, 1))
        mask = weights > 0
        world_points = world_points[mask]
        img_points = img_points[mask]
        _, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(world_points, img_points, Ks[0],
                                                                             dist_coeffs)
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
                                                                                                keep_dim=True,
                                                                                                **render_kwargs)
        else:
            query_points, mask_valid, _ = self.model.query_sdf_point_wocuda_wodeform(rays_o_p, rays_d_p,
                                                                              global_step=global_step,
                                                                              keep_dim=True, **render_kwargs)

        # distance to center:
        dis2center = self.point_to_ray_distance(rays_o_p, rays_d_p, point=self.model.xyz_min + self.model.xyz_max)
        near_surface_loss = (torch.clamp(dis2center - self.model.diagonal_length/3., min=0.0)* (mconf.flatten() > 0)).sum()


        query_points = einops.rearrange(query_points, '(b n) c ->b n c', b=len(index), c=3)
        mask_valid = einops.rearrange(mask_valid, '(b n)->b n', b=len(index))

        index = np.concatenate([j_train, i_train], axis=0)
        camera_pose_j = current_pose[index]
        pc_camera = camera.world2cam(query_points, camera_pose_j)
        if cfg.data.inverse_y:
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

    def get_project_feature_loss(self, global_step,current_pose, imsz,target_tr,rays_o_tr,rays_d_tr, i_list,j_list):
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
                                                                        keep_dim=True, **self.render_kwargs)
        query_points = einops.rearrange(query_points, '(b n) c ->b n c', b=len(i_list), c=3)
        mask_valid = einops.rearrange(mask_valid, '(b n)->b n', b=len(i_list))

        pc_camera_1 = camera.world2cam(query_points, camera_pose_1)
        if cfg.data.inverse_y:
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
                                                                                 keep_dim=True,
                                                                                 **self.render_kwargs)
        query_points_ref = einops.rearrange(query_points_ref, '(b n) c ->b n c', b=len(j_list), c=3)
        valid_point_ref = einops.rearrange(valid_point_ref, '(b n)->b n', b=len(j_list))
        valid_depth_ray = torch.linalg.norm(query_points - query_points_ref, dim=-1) < self.model.voxel_size * 2
        valid_depth_ray = valid_depth_ray * valid_point_ref * mask_valid

        pc_camera_0 = camera.world2cam(query_points, camera_pose_0)
        pc_camera_1 = camera.world2cam(query_points, camera_pose_1)
        if cfg.data.inverse_y:
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
        for vgg_features_layer in data_dict['vgg_features']:
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
        if data_dict['irregular_shape']:
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
                train_poses=c2w[i_train],
                HW=self.HW[i_train], Ks=self.Ks[i_train],
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

    def optimize_increamental(self,  end_step=100000,startobject=5000, opt=None):
        opt.H, opt.W = self.HW[0][0], self.HW[0][1]
        opt.device = self.device
        psnr_lst = []
        psnr_bg_lst = []
        weight_lst = []
        ep_list = []
        mask_lst = []
        bg_mask_lst = []
        weight_sum_lst = []
        weight_nonzero_lst = []
        s_val_lst = []
        time0 = time.time()
        patch_sample = True
        selected_i = 1
        global_step = 0
        skip_incr = 3000
        incremental_step = [0] + [skip_incr]*(len(self.i_train)-2)

        pbar = tqdm(total=end_step+ sum(incremental_step))
        while global_step <=end_step:
            pbar.update(1)
            train_idx = self.i_train[self.selected_i_train]
            if self.cfg.pnp.use_pnp:
                self.current_pose = get_current_pose_pnp(model=self.model_pose, pose_pnp=self.poses_pnp,
                                                         ids=train_idx)
            else:
                self.current_pose = get_current_pose(model=self.model_pose, poses_gt=self.poses)
            if self.cfg.camera.incremental and len(self.selected_i_train)<len(self.i_train) and \
                    global_step == incremental_step[selected_i]:
                selected_i += 1
                global_step = 0
                self.selected_i_train.append(selected_i)
                if getattr(self.cfg.pnp, 'use_identical', False):
                    self.poses_pnp[selected_i] = self.current_pose[selected_i - 1].detach()
                else:
                    self.poses_pnp[selected_i] = self.opencv_pnp_ransac(self.matcher_result[selected_i], selected_i, self.Ks,
                                                                        self.current_pose[selected_i - 1].detach().unsqueeze(0),
                                                                        self.render_kwargs)

            optimize_obejct_nerf = global_step <= self.cfg_train.N_iters and global_step>=startobject \
                                    and len(self.selected_i_train)==len(self.i_train)

            IsOptimizePose = global_step < self.cfg_train.N_iters_bg * 0.7
            loss_scalars, loss_weight = edict(), edict()

            self.optimizer.zero_grad(set_to_none=True)
            self.optim_pose.zero_grad()
            self.optimizer_bg_nerf.zero_grad()
            loss, loss_bg, psnr, psnr_bg = 0., 0, 0, 0
            if optimize_obejct_nerf:
                target_tr, mask_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = self.gather_training_rays \
                    (self.current_pose, train_idx)
                indices = torch.randperm(len(target_tr), device=target_tr.device)[:self.cfg_train.N_rand]
                target = target_tr[indices]
                mask = mask_tr[indices]
                rays_o = rays_o_tr[indices]
                rays_d = rays_d_tr[indices]
                viewdirs = viewdirs_tr[indices]
                render_result = self.model(rays_o, rays_d, viewdirs, global_step=global_step, **self.render_kwargs)
                loss_scalars, loss_weight, loss = object_losses(render_result, self.cfg_train, target, mask)
                psnr = utils.mse2psnr(loss_scalars.img_render.detach()).item()
                # surface-based percetual loss
                if self.cfg_train.weight_surface_projection > 0 and IsOptimizePose:
                    rand_id = np.random.randint(len(train_idx))
                    self_id = self.i_index[rand_id]
                    other_id = self.j_index[rand_id]
                    loss_surface_projection = self.get_project_feature_loss(global_step, self.current_pose,
                                                                            imsz[rand_id], target_tr,
                                                                            rays_o_tr, rays_d_tr, [self_id], [other_id])
                    loss_scalars.overlap_pc = loss_surface_projection
                    loss_weight.overlap_pc = self.cfg_train.weight_surface_projection
                    loss_pc = self.cfg_train.weight_surface_projection * loss_surface_projection
                    loss = loss + loss_pc

                if self.cfg_train.projection_dis_error > 0 and IsOptimizePose:
                    rand_id = np.random.randint(len(train_idx))
                    self_id = [self.i_index[rand_id]]
                    other_id = [self.j_index[rand_id]]
                    mconf_ = self.mconf[self_id]
                    coord_self = self.coord0[self_id]
                    coord_other = self.coord1[self_id]
                    projection_dis_error, near_surface_loss = self.get_project_error(global_step, self.current_pose,
                                                                                     coord_self, coord_other,
                                                                                     other_id, self_id, mconf_,
                                                                                     pixel_thre=100,
                                                                                     use_deform=True,
                                                                                     **self.render_kwargs)
                    loss_scalars.loss_near_surface = near_surface_loss
                    loss_weight.loss_near_surface = self.cfg_train.weight_near_surface
                    loss_near_surface = loss_weight.loss_near_surface * loss_scalars.loss_near_surface
                    loss_scalars.projection_dis_error = projection_dis_error
                    loss_weight.projection_dis_error = self.cfg_train.projection_dis_error
                    loss_pc =  loss_weight.projection_dis_error * loss_scalars.projection_dis_error
                    loss = loss + loss_pc + loss_near_surface

            else:
                if getattr(self.cfg_train, 'weight_probe_constrain', False) and len(self.selected_i_train)<len(self.i_train)\
                        and global_step< skip_incr//2:
                    rand_id = np.random.randint(len(train_idx))
                    self_id = [self.i_index[rand_id]]
                    other_id = [self.j_index[rand_id]]
                    mconf_ = self.mconf[self_id]
                    coord_self = self.coord0[self_id]
                    coord_other = self.coord1[self_id]
                    projection_dis_error, near_surface_loss = self.get_project_error(global_step, self.current_pose,
                                                                                     coord_self, coord_other,
                                                                                     other_id, self_id, mconf_,
                                                                                     use_deform=False,
                                                                                     **self.render_kwargs)
                    loss_scalars.loss_near_surface = near_surface_loss
                    loss_weight.loss_near_surface = self.cfg_train.weight_near_surface* self.cfg_train.weight_probe_constrain
                    loss_near_surface =  loss_weight.loss_near_surface * loss_scalars.loss_near_surface

                    loss_scalars.projection_dis_error = projection_dis_error
                    loss_weight.projection_dis_error = self.cfg_train.projection_dis_error * self.cfg_train.weight_probe_constrain
                    loss_pc = loss_weight.projection_dis_error  * loss_scalars.projection_dis_error
                    loss = loss + loss_pc + loss_near_surface





            ret_bg, ray_idx = self.model_bg.forward_graph(opt, self.current_pose[train_idx],
                                                          intr=self.Ks[train_idx],
                                                          global_step=global_step, mode='train',
                                                          patch_sample=patch_sample)
            target_bg = self.images[train_idx].view(len(train_idx), opt.H * opt.W, 3)
            target_bg = target_bg[:, ray_idx]
            loss_mse_render_bg = F.mse_loss(ret_bg['rgb'], target_bg)
            loss_scalars.mse_render_bg = loss_mse_render_bg
            loss_weight.mse_render_bg = self.cfg_train.weight_main
            loss_bg = loss_bg + self.cfg_train.weight_main * loss_mse_render_bg
            psnr_bg = utils.mse2psnr(loss_mse_render_bg.detach()).item()
            if patch_sample:
                depth = ret_bg['depth'].reshape(-1, 2, 2, 1)
                reg_depth = compute_tv_norm(depth).mean()
                loss_scalars.reg_depth_loss = reg_depth
                loss_weight.reg_depth_loss = 10 ** float(-2.0)
                loss_bg += reg_depth * loss_weight.reg_depth_loss

            if IsOptimizePose:
                self_id = [self.i_index[train_idx]]
                other_id = [self.j_index[train_idx]]
                mconf_ = self.mconf_scene[self_id]
                coord_self = self.coord1_scene[self_id]
                coord_other = self.coord0_scene[self_id]
                pose_w2c_self, pose_w2c_other = torch.eye(4).unsqueeze(0).repeat(coord_self.shape[0],1,1)\
                    .to(self.current_pose.device), \
                    torch.eye(4).unsqueeze(0).repeat(coord_self.shape[0],1,1).to(self.current_pose.device)
                pose_w2c_self[:, :3, :4] = self.current_pose[self_id]
                pose_w2c_other[:, :3, :4] = self.current_pose[other_id]
                loss_corres = self.model_bg.compute_loss_on_image_pair(opt, coord_self, coord_other, pose_w2c_self,
                                                                     pose_w2c_other, mconf_, self.Ks[self_id],
                                                                     global_step=global_step, mode='train')
                loss_scalars.scene_corr = loss_corres
                loss_weight.scene_corr = 10 ** float(-2.0)
                loss_bg += loss_corres * loss_weight.scene_corr
            loss = loss * 0.1 + loss_bg
            loss.backward()
            psnr_bg_lst.append(psnr_bg)
            psnr_lst.append(psnr)
            writer.add_scalar('train/lr_pose', self.optim_pose.state_dict()['param_groups'][0]['lr'], global_step)
            writer.add_scalar('train/psnr_bg', psnr_bg, global_step)
            if global_step % args.i_print == 1:
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
                    writer.add_scalar("{0}/error_R".format(split), error.R.mean(), global_step)
                    writer.add_scalar("{0}/error_t".format(split), error.t.mean(), global_step)
                    log_scalars(loss=loss_scalars, loss_weight=loss_weight, step=global_step, writer=writer)

                    fig = plt.figure(figsize=(10, 10))

                    output_path = os.path.join(cfg.basedir, cfg.expname)
                    cam_path = "{}/poses".format(output_path)
                    os.makedirs(cam_path, exist_ok=True)
                    utils_vis.plot_save_poses_blender(fig, pose_aligned, pose_ref, path=cam_path,
                                                      ep=str(len(train_idx))+'_'+str(global_step))
                    plt.close()
                    ep_list.append(global_step)

                eps_time = time.time() - time0
                eps_time_str = f'{eps_time // 3600:02.0f}:{eps_time // 60 % 60:02.0f}:{eps_time % 60:02.0f}'
                bg_mask_mean = 0. if len(bg_mask_lst) == 0 else np.mean(bg_mask_lst)
                logger.info(f'Optimize cameras: iter{global_step:3d} / '
                            f'(R/t): {error.R.mean():.3f} / {error.t.mean():.3f} / '
                            f'Loss: {loss:.9f} / PSNR: {np.mean(psnr_lst):5.2f} / PSNR_bg: {np.mean(psnr_bg_lst):5.2f} / '
                            f'Wmax: {np.mean(weight_lst):5.2f} / Wsum: {np.mean(weight_sum_lst):5.2f} / W>0: {np.mean(weight_nonzero_lst):5.2f}'
                            f' / s_val: {np.mean(s_val_lst):5.2g} / mask\%: {100 * np.mean(mask_lst):1.2f} / bg_mask\%: {100 * bg_mask_mean:1.2f} '
                            f'Eps: {eps_time_str}')
                psnr_lst, psnr_bg_lst, weight_lst, weight_sum_lst, weight_nonzero_lst, mask_lst, bg_mask_lst, s_val_lst \
                    = [], [], [], [], [], [], [], []
            if global_step % args.i_validate == 1:
                render_id = random.randint(0, len(train_idx) - 1)
                stack_image, psnr = visualize_bg_image(self.model_bg, self.images[train_idx],
                                                       self.current_pose[train_idx],
                                                       self.Ks[train_idx], global_step, opt, id=render_id)
                save_img_path = os.path.join(self.cfg.basedir, self.cfg.expname, 'training_imgs/bg_nerf')
                os.makedirs(save_img_path, exist_ok=True)
                img_name = 'step-' + str(global_step) + '_id-' + str(render_id) + '.png'
                save_image(stack_image.permute(2, 0, 1), os.path.join(save_img_path, img_name))

            if optimize_obejct_nerf:
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
                writer.add_scalar('train/psnr', psnr, global_step)
                writer.add_scalar('train/sdf_alpha', self.model.sdf_alpha, global_step)
                writer.add_scalar('train/sdf_beta', self.model.sdf_beta, global_step)
                writer.add_scalar('train/mask', mask_lst[-1], global_step)
                writer.add_scalar('train/s_val', s_val, global_step)
                if not getattr(self.cfg_train, 'cosine_lr', ''):
                    decay_steps = self.cfg_train.lrate_decay * 1000
                    decay_factor = 0.1 ** (1 / decay_steps)
                    for i_opt_g, param_group in enumerate(self.optimizer.param_groups):
                        param_group['lr'] = param_group['lr'] * decay_factor
                        writer.add_scalar('train/lr_' + param_group['name'], param_group['lr'], global_step)
                if global_step % args.i_validate_mesh == 0 or global_step== self.cfg_train.N_iters:
                    validate_deform_mesh(self.model, 128, threshold=0.0, prefix="{}final".format(global_step),
                                         world_space=True,
                                         scale_mats_np=data_dict['scale_mats_np'], gt_eval='dtu' in cfg.basedir,
                                         runtime=False,
                                         scene=args.scene)

                if global_step % args.i_validate == 1:
                    render_id = random.randint(0, len(train_idx) - 1)
                    stack_image, psnr = visualize_val_image(self.model, None, self.images[train_idx],
                                                            self.current_pose[train_idx],
                                                            self.Ks[train_idx], self.HW[train_idx],
                                                            global_step, None, self.cfg.data.ndc,
                                                            self.render_kwargs, id=render_id)
                    save_img_path = os.path.join(self.cfg.basedir, self.cfg.expname, 'training_imgs/object_nerf')
                    os.makedirs(save_img_path, exist_ok=True)
                    img_name = 'step-' + str(global_step) + '_id-' + str(render_id) + '.png'  # '_psnr-'+f'{psnr:.2f}'
                    save_image(stack_image.permute(2, 0, 1), os.path.join(save_img_path, img_name))

            self.optimizer_bg_nerf.step()
            self.sched_bg_nerf.step()
            if optimize_obejct_nerf:
                self.optimizer.step()
            # if IsOptimizePose:
            self.optim_pose.step()
            self.sched_pose.step()
            if global_step == end_step:
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

                last_ckpt_path = os.path.join(self.cfg.basedir, self.cfg.expname, 'last_ckpt_bg.tar')
                torch.save({
                    'global_step': global_step,
                    'current_pose': self.current_pose,
                    'model_state_dict': self.model_bg.state_dict(),
                    'optimizer_state_dict': self.optimizer_bg_nerf.state_dict(),
                    'sched_optimizer_state_dict': self.sched_bg_nerf.state_dict(),
                    'sched_pose_state_dict': self.sched_pose.state_dict(),
                    'optimizer_pose_state_dict': self.optim_pose.state_dict(),
                }, last_ckpt_path)
                logger.info(f'scene_rep_reconstruction ({global_step}): saved checkpoints at ' + last_ckpt_path)

            global_step +=1

    def get_bg_model_barf(self,  load_latest=False):
        with open(self.args.barf_opt) as file:
            opt = edict(yaml.safe_load(file))
        # scene branch
        self.model_bg = bg_nerf.BG_NeRF(N_iters=self.cfg_train.N_iters_bg, opt=opt)
        self.model_bg = self.model_bg.to(self.device)
        self.optimizer_bg_nerf, self.sched_bg_nerf = utils.setup_optimizer(self.model_bg, opt)
        checkpoint_path = os.path.join(cfg.basedir, cfg.expname, 'last_ckpt_bg.tar')
        latest_step = 0
        if load_latest and os.path.exists(checkpoint_path):
            checkpoint_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            # Load model dict
            self.model_bg.load_state_dict(checkpoint_dict['model_state_dict'], strict=True)
            self.optimizer_bg_nerf.load_state_dict(checkpoint_dict['optimizer_state_dict'])
            self.sched_bg_nerf.load_state_dict(checkpoint_dict['sched_pose_state_dict'])
            self.current_pose = checkpoint_dict['current_pose']
            latest_step = checkpoint_dict['global_step'] - 1
        return latest_step, opt

    def forward(self):
        latest_step, opt = self.get_bg_model_barf()
        self.optimize_increamental(end_step=self.cfg.surf_train.N_iters_bg, startobject=0, opt=opt)

def train(args, cfg, data_dict):
    # init
    logger.info('train: start')
    eps_time = time.time()
    with open(os.path.join(cfg.basedir, cfg.expname, 'args.txt'), 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    cfg.dump(os.path.join(cfg.basedir, cfg.expname, 'config.py'))

    xyz_min_fine, xyz_max_fine = torch.tensor(cfg.data.xyz_min).cuda(), torch.tensor(cfg.data.xyz_max).cuda()

    if hasattr(cfg, 'surf_train'):
        eps_surf = time.time()
        recon = scene_rep_reconstruction(
            args=args, cfg=cfg,
            cfg_model=cfg.surf_model_and_render, cfg_train=cfg.surf_train,
            xyz_min=xyz_min_fine, xyz_max=xyz_max_fine,
            data_dict=data_dict, stage='surf')
        recon.forward()
        eps_surf = time.time() - eps_surf
        eps_time_str = f'{eps_surf//3600:02.0f}:{eps_surf//60%60:02.0f}:{eps_surf%60:02.0f}'
        logger.info("+ "*10 + 'train: fine detail reconstruction in' + eps_time_str + " +"*10 )

        eps_time = time.time() - eps_time
        eps_time_str = f'{eps_time//3600:02.0f}:{eps_time//60%60:02.0f}:{eps_time%60:02.0f}'
        logger.info('train: finish (eps time' + eps_time_str + ')')



if __name__=='__main__':
    parser = config_parser()
    args = parser.parse_args()
    cfg = config.Config.fromfile(args.config)
    cfg.data.inst_seg_tag = args.inst_seg_tag
    if args.scene:
        cfg.expname += "{}".format(args.scene)
        cfg.data.datadir += "{}".format(args.scene)
    else:
        cfg.data.datadir += "{}".format(cfg.expname)
    if args.suffix:
        cfg.expname += "_" + args.suffix
    cfg.load_expname = args.load_expname if args.load_expname else cfg.expname
    if args.prefix:
        cfg.basedir = os.path.join(cfg.basedir, args.prefix)
    log_dir = os.path.join(cfg.basedir, cfg.expname, 'log')
    os.makedirs(log_dir, exist_ok=True)
    now = datetime.now()
    time_str = now.strftime('%Y-%m-%d_%H-%M-%S')
    logger = get_root_logger(logging.INFO, handlers=[
        logging.FileHandler(os.path.join(log_dir, '{}_train.log').format(time_str))])
    writer = SummaryWriter(log_dir=log_dir, filename_suffix=time_str)
    logger.info("+ "*10 + cfg.expname + " +"*10)
    logger.info("+ "*10 + log_dir + " +"*10)
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    seed_everything()
    if getattr(cfg, 'load_expname', None) is None:
        cfg.load_expname = args.load_expname if args.load_expname else cfg.expname
    logger.info(cfg.load_expname)
    os.makedirs(os.path.join(cfg.basedir, cfg.expname, 'recording'), exist_ok=True)
    if not args.render_only or args.mesh_from_sdf:
        copyfile('run.py', os.path.join(cfg.basedir, cfg.expname, 'recording', 'run.py'))
        copyfile(args.config, os.path.join(cfg.basedir, cfg.expname, 'recording', args.config.split('/')[-1]))

    import lib.voxurf_coarse as Model
    copyfile('lib/voxurf_coarse.py', os.path.join(cfg.basedir, cfg.expname, 'recording','voxurf_coarse.py'))


    # load images / poses / camera settings / data split
    data_dict = load_everything(args=args, cfg=cfg, device=device)

    # export scene bbox and camera poses in 3d for debugging and visualization
    if args.export_bbox_and_cams_only:
        logger.info('Export bbox and cameras...')
        xyz_min, xyz_max = compute_bbox_by_cam_frustrm(args=args, cfg=cfg, **data_dict)
        poses, HW, Ks, i_train = data_dict['poses'], data_dict['HW'], data_dict['Ks'], data_dict['i_train']
        near, far = data_dict['near'], data_dict['far']
        cam_lst = []
        for c2w, (H, W), K in zip(poses[i_train], HW[i_train], Ks[i_train]):
            rays_o, rays_d, viewdirs = Model.get_rays_of_a_view(
                H, W, K, c2w, cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y,)
            cam_o = rays_o[0,0].cpu().numpy()
            cam_d = rays_d[[0,0,-1,-1],[0,-1,0,-1]].cpu().numpy()
            cam_lst.append(np.array([cam_o, *(cam_o+cam_d*max(near, far*0.05))]))
        np.savez_compressed(args.export_bbox_and_cams_only,
                            xyz_min=xyz_min.cpu().numpy(), xyz_max=xyz_max.cpu().numpy(),
                            cam_lst=np.array(cam_lst))
        logger.info('done')
        sys.exit()

    if args.mesh_from_sdf:
        logger.info('Extracting mesh from sdf...')
        with torch.no_grad():
            ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'surf_last.tar')
            if os.path.exists(ckpt_path):
                new_kwargs = cfg.surf_model_and_render
            else:
                ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'fine_last.tar')
                new_kwargs = cfg.fine_model_and_render
            model, optimized_poses = utils.load_model(Model.Voxurf, ckpt_path, new_kwargs)
            model = model.to(device)
            prefix = args.prefix + '_' if args.prefix else ''
            prefix += args.suffix + '_' if args.suffix else ''
            gt_eval = 'dtu' in cfg.basedir
            validate_mesh(model, 512, threshold=0.0, prefix="{}final_mesh".format(prefix), world_space=True,
                          scale_mats_np=data_dict['scale_mats_np'], gt_eval=gt_eval, runtime=False, scene=args.scene, extract_color=args.extract_color)
        logger.info('done')
        sys.exit()

    # train
    if not args.render_only:
        train(args, cfg, data_dict)

    # load model for rendring
    if args.render_test or args.render_train or args.render_video or args.interpolate:
        if args.ft_path:
            ckpt_path = args.ft_path
            new_kwargs = cfg.fine_model_and_render
        else:
            ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'last_ckpt.tar')
            new_kwargs = cfg.surf_model_and_render

        ckpt_name = ckpt_path.split('/')[-1][:-4]
        model,optimized_poses = utils.load_model(Model.Voxurf, ckpt_path, new_kwargs)

        recon = scene_rep_reconstruction(
            args=args, cfg=cfg,
            cfg_model=cfg.surf_model_and_render, cfg_train=cfg.surf_train,
            xyz_min=torch.tensor([-1.,-1.,-1.]).cuda(), xyz_max=torch.tensor([1.,1.,1.]).cuda(),
            data_dict=data_dict, stage='surf')
        latest_step, opt_bg = recon.get_bg_model_barf(load_latest=True)
        optimized_poses = recon.current_pose
        opt_bg.H, opt_bg.W = recon.HW[0][0], recon.HW[0][1]
        opt_bg.device = device
        recon.model_bg = recon.model_bg.to(device)
        model = model.to(device)
        stepsize = cfg.fine_model_and_render.stepsize
        render_viewpoints_kwargs = {
            'model': model,
            'ndc': cfg.data.ndc,
            'render_kwargs': {
                'near': data_dict['near'],
                'far': data_dict['far'],
                'bg': 1 if cfg.data.white_bkgd else 0,
                'stepsize': stepsize,
                'inverse_y': cfg.data.inverse_y,
                'flip_x': cfg.data.flip_x,
                'flip_y': cfg.data.flip_y,
                'render_grad': True,
                'render_depth': True,
                'render_in_out': True,
            },
        }
    optimized_poses = optimized_poses.detach()
    if cfg.data.dataset_type=='custom':
        data_dict['poses'][data_dict['i_train']] = optimized_poses

    if args.interpolate and False:
        img_idx_0 = len(args.selected_id)//2
        img_idx_1 = len(args.selected_id)//2 + 1
        savedir = os.path.join(cfg.basedir, cfg.expname, f'interpolate_{img_idx_0}_{img_idx_1}')
        interpolate_view(recon.model_bg, savedir, img_idx_0, img_idx_1,
                         render_poses=data_dict['poses'],
                         HW=data_dict['HW'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
                         Ks=data_dict['Ks'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 1,1),
                         render_factor=args.render_video_factor,
                         **render_viewpoints_kwargs
                         )

    # render trainset and eval
    if args.render_train:
        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_train_{ckpt_name}')
        os.makedirs(testsavedir, exist_ok=True)
        render_poses = data_dict['poses'][data_dict['i_train']]
        Ks = data_dict['Ks'][data_dict['i_train']]
        HW = data_dict['HW'][data_dict['i_train']]
        gt_img = data_dict['images']
        results = []
        for i in trange(data_dict['poses'][data_dict['i_train']].shape[0]):
            image_gt, image_mip, depth, image_object, result_nums = visualize_test_image(model, recon.model_bg, gt_img, render_poses,
                                                    Ks, HW, latest_step, opt_bg, cfg.data.ndc,
                                                    render_viewpoints_kwargs['render_kwargs'],id=i)
            results.append(result_nums)
            filename = os.path.join(testsavedir, 'gt_{:03d}.png'.format(i))
            imageio.imwrite(filename, np.uint8(image_gt * 255))
            if cfg.data.dataset_type == 'scene_with_shapenet':
                filename = os.path.join(testsavedir, 'gt_depth_{:03d}.png'.format(i))
                depth_gt = data_dict['depths'][i]
                depth_gt = visualize_depth(torch.tensor(depth_gt))
                depth_gt = einops.rearrange(depth_gt, 'b h w -> h w b')
                imageio.imwrite(filename, np.uint8(depth_gt * 255))
            filename = os.path.join(testsavedir, 'rgb_{:03d}.png'.format(i))
            imageio.imwrite(filename, np.uint8(image_mip * 255))
            filename = os.path.join(testsavedir, 'depth_{:03d}.png'.format(i))
            imageio.imwrite(filename, np.uint8(depth.cpu().numpy() * 255))
            filename = os.path.join(testsavedir, 'object_{:03d}.png'.format(i))
            imageio.imwrite(filename, np.uint8(image_object*255))
        with open(testsavedir+'/result_train.txt','w') as file:
            for result in results:
                tensor_str = ' '.join(f"{elem:.3f}" for elem in result.cpu().numpy())
                file.write(tensor_str+'\n')


    logger.info('Done')
