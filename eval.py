import os, sys, copy, glob, json, time, random, argparse, cv2
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import einops
from lib import utils_vis
from lib import common
from shutil import copyfile
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
import yaml
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
from lib.utils_vis import visualize_depth
from lib.align_trajectories import align_ate_c2b_use_a2b

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



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
    parser.add_argument("--inst_seg_tag", type=int, default=3)
    # logging/saving options
    parser.add_argument("--i_print", type=int, default=200,
                        help='frequency of console printout and metric loggin')

    parser.add_argument("--i_validate_mesh", type=int, default=5000,
                        help='step of validating mesh')

    parser.add_argument("--i_validate", type=int, default=500)
    parser.add_argument("--i_weights", type=int, default=10000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("-s", "--suffix", type=str, default="",
                        help='suffix for exp name')
    parser.add_argument("-p", "--prefix", type=str, default="",
                        help='prefix for exp name')
    parser.add_argument("--load_density_only", type=int, default=1)
    parser.add_argument("--load_expname", type=str, default="") # dvgo_Statues_original
    parser.add_argument("--sdf_mode", type=str, default="density")
    parser.add_argument("--selected_id", nargs='+', type=int, default=None)
    parser.add_argument("--scene", type=str, default=0)
    parser.add_argument("--no_dvgo_init", action='store_true')
    parser.add_argument("--run_dvgo_init", action='store_true')
    parser.add_argument("--interpolate", default='0_1')
    parser.add_argument("--extract_color", action='store_true')
    parser.add_argument("--barf_opt", type=str, default='./lib/barf_model/barf_acc.yaml')
    # mipnerf
    parser.add_argument("--out_dir", help="Output directory.", type=str, default='./out')
    parser.add_argument("--dataset_name", help="Single or multi data.", type=str, default="blender")
    parser.add_argument("--mip_config", help="Path to config file.", required=False, default='./lib/mipnerf/configs/')
    parser.add_argument("--opts", nargs=argparse.REMAINDER,default=None,
                        help="Modify hparams. Example: train.py resume out_dir TRAIN.BATCH_SIZE 2")

    return parser


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
    rgb_map_bg, depth_map_bg, opacity = model_bg.visualize(opt, ret, split="val",append_cbar=False)
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
        fine_rgb, depth_bg, opacity = model_bg.visualize(opt, ret, split="val", append_cbar=False )# (4, 3, H, W)


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
def interpolate_view(model_bg, savedir, img_idx_0, img_idx_1, render_poses, HW, Ks, ndc, repeat=1, **render_kwargs):
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
         'poses','render_poses', 'images','images_gray', 'scale_mats_np', 'masks','matcher_infos','sg_config'}  #
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

@torch.no_grad()
def log_scalars(loss,loss_weight,metric=None,step=0,split="train",writer=None):
    for key, value in loss.items():
        if key=="all": continue
        if loss_weight[key] is not None:
            writer.add_scalar("{0}/loss_{1}".format(split,key),value,step)
    if metric is not None:
        for key,value in metric.items():
            writer.add_scalar("{0}/{1}".format(split,key),value,step)

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
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
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



@torch.no_grad()
def prealign_w2c_large_camera_systems(pose_w2c, pose_GT_w2c):
    """Compute the 3D similarity transform relating pose_w2c to pose_GT_w2c. Save the inverse
    transformation for the evaluation, where the test poses must be transformed to the coordinate
    system of the optimized poses.

    Args:
        opt (edict): settings
        pose_w2c (torch.Tensor): Shape is (B, 3, 4)
        pose_GT_w2c (torch.Tensor): Shape is (B, 3, 4)
    """
    pose_c2w = camera.pose.invert(pose_w2c)
    pose_GT_c2w = camera.pose.invert(pose_GT_w2c)
    try:
        pose_aligned_c2w, ssim_est_gt_c2w = align_ate_c2b_use_a2b(pose_c2w, pose_GT_c2w, method='sim3')
        pose_aligned_w2c = camera.pose.invert(pose_aligned_c2w[:, :3])
        ssim_est_gt_c2w.type = 'traj_align'
    except:
        logger.info("warning: SVD did not converge...")
        pose_aligned_w2c = pose_w2c
        ssim_est_gt_c2w = edict(R=torch.eye(3, device=device).unsqueeze(0), type='traj_align',
                                t=torch.zeros(1, 3, 1, device=device), s=1.)
    return pose_aligned_w2c, ssim_est_gt_c2w


@torch.no_grad()
def prealign_w2c_small_camera_systems(pose_w2c, pose_GT_w2c):
    """Compute the transformation from pose_w2c to pose_GT_w2c by aligning the each pair of pose_w2c
    to the corresponding pair of pose_GT_w2c and computing the scaling. This is more robust than the
    technique above for small number of input views/poses (<10). Save the inverse
    transformation for the evaluation, where the test poses must be transformed to the coordinate
    system of the optimized poses.

    Args:
        opt (edict): settings
        pose_w2c (torch.Tensor): Shape is (B, 3, 4)
        pose_GT_w2c (torch.Tensor): Shape is (B, 3, 4)
    """

    def alignment_function(poses_c2w_from_padded: torch.Tensor,
                           poses_c2w_to_padded: torch.Tensor, idx_a: int, idx_b: int):
        """Args: FInd alignment function between two poses at indixes ix_a and idx_n

            poses_c2w_from_padded: Shape is (B, 4, 4)
            poses_c2w_to_padded: Shape is (B, 4, 4)
            idx_a:
            idx_b:

        Returns:
        """
        # We take a copy to keep the original poses unchanged.
        poses_c2w_from_padded = poses_c2w_from_padded.clone()
        # We use the distance between the same two poses in both set to obtain
        # scale misalgnment.
        dist_from = torch.norm(
            poses_c2w_from_padded[idx_a, :3, 3] - poses_c2w_from_padded[idx_b, :3, 3]
        )
        dist_to = torch.norm(
            poses_c2w_to_padded[idx_a, :3, 3] - poses_c2w_to_padded[idx_b, :3, 3])
        scale = dist_to / dist_from

        # alternative for scale
        # dist_from = poses_w2c_from_padded[idx_a, :3, 3] @ poses_c2w_from_padded[idx_b, :3, 3]
        # dist_to = poses_w2c_to_padded[idx_a, :3, 3] @ poses_c2w_to_padded[idx_b, :3, 3]
        # scale = onp.abs(dist_to /dist_from).mean()

        # We bring the first set of poses in the same scale as the second set.
        poses_c2w_from_padded[:, :3, 3] = poses_c2w_from_padded[:, :3, 3] * scale

        # Now we simply apply the transformation that aligns the first pose of the
        # first set with first pose of the second set.
        transformation_from_to = poses_c2w_to_padded[idx_a] @ camera.pose_inverse_4x4(
            poses_c2w_from_padded[idx_a])
        poses_aligned_c2w = transformation_from_to[None] @ poses_c2w_from_padded

        poses_aligned_w2c = camera.pose_inverse_4x4(poses_aligned_c2w)
        ssim_est_gt_c2w = edict(R=transformation_from_to[:3, :3].unsqueeze(0), type='traj_align',
                                t=transformation_from_to[:3, 3].reshape(1, 3, 1), s=scale)

        return poses_aligned_w2c[:, :3], ssim_est_gt_c2w

    pose_c2w = camera.pose.invert(pose_w2c)
    pose_GT_c2w = camera.pose.invert(pose_GT_w2c)
    B = pose_c2w.shape[0]



    # try every combination of pairs and get the rotation/translation
    # take the one with the smallest error
    # this is because for small number of views, the procrustes alignement with SVD is not robust.
    pose_aligned_w2c_list = []
    ssim_est_gt_c2w_list = []
    error_R_list = []
    error_t_list = []
    full_error = []
    for pair_id_0 in range(min(B, 10)):  # to avoid that it is too long
        for pair_id_1 in range(min(B, 10)):
            if pair_id_0 == pair_id_1:
                continue
            pose_aligned_w2c_, ssim_est_gt_c2w_ = alignment_function \
                (camera.pad_poses(pose_c2w), camera.pad_poses(pose_GT_c2w),
                 pair_id_0, pair_id_1)
            pose_aligned_w2c_list.append(pose_aligned_w2c_)
            ssim_est_gt_c2w_list.append(ssim_est_gt_c2w_)

            error = evaluate_camera_alignment(pose_aligned_w2c_, pose_GT_w2c)
            error_R_list.append(error.R.mean().item() * 180. / np.pi)
            error_t_list.append(error.t.mean().item())
            full_error.append(error.t.mean().item() * (error.R.mean().item() * 180. / np.pi))

        ind_best = np.argmin(full_error)
        # print(np.argmin(error_R_list), np.argmin(error_t_list), ind_best)
        pose_aligned_w2c = pose_aligned_w2c_list[ind_best]
        ssim_est_gt_c2w = ssim_est_gt_c2w_list[ind_best]

    return pose_aligned_w2c, ssim_est_gt_c2w

def backtrack_from_aligning_the_trajectory(pose_GT_w2c, ssim_est_gt_c2w):
    pose_GT_c2w = camera.pose.invert(pose_GT_w2c)
    R_GT_c2w_aligned = ssim_est_gt_c2w.R.transpose(-2, -1) @ pose_GT_c2w[:, :3, :3]
    t_GT_c2w_aligned = ssim_est_gt_c2w.R.transpose(-2, -1) / ssim_est_gt_c2w.s @ (pose_GT_c2w[:, :3, 3:4] - ssim_est_gt_c2w.t)
    pose_GT_c2w_aligned = camera.pose(R=R_GT_c2w_aligned,t=t_GT_c2w_aligned.reshape(-1, 3))
    pose_w2d_recovered = camera.pose.invert(pose_GT_c2w_aligned)
    return pose_w2d_recovered


def get_w2c_pose_test_optimize(data_dict):
    pose_GT_w2c = data_dict.pose
    ssim_est_gt_c2w = data_dict.sim3_est_to_gt_c2w
    if ssim_est_gt_c2w.type == 'traj_align':
        pose = backtrack_from_aligning_the_trajectory(pose_GT_w2c, ssim_est_gt_c2w)
    else:
        raise ValueError
    # Here, we align the test pose to the poses found during the optimization (otherwise wont be valid)
    # that's pose. And can learn an extra alignement on top
    # additionally factorize the remaining pose imperfection
    pose = camera.pose.compose([data_dict.pose_refine_test,pose])
    return pose

@torch.enable_grad()
def evaluate_test_time_photometric_optim(model,data_dict, gt_img, Ks,opt):
    """Run test-time optimization. Optimizes over data_dict.se3_refine_test"""
    # only optimizes for the test pose here
    data_dict.se3_refine_test = torch.nn.Parameter(torch.zeros(1, 6, device=opt['device']))
    optimizer = getattr(torch.optim, 'Adam')
    optim_pose = optimizer([dict(params=[data_dict.se3_refine_test], lr=5e-4)])
    # iterator = tqdm.trange(opt.optim.test_iter,desc="test-time optim.",leave=False,position=1)
    for it in range(500):
        optim_pose.zero_grad()
        data_dict.pose_refine_test = camera.lie.se3_to_SE3(data_dict.se3_refine_test)
        poses_w2c = get_w2c_pose_test_optimize(data_dict)  # is it world to camera
        data_dict.poses_w2c = poses_w2c
        rays_bg, ray_idx = model.forward_graph(opt, poses_w2c, intr=Ks, mode='train')
        target_bg = gt_img.view(1, opt.H* opt.W, 3)
        target_bg = target_bg[:, ray_idx].reshape(-1, 3)
        loss = F.mse_loss(rays_bg['rgb'], target_bg)
        # loss, psnr, _ = model.training_step(batch=[rays_bg, target_bg], global_step=1e10)
        # current estimate of the pose
        loss.backward()
        optim_pose.step()
    return data_dict

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
        self.cfg_model.world_bound_scale = 2.5 #1.05
        if abs(cfg_model.world_bound_scale - 1) > 1e-9:
            xyz_shift = (xyz_max - xyz_min) * (cfg_model.world_bound_scale - 1) / 2
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
        self.rect_size = ((xyz_max - xyz_min) / (self.cfg_model.world_bound_scale * 1.05)).tolist()
        self.range_shape = (xyz_max - xyz_min) / (self.cfg_model.world_bound_scale * 1.05)
        if cfg.surf_model_and_render.load_sdf:
            sdf_grid_path = os.path.join(cfg.data.datadir, 'sdf_grid.npy')
            if os.path.exists(sdf_grid_path):
                sdf_dict = np.load(sdf_grid_path, allow_pickle=True).tolist()
                # sdf0 = torch.tensor(sdf_dict['sdf_grid_xyz']).to(xyz_min.device)
                # sdf0 = rearrange(sdf0, 'd h w -> 1 1 d h w')
                range_shape = (sdf_dict['xyz_max'] - sdf_dict['xyz_min']) * self.cfg_model.world_bound_scale / 2
                xyz_min = -torch.tensor(range_shape).to(self.device).to(torch.float32)
                xyz_max = torch.tensor(range_shape).to(self.device).to(torch.float32)
                self.rect_size = ((xyz_max - xyz_min) / (self.cfg_model.world_bound_scale * 1.05)).tolist()
                self.range_shape = (xyz_max - xyz_min) / (self.cfg_model.world_bound_scale * 1.05)
        self.last_ckpt_path = os.path.join(cfg.basedir, cfg.expname, f'{stage}_pose_last.tar')
        # init model
        model_kwargs = copy.deepcopy(cfg_model)
        scale_ratio = getattr(cfg_train, 'scale_ratio', 2)
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
            rect_size=self.rect_size,
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
                                                                              global_step=-1,
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

    def get_project_error(self,global_step,current_pose, coord0, coord1, i_train, j_train,
                          mconf, use_deform=True, **render_kwargs):
        coord = torch.concat([coord0, coord1], dim=0)
        index = np.concatenate([i_train, j_train], axis=0)
        mconf = torch.concat([mconf, mconf], dim=0)
        rays_o_p, rays_d_p = get_ray_dir(coord, self.Ks[index], c2w=camera.pose.invert(current_pose[index]),
                                         inverse_y=self.cfg.data.inverse_y,
                                         flip_x=self.cfg.data.flip_x, flip_y=self.cfg.data.flip_y, mode='no_center')
        rays_o_p = einops.rearrange(rays_o_p, 'b n c ->(b n) c', c=3)
        rays_d_p = einops.rearrange(rays_d_p, 'b n c ->(b n) c', c=3)
        if use_deform:
            query_points, mask_valid, sdf_ray_step = self.model.query_sdf_point_wocuda(rays_o_p, rays_d_p,
                                                                                                global_step=global_step,
                                                                                                keep_dim=True,
                                                                                                **render_kwargs)
        else:
            query_points, mask_valid, sdf_ray_step = self.model.query_sdf_point_wocuda_wodeform(rays_o_p, rays_d_p,
                                                                              global_step=global_step,
                                                                              keep_dim=True, **render_kwargs)
        query_points = einops.rearrange(query_points, '(b n) c ->b n c', b=len(index), c=3)
        mask_valid = einops.rearrange(mask_valid, '(b n)->b n', b=len(index))
        # mask_valid = mask_valid[0:len(i_train)] * mask_valid[len(i_train):]
        # mask_valid = torch.concat([mask_valid, mask_valid], dim=0)
        ray_near_surface, _ = torch.min(abs(sdf_ray_step), dim=-1)
        ray_near_surface = einops.rearrange(ray_near_surface, '(b n)->b n', b=len(index))
        near_surface_loss = (ray_near_surface * (mconf > 0)).mean()
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
        valid_corr = diff.detach().le(200)
        valid = valid & valid_corr
        projection_dis_error = compute_diff_loss('huber', diff, weights=mconf, mask=valid, delta=1.)
        return near_surface_loss, projection_dis_error

    def get_project_feature_loss(self, global_step,current_pose, imsz,target_tr,rays_o_tr,rays_d_tr, i_list,j_list):
        num_min = min(1024, imsz)
        indices = [(torch.randperm(imsz, device=target_tr.device)[:num_min])]
        rays_o_tr = einops.rearrange(rays_o_tr[indices], '(b n) c ->b n c', b=len(i_list),c=3)
        rays_d_tr = einops.rearrange(rays_d_tr[indices], '(b n) c ->b n c', b=len(i_list),c=3)

        loss_surface_projection = 0
        rays_o = rays_o_tr.reshape(-1, 3)
        rays_d = rays_d_tr.reshape(-1, 3)
        camera_pose_0 = current_pose[i_list]  # true pose
        camera_pose_1 = current_pose[j_list]
        query_points, mask_valid, _ = self.model.query_sdf_point_wocuda(rays_o, rays_d,
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
        query_points_ref, valid_point_ref, _ = self.model.query_sdf_point_wocuda(rays_o_ref, rays_d_ref,
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
        # valid_mask = valid_mask*valid_depth_ray
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
            # mask_tr = mask_tr[indices]
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


if __name__=='__main__':
    # load setup
    parser = config_parser()
    args = parser.parse_args()
    cfg = config.Config.fromfile(args.config)
    # reset the root by the scene id
    if args.scene:
        cfg.expname += "{}".format(args.scene)
        cfg.data.datadir += "{}".format(args.scene)
    else:
        cfg.data.datadir += "{}".format(cfg.expname)

    if args.suffix:
        cfg.expname += "_" + args.suffix
    cfg.load_expname = args.load_expname if args.load_expname else cfg.expname
    if args.prefix:
        cfg.evaldir = os.path.join(cfg.evaldir, args.prefix)
        cfg.basedir = os.path.join(cfg.basedir, args.prefix)
    log_dir = os.path.join(cfg.evaldir, cfg.expname, 'eval')

    os.makedirs(log_dir, exist_ok=True)
    now = datetime.now()
    time_str = now.strftime('%Y-%m-%d_%H-%M-%S')
    logger = get_root_logger(logging.INFO, handlers=[
        logging.FileHandler(os.path.join(log_dir, '{}_eval.log').format(time_str))])
    writer = SummaryWriter(log_dir=log_dir, filename_suffix=time_str)
    logger.info("+ "*10 + cfg.expname + " +"*10)
    logger.info("+ "*10 + log_dir + " +"*10)

    # init enviroment
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    seed_everything()
    if getattr(cfg, 'load_expname', None) is None:
        cfg.load_expname = args.load_expname if args.load_expname else cfg.expname
    logger.info(cfg.load_expname)
    os.makedirs(os.path.join(cfg.evaldir, cfg.expname, 'recording'), exist_ok=True)
    if not args.render_only or args.mesh_from_sdf:
        copyfile('run.py', os.path.join(cfg.evaldir, cfg.expname, 'recording', 'run.py'))
        copyfile(args.config, os.path.join(cfg.evaldir, cfg.expname, 'recording', args.config.split('/')[-1]))
    import lib.dvgo_ori as dvgo_ori
    if args.sdf_mode == "voxurf_coarse":
        import lib.voxurf_coarse as Model
        copyfile('lib/voxurf_coarse.py', os.path.join(cfg.evaldir, cfg.expname, 'recording','voxurf_coarse.py'))
    elif args.sdf_mode == "voxurf_fine":
        import lib.voxurf_fine as Model
        copyfile('lib/voxurf_fine.py', os.path.join(cfg.evaldir, cfg.expname, 'recording','voxurf_fine.py'))
    elif args.sdf_mode == "voxurf_womask_coarse":
        import lib.voxurf_womask_coarse as Model
        copyfile('lib/voxurf_womask_coarse.py', os.path.join(cfg.evaldir, cfg.expname, 'recording','voxurf_womask_coarse.py'))
    elif args.sdf_mode == "voxurf_womask_fine":
        import lib.voxurf_womask_fine as Model
        copyfile('lib/voxurf_womask_fine.py', os.path.join(cfg.evaldir, cfg.expname, 'recording','voxurf_womask_fine.py'))
    else:
        raise NameError
    # load images / poses / camera settings / data split
    data_dict = load_everything(args=args, cfg=cfg, device=device)

    # load model for rendring
    if args.render_test or args.render_train or args.render_video or args.interpolate:
        if args.ft_path:
            ckpt_path = args.ft_path
            new_kwargs = cfg.fine_model_and_render
        else:
            ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'last_ckpt.tar')
            new_kwargs = cfg.surf_model_and_render
        ckpt_name = ckpt_path.split('/')[-1][:-4]
        # model, optimized_poses = utils.load_model(Model.Voxurf, ckpt_path, new_kwargs)
        # model = model.to(device)
        model = None
        recon = scene_rep_reconstruction(
            args=args, cfg=cfg,
            cfg_model=cfg.surf_model_and_render, cfg_train=cfg.surf_train,
            xyz_min=torch.tensor([-1., -1., -1.]).cuda(), xyz_max=torch.tensor([1., 1., 1.]).cuda(),
            data_dict=data_dict, stage='surf')
        latest_step, opt_bg = recon.get_bg_model_barf(load_latest=True)
        optimized_poses = recon.current_pose
        # opt_bg['h'], opt_bg['w'], opt_bg['focal'] = data_dict['hwf']
        opt_bg.H, opt_bg.W = recon.HW[0][0], recon.HW[0][1]
        opt_bg.device = device
        recon.model_bg = recon.model_bg.to(device)

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
    # if cfg.data.dataset_type=='custom':
    #     data_dict['poses'][data_dict['i_train']] = optimized_poses
    #     # data_dict['poses'][data_dict['i_val']] = optimized_poses
    #     # data_dict['poses'][data_dict['i_test']] = optimized_poses

    if args.interpolate and False:
        img_idx_0 = len(cfg.data.selected_id)//2
        img_idx_1 = len(cfg.data.selected_id)//2 + 1
        savedir = os.path.join(cfg.evaldir, cfg.expname, f'interpolate_{img_idx_0}_{img_idx_1}')
        interpolate_view(recon.model_bg, savedir, img_idx_0, img_idx_1,
                         render_poses=data_dict['poses'],
                         HW=data_dict['HW'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
                         Ks=data_dict['Ks'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 1,1),
                         render_factor=args.render_video_factor,
                         **render_viewpoints_kwargs
                         )




    def get_all_training_poses(pose_w2c, pose_GT_w2c, idx_optimized_pose=None):
        if idx_optimized_pose is not None:
            pose_GT_w2c = pose_GT_w2c[idx_optimized_pose].reshape(-1, 3, 4)
        return pose_w2c, pose_GT_w2c


    optimized_poses = optimized_poses[0:len(cfg.data.selected_id)]
    # render testset and eval

    pose_pred, pose_GT = get_all_training_poses(optimized_poses, data_dict['poses'][data_dict['i_train']])
    if pose_pred.shape[0] > 9:
        # alignment of the trajectory
        pose_aligned, sim3_est_to_gt_c2w = prealign_w2c_large_camera_systems(pose_pred, pose_GT)
    else:
        # alignment of the first cameras
        pose_aligned, sim3_est_to_gt_c2w = prealign_w2c_small_camera_systems(pose_pred, pose_GT)
    error = evaluate_camera_alignment(pose_aligned, pose_GT)
    fig = plt.figure(figsize=(10, 10))
    cam_path = os.path.join(cfg.evaldir, cfg.expname)
    utils_vis.plot_save_poses_blender(fig, pose_aligned.detach().cpu(), pose_ref=pose_GT.detach().cpu(),
                                      path=cam_path, ep='-1')

    logger.info("--------------------------")
    logger.info("rot:   {:8.3f}".format(error.R.mean().cpu()))
    logger.info("trans: {:10.5f}".format(error.t.mean()))
    logger.info("--------------------------")
    recon.model_bg.eval()
    testsavedir = os.path.join(cfg.evaldir, cfg.expname, f'render_test_{ckpt_name}')
    os.makedirs(testsavedir, exist_ok=True)
    if args.render_test:
        render_poses = data_dict['poses'][data_dict['i_test']]
        Ks = data_dict['Ks'][data_dict['i_test']]
        HW = data_dict['HW'][data_dict['i_test']]
        test_img_id = data_dict['i_train'].shape[0]+data_dict['i_val'].shape[0]
        gt_img = data_dict['images'][test_img_id:]
        if cfg.data.dataset_type=='scene_with_shapenet':
            gt_depths = data_dict['depths'][test_img_id:]

        logger.info("---------------------------------")
        logger.info("---------------------------------")
        logger.info("---------------------------------")
        logger.info("optimize pose in the test stage:")
        def get_test_id(train_id):
            testId = []
            for i in range(len(train_id)-1):
                current_num = train_id[i]
                next_num = train_id[(i + 1)]
                if current_num < next_num:
                    interval = list(range(current_num + 1, next_num))
                else:
                    interval = list(range(current_num + 1, 100)) + list(range(0, next_num))
                testId +=interval
            return testId

        results = []
        test_id = get_test_id(cfg.data.selected_id)
        for i in tqdm(range(data_dict['poses'][data_dict['i_test']].shape[0])):
            if cfg.data.dataset_type != 'dtu':
                if i not in test_id:
                    continue
            optimized_dict = edict()
            optimized_dict.sim3_est_to_gt_c2w = sim3_est_to_gt_c2w
            optimized_dict.pose = render_poses[i][None]
            optimized_dict = evaluate_test_time_photometric_optim(recon.model_bg,optimized_dict,gt_img[i][None], Ks[i][None],opt_bg)
            render_poses[i]=optimized_dict.poses_w2c.squeeze(0).detach()
            image_gt, image_mip, depth, image_object, result_nums = visualize_test_image(model, recon.model_bg, gt_img,
                                                                                         render_poses,
                                                                                         Ks, HW,
                                                                                         latest_step, opt_bg,
                                                                                         cfg.data.ndc,
                                                                                         render_viewpoints_kwargs[
                                                                                             'render_kwargs'], id=i)
            results.append(result_nums)
            if cfg.data.dataset_type=='scene_with_shapenet':
                filename = os.path.join(testsavedir, 'gt_depth_{:03d}.png'.format(i))
                depth_gt = gt_depths[i]
                depth_gt = visualize_depth(torch.tensor(depth_gt))
                depth_gt = einops.rearrange(depth_gt, 'b h w -> h w b')
                imageio.imwrite(filename, np.uint8(depth_gt * 255))


            filename = os.path.join(testsavedir, 'optimized_rgb_{:03d}.png'.format(i))
            imageio.imwrite(filename, np.uint8(image_mip * 255))
            filename = os.path.join(testsavedir, 'optimized_depth_{:03d}.png'.format(i))
            imageio.imwrite(filename, np.uint8(depth * 255))
            filename = os.path.join(testsavedir, 'optimized_object_{:03d}.png'.format(i))
            imageio.imwrite(filename, np.uint8(image_object * 255))

        with open(testsavedir + '/result_train.txt', 'w') as file:
            file.write("rot:   {:8.3f}".format(error.R.mean().cpu()) + '\n')
            file.write("trans: {:10.5f}".format(error.t.mean()) + '\n')
            mean_result = torch.stack(results).mean(dim=0)
            tensor_str = "mean: " + ' '.join(f"{elem:.3f}" for elem in mean_result.cpu().numpy())
            file.write(tensor_str + '\n')
            for result in results:
                tensor_str = ' '.join(f"{elem:.3f}" for elem in result.cpu().numpy())
                file.write(tensor_str + '\n')


    if args.render_video:
        from lib.gen_videos import generate_videos_synthesis

        intr = data_dict['Ks'][data_dict['i_train']]
        HW = data_dict['HW'][data_dict['i_test']]
        generate_videos_synthesis(recon.model_bg, pose_pred, intr, opt_bg, cfg, testsavedir,latest_step, eps=1e-10)

    logger.info('Done')
