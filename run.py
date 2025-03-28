import argparse
import os
import random
import sys
import time

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import einops
from lib.recon_scene import scene_rep_reconstruction
from shutil import copyfile
from tqdm import trange
from mmengine import config
import imageio
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import trimesh
import logging
import torch
from datetime import datetime
from lib import utils, dtu_eval
from torch.utils.tensorboard import SummaryWriter
from lib.load_data import load_data
from lib.utils import  get_root_logger
from lib import camera, vgg_loss
from easydict import EasyDict as edict
from lib.nvs_fun import visualize_test_image
from lib.utils_vis import visualize_depth
import lib.voxurf_coarse as Model


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
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', required=True,
                        help='config file path')
    parser.add_argument("--seed", type=int, default=777,
                        help='Random seed')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    # testing options
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true')
    parser.add_argument("--render_train", action='store_true')
    parser.add_argument("--render_video", action='store_true')

    parser.add_argument("--i_print", type=int, default=200,
                        help='frequency of console printout and metric loggin')

    parser.add_argument("--inst_seg_tag", type=int, default=0,
                        help='the id of pose probe for toydesk dataset')
    parser.add_argument("--i_validate", type=int, default=5000,
                        help='step of validating mesh')
    parser.add_argument("--i_validate_mesh", type=int, default=2000,
                        help='step of validating mesh')

    parser.add_argument("-s", "--suffix", type=str, default="",
                        help='suffix for exp name')
    parser.add_argument("-p", "--prefix", type=str, default="",
                        help='prefix for exp name')
    return parser


def validate_mesh(model, cfg, resolution=128, threshold=0.0, prefix="", world_space=False,
                  scale_mats_np=None, gt_eval=False, runtime=True, scene=122, smooth=True,
                  extract_color=False):
    # os.makedirs(os.path.join(self.cfg.basedir, self.cfg.expname, 'meshes'), exist_ok=True)
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
        vertex_colors = (torch.concat(vertex_colors).cpu().detach().numpy() * 255.).astype(np.uint8)
        mesh = trimesh.Trimesh(vertices, triangles, vertex_colors=vertex_colors)
    else:
        mesh = trimesh.Trimesh(vertices, triangles)
    mesh_path = os.path.join(cfg.basedir, cfg.expname, 'meshes', "{}_".format(scene) + prefix + '.ply')
    mesh.export(mesh_path)
    if gt_eval:
        mean_d2s, mean_s2d, over_all = dtu_eval.eval(mesh_path, scene=scene,
                                                     eval_dir=os.path.join(cfg.basedir, cfg.expname, 'meshes'),
                                                     dataset_dir='data/DTU', suffix=prefix + 'eval', use_o3d=False,
                                                     runtime=runtime)
        res = "standard point cloud sampling" if not runtime else "down sampled point cloud for fast eval (NOT standard!):"
        logger.info("mesh evaluation with {}".format(res))
        logger.info(" [ d2s: {:.3f} | s2d: {:.3f} | mean: {:.3f} ]".format(mean_d2s, mean_s2d, over_all))
        return over_all
    return 0.





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

    eps_surf = time.time()
    recon = scene_rep_reconstruction(
        args=args, cfg=cfg, logger=logger,
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
    cfg.data.datadir += "{}".format(cfg.expname)
    if args.suffix:
        cfg.expname += "_" + args.suffix
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

    os.makedirs(os.path.join(cfg.basedir, cfg.expname, 'recording'), exist_ok=True)
    if not args.render_only:
        copyfile('run.py', os.path.join(cfg.basedir, cfg.expname, 'recording', 'run.py'))
        copyfile(args.config, os.path.join(cfg.basedir, cfg.expname, 'recording', args.config.split('/')[-1]))
        copyfile('lib/voxurf_coarse.py', os.path.join(cfg.basedir, cfg.expname, 'recording', 'voxurf_coarse.py'))

    # load images / poses / camera settings / data split
    data_dict = load_everything(args=args, cfg=cfg, device=device)

    # train
    if not args.render_only:
        train(args, cfg, data_dict)

    # load model for rendering
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
            args=args, cfg=cfg, logger=logger,
            cfg_model=cfg.surf_model_and_render, cfg_train=cfg.surf_train,
            xyz_min=torch.tensor([-1.,-1.,-1.]).cuda(), xyz_max=torch.tensor([1.,1.,1.]).cuda(),
            data_dict=data_dict, stage='surf')
        latest_step, opt_bg = recon.get_bg_model(load_latest=True)
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


    if args.render_train:
        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_train_{ckpt_name}')
        os.makedirs(testsavedir, exist_ok=True)
        render_poses = data_dict['poses'][data_dict['i_train']]
        Ks = data_dict['Ks'][data_dict['i_train']]
        HW = data_dict['HW'][data_dict['i_train']]
        gt_img = data_dict['images']
        results = []
        recon.run_eval(recon.model_bg.settings, testsavedir, cfg.expname, split='i_train')
        for i in trange(data_dict['poses'][data_dict['i_train']].shape[0]):
            image_gt, image_mip, depth, image_object, result_nums = visualize_test_image(model, None, gt_img, render_poses,
                                                    cfg, Ks, HW, latest_step, opt_bg, cfg.data.ndc,
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
            filename = os.path.join(testsavedir, 'object_{:03d}.png'.format(i))
            imageio.imwrite(filename, np.uint8(image_object*255))
        with open(testsavedir+'/result_train.txt','w') as file:
            for result in results:
                tensor_str = ' '.join(f"{elem:.3f}" for elem in result.cpu().numpy())
                file.write(tensor_str+'\n')


    if args.render_test:
        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_test_{ckpt_name}')
        os.makedirs(testsavedir, exist_ok=True)
        render_poses = data_dict['poses'][data_dict['i_test']]
        Ks = data_dict['Ks'][data_dict['i_test']]
        HW = data_dict['HW'][data_dict['i_test']]
        gt_img = data_dict['images']
        results = []
        recon.run_eval(recon.model_bg.settings, testsavedir, cfg.expname, split='i_test')
        for i in trange(data_dict['poses'][data_dict['i_test']].shape[0]):
            image_gt, image_mip, depth, image_object, result_nums = visualize_test_image(model, None, gt_img, render_poses,
                                                    cfg, Ks, HW, latest_step, opt_bg, cfg.data.ndc,
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

            filename = os.path.join(testsavedir, 'object_{:03d}.png'.format(i))
            imageio.imwrite(filename, np.uint8(image_object*255))
        with open(testsavedir+'/result_test.txt','w') as file:
            mean_result = torch.stack(results).mean(dim=0)
            tensor_str = "mean: " + ' '.join(f"{elem:.3f}" for elem in mean_result.cpu().numpy())
            file.write(tensor_str + '\n')
            for result in results:
                tensor_str = ' '.join(f"{elem:.3f}" for elem in result.cpu().numpy())
                file.write(tensor_str+'\n')

    if args.render_video:
        save_dir = os.path.join(cfg.basedir, cfg.expname)
        os.makedirs(save_dir, exist_ok=True)
        recon.novel_view(recon.model_bg.settings, save_dir)

    logger.info('Done')



