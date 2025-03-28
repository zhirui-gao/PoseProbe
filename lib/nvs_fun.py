import os
import lib.voxurf_coarse as Model
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import einops
from tqdm import tqdm
import imageio
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import logging
import torch
import torch.nn.functional as F
from lib import utils
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
from lib import camera
from lib.utils_vis import visualize_depth
import os, random


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


def render_viewpoints(model, render_poses, cfg, HW, Ks, ndc, render_kwargs,
                      gt_imgs=None, masks=None, savedir=None, render_factor=0, idx=None,
                      eval_ssim=True, eval_lpips_alex=True, eval_lpips_vgg=True,
                      use_bar=True, step=0, rgb_only=False):
    '''Render images for the given viewpoints; run evaluation if gt given.
    '''
    if render_poses.shape[1] == 4:
        render_poses = render_poses[:, :3, :]
    assert len(render_poses) == len(HW) and len(HW) == len(Ks)
    render_poses = camera.pose.invert(render_poses)  # gzr
    if render_factor != 0:
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
            {k: v for k, v in model.inference(ro, rd, vd, training=False, **render_kwargs).items() if k in keys}
            for ro, rd, vd in zip(rays_o.split(4096, 0), rays_d.split(4096, 0), viewdirs.split(4096, 0))
        ]
        render_result = {
            k: torch.cat([ret[k] for ret in render_result_chunks]).reshape(H, W, -1)
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
                mask = masks[i].cpu().numpy()  # .reshape(H, W, 1)
            else:
                mask = masks[i]  # .reshape(H, W, 1)
            if mask.ndim == 2:
                mask = mask.reshape(H, W, 1)
            bg_rgb = rgb * (1 - mask)
            bg_gt = gt_imgs[i] * (1 - mask)
        else:
            mask, bg_rgb, bg_gt = np.ones(rgb.shape[:2]), np.ones(rgb.shape), np.ones(rgb.shape)

        if i == 0:
            print('Testing {} {}'.format(rgb.shape, disp.shape))
        if gt_imgs is not None and render_factor == 0:
            p = -10. * np.log10(np.mean(np.square(rgb - gt_imgs[i])))
            back_p, fore_p = 0., 0.
            if masks is not None:
                back_p = -10. * np.log10(np.sum(np.square(bg_rgb - bg_gt)) / np.sum(1 - mask))
                fore_p = -10. * np.log10(np.sum(np.square(rgb * mask - gt_imgs[i] * mask)) / np.sum(mask))
            error = 1 - np.exp(-20 * np.square(rgb - gt_imgs[i]).sum(-1))[..., None].repeat(3, -1)
            logging.info(
                "{} | full-image psnr {:.2f} | foreground psnr {:.2f} | background psnr: {:.2f} ".format(i, p,
                                                                                                         fore_p,
                                                                                                         back_p))
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
            filename = os.path.join(savedir, step_pre + '{:03d}.png'.format(id))
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
                normal = (rot @ normals[-1][..., None])[..., 0]
                normal = 0.5 - 0.5 * normal
                if masks is not None:
                    normal = normal * mask.mean(-1)[..., None] + (1 - mask)
                normal8 = utils.to8b(normal)
                step_pre = str(step) + '_' if step > 0 else ''
                filename = os.path.join(savedir, step_pre + '{:03d}_normal.png'.format(id))
                imageio.imwrite(filename, normal8)

    rgbs = np.array(rgbs)
    disps = np.array(disps)
    if len(psnrs):
        print('Testing psnr {:.2f} (avg) | foreground {:.2f} | background {:.2f}'.format(
            np.mean(psnrs), np.mean(fore_psnrs), np.mean(bg_psnrs)))
        if eval_ssim: print('Testing ssim {} (avg)'.format(np.mean(ssims)))
        if eval_lpips_vgg: print('Testing lpips (vgg) {} (avg)'.format(np.mean(lpips_vgg)))
        if eval_lpips_alex: print('Testing lpips (alex) {} (avg)'.format(np.mean(lpips_alex)))

    return rgbs, disps

@torch.no_grad()
def visualize_object_image(model,images_gt, render_poses, cfg,intr,HW, global_step,
                        opt,ndc,render_kwargs,id=None):
    if id is None:
        rand_idx = random.randint(0, len(render_poses) - 1)
    else:
        rand_idx = id
    render_poses = render_poses[rand_idx][None]
    intr = intr[rand_idx][None]
    HW = HW[rand_idx][None]
    image_gt = images_gt[rand_idx][None].cpu()
    rgbs, disps = render_viewpoints(model, render_poses, cfg, HW, intr, ndc, render_kwargs)
    rgbs = torch.tensor(rgbs).cpu()
    disps = torch.tensor(disps).cpu()
    disps = disps.repeat(1, 1, 1, 3).cpu()
    stack_image = torch.cat([image_gt[0], rgbs[0], disps[0]],dim=1)  # ( H, W,3)
    loss_mse_render = F.mse_loss(image_gt[0], rgbs)
    psnr = utils.mse2psnr(loss_mse_render.detach()).item()
    return stack_image, psnr

@torch.no_grad()
def visualize_test_image(model,model_bg, images_gt, render_poses,cfg, intr,HW, global_step,
                        opt,ndc,render_kwargs,id=None):
    print('image id {}'.format(id))
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
        rgbs, disps = render_viewpoints(model, render_poses,cfg, HW,intr, ndc,render_kwargs)
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

    print('Testing psnr {} ssim {} lpips (alex) {} lpips (vgg) {}'.format(psnr,ssims,lpips_alex, lpips_vgg))
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
def visualize_val_image(model, model_bg, images_gt, render_poses,cfg, intr, HW, global_step,
                        opt, ndc, render_kwargs, id=None, render_only=False):
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
            rgbs, disps = render_viewpoints(model, render_poses,cfg, HW, intr, ndc, render_kwargs)
            rgbs = torch.tensor(rgbs).cpu()
            disps = torch.tensor(disps).cpu()
            disps = disps.repeat(1, 1, 1, 3).cpu()
    if model_bg is None or opt is None:
        depth_bg, fine_rgb, coarse_rgb = torch.zeros_like(image_gt[0]), torch.zeros_like(
            image_gt[0]), torch.zeros_like(image_gt[0])
    else:
        rays, _ = model_bg.forward_graph(opt, render_poses, intr=intr, mode='val', global_step=global_step)
        stack = model_bg.validation_step(batch=[rays, image_gt], global_step=global_step)  # (4, 3, H, W)
        coarse_rgb = einops.rearrange(stack[1, ...], 'b h w -> h w b')
        fine_rgb = einops.rearrange(stack[2, ...], 'b h w -> h w b')

        depth_bg = visualize_depth(stack[3, 0, ...])
        depth_bg = torch.tensor(einops.rearrange(depth_bg, 'b h w -> h w b'), device=fine_rgb.device)
    if render_only:
        return fine_rgb, depth_bg
    else:
        stack_image = torch.cat([image_gt[0], coarse_rgb, fine_rgb, depth_bg, rgbs[0], disps[0]],
                                dim=1)  # ( H, W,3)
        loss_mse_render = F.mse_loss(image_gt[0], fine_rgb)
        psnr = utils.mse2psnr(loss_mse_render.detach()).item()
        return stack_image, psnr












