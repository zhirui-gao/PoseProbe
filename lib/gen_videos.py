import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from tqdm import tqdm
import imageio
import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend('agg')
import torch
from lib import camera


def pad_poses(p: np.ndarray):
    """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
    bottom = np.broadcast_to([0, 0, 0, 1.], p[Ellipsis, :1, :4].shape)
    return np.concatenate([p[Ellipsis, :3, :4], bottom], axis=-2)


def unpad_poses(p: np.ndarray):
    """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
    return p[Ellipsis, :3, :4]


def recenter_poses(poses: np.ndarray):
    """Recenter poses around the origin."""
    cam2world = poses_avg(poses)
    poses = np.linalg.inv(pad_poses(cam2world)) @ pad_poses(poses)
    return unpad_poses(poses)


def shift_origins(origins: np.ndarray, directions: np.ndarray, near=0.0):
    """Shift ray origins to near plane, such that oz = near."""
    t = (near - origins[Ellipsis, 2]) / directions[Ellipsis, 2]
    origins = origins + t[Ellipsis, None] * directions
    return origins


def poses_avg(poses: np.ndarray):
    """New pose using average position, z-axis, and up vector of input poses."""
    position = poses[:, :3, 3].mean(0)
    z_axis = poses[:, :3, 2].mean(0)
    up = poses[:, :3, 1].mean(0)
    cam2world = viewmatrix(z_axis, up, position)
    return cam2world


def focus_pt_fn(poses: np.ndarray):
    """Calculate nearest point to all focal axes in poses."""
    directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
    m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
    mt_m = np.transpose(m, [0, 2, 1]) @ m
    focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
    return focus_pt


def normalize(x: np.ndarray):
    """Normalization helper function."""
    return x / np.linalg.norm(x)


def viewmatrix(lookdir: np.ndarray, up: np.ndarray, position: np.ndarray,
               subtract_position=False):
    """Construct lookat view matrix."""
    vec2 = normalize((lookdir - position) if subtract_position else lookdir)
    vec0 = normalize(np.cross(up, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, position], axis=1)
    return m


def generate_spiral_path(poses_c2w, bounds,
                         n_frames=240, n_rots=2, zrate=.5):
    """Calculates a forward facing spiral path for rendering.
    poses are in opencv format. """
    is_torch = False
    if isinstance(poses_c2w, torch.Tensor):
        is_torch = True
        poses_c2w = poses_c2w.detach().cpu().numpy()

    # Find a reasonable 'focus depth' for this dataset as a weighted average
    # of near and far bounds in disparity space.
    close_depth, inf_depth = bounds.min() * .9, bounds.max() * 5.
    dt = .75
    focal = 1 / (((1 - dt) / close_depth + dt / inf_depth))

    # Get radii for spiral path using 90th percentile of camera positions.
    positions = poses_c2w[:, :3, 3]
    radii = np.percentile(np.abs(positions), 90, 0)
    radii = np.concatenate([radii, [1.]])

    # Generate poses for spiral path.
    render_poses = []
    cam2world = poses_avg(poses_c2w)
    up = poses_c2w[:, :3, 1].mean(0)
    for theta in np.linspace(0., 2. * np.pi * n_rots, n_frames, endpoint=False):
        t = radii * [np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]
        position = cam2world @ t
        lookat = cam2world @ [0, 0, -focal, 1.]
        z_axis = position - lookat
        render_poses.append(viewmatrix(z_axis, up, position))
    render_poses = np.stack(render_poses, axis=0)
    if is_torch:
        render_poses = torch.from_numpy(render_poses)[:, :3]
    return render_poses


def generate_spiral_path_dtu(poses_c2w,
                             n_frames=240, n_rots=2, zrate=.5, perc=60):
    """Calculates a forward facing spiral path for rendering for DTU.
    poses are in opencv format. """
    is_torch = False
    if isinstance(poses_c2w, torch.Tensor):
        is_torch = True
        poses_c2w = poses_c2w.detach().cpu().numpy()
    # Get radii for spiral path using 60th percentile of camera positions.
    positions = poses_c2w[:, :3, 3]
    radii = np.percentile(np.abs(positions), perc, 0)
    radii = np.concatenate([radii, [1.]])

    # Generate poses for spiral path.
    render_poses = []
    cam2world = poses_avg(poses_c2w)
    up = poses_c2w[:, :3, 1].mean(0)
    z_axis = focus_pt_fn(poses_c2w)
    for theta in np.linspace(0., 2. * np.pi * n_rots, n_frames, endpoint=False):
        t = radii * [np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]
        position = cam2world @ t
        render_poses.append(viewmatrix(z_axis, up, position, True))
    render_poses = np.stack(render_poses, axis=0)
    if is_torch:
        render_poses = torch.from_numpy(render_poses)[:, :3]
    return render_poses


@torch.no_grad()
def generate_videos_synthesis(model, pose_pred, intr, opt, cfg, output_path, global_step, eps=1e-10):
    """
    Will generate a video by sampling poses and rendering the corresponding images.
    """
    opt.output_path = output_path
    model.eval()
    poses = pose_pred  # pose_GT
    poses_c2w = camera.pose.invert(poses)
    focal = intr[0][0, 0]
    depth_range = opt.nerf.depth.range

    # alternative 1
    if 'shapenet1' in cfg.data.dataset_type:
        # render the novel views
        novel_path = "{}/novel_view".format(opt.output_path)
        os.makedirs(novel_path, exist_ok=True)

        pose_novel_c2w = generate_spiral_path(poses_c2w, bounds=np.array(depth_range),
                                              n_frames=60).to(opt.device)

        pose_novel = camera.pose.invert(pose_novel_c2w)

    elif 'dtu' in cfg.data.dataset_type:
        # dtu
        n_frame = 60
        novel_path = "{}/novel_view".format(opt.output_path)
        os.makedirs(novel_path, exist_ok=True)

        pose_novel_c2w = generate_spiral_path_dtu(poses_c2w, n_rots=1, n_frames=n_frame).to(opt.device)
        pose_novel_c2w = pose_novel_c2w
        pose_novel_1 = camera.pose.invert(pose_novel_c2w)

        # rotate novel views around the "center" camera of all poses
        scale = 1
        test_poses_w2c = poses
        idx_center = (test_poses_w2c - test_poses_w2c.mean(dim=0, keepdim=True))[..., 3].norm(dim=-1).argmin()
        pose_novel_2 = camera.get_novel_view_poses(opt, test_poses_w2c[idx_center], N=n_frame, scale=scale).to(
            opt.device)

        pose_novel = torch.cat((pose_novel_1, pose_novel_2), dim=0)
        # render the novel views
    else:
        n_frame = 60
        novel_path = "{}/novel_view".format(opt.output_path)
        os.makedirs(novel_path, exist_ok=True)

        scale = 0.5  # 0.1
        test_poses_w2c = poses  # .detach().cpu()
        idx_center = (test_poses_w2c - test_poses_w2c.mean(dim=0, keepdim=True))[..., 3].norm(dim=-1).argmin()
        pose_novel = camera.get_novel_view_poses(opt, test_poses_w2c[idx_center], N=n_frame, scale=scale).to(
            opt.device)
    pose_novel_tqdm = tqdm(pose_novel, desc="rendering novel views", leave=False)

    for i, pose_aligned in enumerate(pose_novel_tqdm):
        ret, _ = model.forward_graph(opt, pose_aligned, intr=intr, mode='val', global_step=global_step)
        rgb, depth, opacity = model.visualize(opt, ret, split="val", append_cbar=False)  # (4, 3, H, W)
        filename = os.path.join(novel_path, 'rgb_{}.png'.format(i))
        imageio.imwrite(filename, np.uint8(rgb * 255))
        filename = os.path.join(novel_path, 'depth_{}.png'.format(i))
        imageio.imwrite(filename, np.uint8(depth * 255))

    # write videos
    print("writing videos...")

    rgb_vid_fname = "{}/novel_view_{}_rgb".format(opt.output_path, cfg.expname)
    depth_vid_fname = "{}/novel_view_{}_depth".format(opt.output_path, cfg.expname)
    os.system("ffmpeg -y -framerate 10 -i {0}/rgb_%d.png -pix_fmt yuv420p {1}.mp4 >/dev/null 2>&1".format(novel_path,
                                                                                                          rgb_vid_fname))
    os.system("ffmpeg -y -framerate 10 -i {0}/depth_%d.png -pix_fmt yuv420p {1}.mp4 >/dev/null 2>&1".format(novel_path,
                                                                                                            depth_vid_fname))

    os.system("ffmpeg -f image2 -framerate 10 -i {0}/rgb_%d.png -loop -1 {1}.gif >/dev/null 2>&1".format(novel_path,
                                                                                                         rgb_vid_fname))
    os.system("ffmpeg -f image2 -framerate 10 -i {0}/depth_%d.png -loop -1 {1}.gif >/dev/null 2>&1".format(novel_path,
                                                                                                           depth_vid_fname))
    print('Finished creating video')
    return
