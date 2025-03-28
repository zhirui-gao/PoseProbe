import numpy as np
import os

from .load_blender import load_blender_data
from .load_scene_with_shapenet import load_scene_with_shapenet_data
from .load_toy import load_toy_data
from .load_custom import load_custom_data
from .load_dtu import load_dtu_data
from .load_replica import ReplicaPerScene

def load_data(args, reso_level=2, train_all=True, wmask=True, white_bg=True):
    print("[ resolution level {} | train all {} | wmask {} | white_bg {}]".format(reso_level, train_all, wmask, white_bg))
    K, depths = None, None
    scale_mats_np = None
    masks = None


    if args.dataset_type == 'blender':
        images, images_gray, poses, render_poses, hwf, i_split,matcher_infos,sg_config = load_blender_data(args.datadir, args.half_res, args.testskip, args.trainskip, args.matching)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near, far = 2., 6.

        if images.shape[-1] == 4:
            masks = images[...,3:]
            if args.white_bkgd:
                images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
            else:
                images = images[...,:3]*images[...,-1:]
    elif args.dataset_type == 'scene_with_shapenet':
        images, depths, masks, images_gray, poses, render_poses, hwf, i_split, matcher_infos, sg_config,images_object,align_pose =\
            load_scene_with_shapenet_data(
            args.datadir, reso_level, args.testskip, args.trainskip, args.matching,args.selected_id)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split
        near, far = 0.5, 6.
        if images.shape[-1] == 4:
            if args.white_bkgd:
                images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
            else:
                images = images[..., :3] * images[..., -1:]



    elif args.dataset_type == 'toy':
        images, masks, images_gray, poses, render_poses, hwf, i_split, matcher_infos, sg_config,images_object,align_pose = load_toy_data(
            args.datadir, reso_level, args.matching,args.selected_id, args.test_id, args.inst_seg_tag)
        print('Loaded toy', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split
        # near, far = 0.1, 0.8
        near, far = args.near, args.far

    elif args.dataset_type == 'replica':
        replica = ReplicaPerScene(args, 'train', os.path.basename(args.datadir))

        images, masks, images_gray, poses, render_poses, hwf, K,i_split, matcher_infos, sg_config, images_object, align_pose \
             = replica.load_replica_data(args.matching, args.selected_id)
        print('Loaded toy', images.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split
        # near, far = 0.1, 0.8
        near, far = replica.near, replica.far


    elif args.dataset_type =='custom':
        images, masks, images_gray, poses, render_poses, hwf, i_split, matcher_infos, sg_config, images_object, align_pose = load_custom_data(
            args.datadir, reso_level, args.testskip, args.trainskip, args.matching, args.selected_id)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split
        near, far = 0.3, 4  # scene_customed    for 0.3 3 for cole
        if images.shape[-1] == 4:
            if args.white_bkgd:
                images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
            else:
                images = images[..., :3] * images[..., -1:]


    elif args.dataset_type == 'dtu':
        images,images_gray, poses, render_poses, hwf, K, i_split, scale_mats_np, masks,\
            matcher_infos, sg_config, images_object, align_pose= load_dtu_data(args.datadir, reso_level=reso_level, mask=wmask,
                                                                    white_bg=white_bg, matching_config=args.matching,
                                                                    selected_id=args.selected_id)
        print('Loaded dtu', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split
        near, far = args.near, args.far  #inward_nearfar_heuristic(poses[i_train, :3, 3])
        assert images.shape[-1] == 3

    else:
        raise NotImplementedError(f'Unknown dataset type {args.dataset_type} exiting')

    near, far = near * (1 - 0.2), far * (1 + 0.2)
    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]
    HW = np.array([im.shape[:2] for im in images])
    irregular_shape = (images.dtype is np.dtype('object'))

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    if len(K.shape) == 2:
        Ks = K[None].repeat(len(poses), axis=0)
    else:
        Ks = K

    render_poses = render_poses[...,:4]
    print("Split: train {} | validate {} | test {}".format(
        len(i_train), len(i_val), len(i_test)))
    print('near, far: ', near, far)
    if wmask and masks is None:
        masks = images.mean(-1) > 0



    data_dict = dict(
        hwf=hwf, HW=HW, Ks=Ks, near=near, far=far,
        i_train=i_train, i_val=i_val, i_test=i_test,
        poses=poses, render_poses=render_poses,
        images=images, depths=depths, images_gray=images_gray,
        irregular_shape=irregular_shape,masks=masks,
        scale_mats_np=scale_mats_np, sg_config=sg_config,
        matcher_infos=matcher_infos, images_object=images_object,align_pose=align_pose
    )
    return data_dict


def inward_nearfar_heuristic(cam_o, ratio=0.05):
    dist = np.linalg.norm(cam_o[:,None] - cam_o, axis=-1)
    far = dist.max()
    near = far * ratio
    return near, far

