from copy import deepcopy

expname = None                    # experiment name
basedir = './logs/'               # where to store ckpts and logs

''' Template of data options
'''
data = dict(
    datadir=None,                 # path to dataset root folder
    dataset_type=None,            # blender | nsvf | blendedmvs | tankstemple | deepvoxels | co3d
    inverse_y=False,              # intrinsict mode (to support blendedmvs, nsvf, tankstemple)
    flip_x=False,                 # to support co3d
    flip_y=False,                 # to support co3d
    annot_path='',                # to support co3d
    split_path='',                # to support co3d
    sequence_name='',             # to support co3d
    load2gpu_on_the_fly=False,    # do not load all images into gpu (to save gpu memory)
    testskip=1,                   # subsample testset to preview results
    trainskip=5,                  # subsample trainset to preview results
    white_bkgd=False,             # use white background (note that some dataset don't provide alpha and with blended bg color)
    # Below are forward-facing llff specific settings. Not support yet.
    ndc=False,                    # use ndc coordinate (only for forward-facing; not support yet)
    spherify=False,               # inward-facing
    llffhold=8,                   # testsplit
    load_depths=False,            # load depth
    movie_render_kwargs=dict(),
)

''' Template of training options
'''
coarse_train = dict(
    N_iters=5000,                 # number of optimization steps
    lrate_density=1e-1,           # lr of density voxel grid
    lrate_k0=1e-1,                # lr of color/feature voxel grid
    lrate_rgbnet=1e-3,            # lr of the mlp to preduct view-dependent color
    lrate_decay=20,               # lr decay by 0.1 after every lrate_decay*1000 steps
    pervoxel_lr=True,             # view-count-based lr
    pervoxel_lr_downrate=1,       # downsampled image for computing view-count-based lr
    ray_sampler='random',         # ray sampling strategies
    weight_main=1.0,              # weight of photometric loss
    weight_mask=0.1,
    weight_mip=1.0,
    weight_consistency_nerf= 0.1, #
    weight_depth_consistency=0,
    weight_eikonal_loss=0.1,
    weight_warp_loss=0,
    weight_surface_projection= 1, # 0.1
    projection_dis_error=1, # 1.0
    weight_near_surface=10., # 10
    space_dis_error=0.,
    weight_entropy_last=0.01,     # weight of background entropy loss
    weight_rgbper=0.1,            # weight of per-point rgb loss
    tv_every=1e5,                   # count total variation loss every tv_every step
    tv_from=0,                    # count total variation loss from tv_from step
    tv_end=20000,                 # count total variation loss from tv_from step
    weight_tv_density=0.0,        # weight of total variation loss of density voxel grid
    weight_tv_k0=0.0,             # weight of total variation loss of color/feature voxel grid
    pg_scale=[],                  # checkpoints for progressive scaling
    save_iter=10000
)

fine_train = deepcopy(coarse_train)
fine_train.update(dict(
    N_iters=5000,
    N_rand=1024,
    pervoxel_lr=False,
    ray_sampler='flatten', # in_maskcache
    weight_entropy_last=0.001,
    weight_rgbper=0.01,
    pg_scale=[1000, 2000, 3000],
    # pg_scale = []
))

surf_train = deepcopy(fine_train)
surf_train.update(dict(
    weight_rgbper=0.0,            # weight of per-point rgb loss
    lrate_sdf=2e-3,           # lr of sdf voxel grid
    pg_scale = [],
    weight_tv_density=0.001,
    tv_terms=dict(
        sdf_tv=1,
        grad_norm=0,
        grad_tv=0
    ),
    lrate_sdfnet=1e-3,
    # weight_diffnorm=1,
))
''' Template of model and rendering options
'''
coarse_model_and_render = dict(
    num_voxels=1024000,           # expected number of voxel
    num_voxels_base=1024000,      # to rescale delta distance
    nearest=False,                # nearest interpolation
    pre_act_density=False,        # pre-activated trilinear interpolation
    in_act_density=False,         # in-activated trilinear interpolation
    bbox_thres=1e-3,              # threshold to determine known free-space in the fine stage
    mask_cache_thres=1e-3,        # threshold to determine a tighten BBox in the fine stage
    rgbnet_dim=0,                 # feature voxel grid dim
    rgbnet_full_implicit=False,   # let the colors MLP ignore feature voxel grid
    rgbnet_direct=True,           # set to False to treat the first 3 dim of feature voxel grid as diffuse rgb
    rgbnet_depth=3,               # depth of the colors MLP (there are rgbnet_depth-1 intermediate features)
    rgbnet_width=128,             # width of the colors MLP
    alpha_init=1e-6,              # set the alpha values everywhere at the begin of training
    fast_color_thres=0,           # threshold of alpha value to skip the fine stage sampled point
    maskout_near_cam_vox=True,    # maskout grid points that between cameras and their near planes
    world_bound_scale=1,          # rescale the BBox enclosing the scene
    stepsize=1.5, # 0.5                 # sampling stepsize in volume rendering

    sdfnet_dim=0,
)

fine_model_and_render = deepcopy(coarse_model_and_render)
fine_model_and_render.update(dict(
    num_voxels=160**3,
    num_voxels_base=160**3,
    rgbnet_dim=12,
    alpha_init=1e-2,
    fast_color_thres=0,
    maskout_near_cam_vox=False,
    world_bound_scale=1.0,  # 1.05
))

surf_model_and_render = deepcopy(fine_model_and_render)
surf_model_and_render.update(dict(
    geo_rgb_dim=3,
    sdfnet_dim=12,     #12            # feature voxel grid dim
    sdfnet_depth=3,               # depth of the colors MLP (there are rgbnet_depth-1 intermediate features)
    sdfnet_width=128,             # width of the colors MLP
    sdf_refine=True,
    alpha_refine=True,
    displace_step=0.1,

    rgbnet_dim=12,
    rgbnet_direct=True,
    # surface_sampling=True,
    # n_importance=128,
    # up_sample_steps=4,
    #
    rgbnet_full_implicit=False,
    # s_ratio=1000,
    # s_start=0.5,
    # stepsize=2
))

del deepcopy
