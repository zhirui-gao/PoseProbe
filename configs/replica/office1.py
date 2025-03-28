_base_ = '../default_fine_s.py'


basedir = '/data/gaozhirui/pose_probe_deform/replica'
evaldir = '/data/gaozhirui/pose_probe_deform/replica'
train_all = True
reso_level = 1
exp_stage = 'coarse'


use_sp_color = True


expname = 'office1'
data = dict(
    datadir='/data/gaozhirui/datasets/Replica/',
    dataset_type='replica',
    selected_id = [200,300,400],
    inverse_y=True,
    white_bkgd= False,
    matching=dict(
        nms_radius=2,
        keypoint_threshold=0.00, # 0.002
        max_keypoints=512,
        max_matcher=256,
        superglue='outdoor', # cole； outdoor
        sinkhorn_iterations=20,
        match_threshold=0.2,
        device='cuda',
        use_kornia=False, # loftr
    ),
    testskip = 1,  # subsample testset to preview results
    trainskip = 10 ,
    xyz_min=[-0.18, 0., 0.25],  # objet size
    xyz_max=[0., 0.16, 0.4],

)

camera = dict(
    noise=0.,
    barf_c2f=[0.6, 1],
    pc_ratio=4,
    incremental=True,
    incremental_step=2000,
    optimize_sp=True,
)
pnp = dict(
    use_pnp=False,
    ransac=False,
)

surf_train=dict(
    world_bound_scale=1.25,
    load_density_from='',
    pg_filter=[1000,],
    tv_add_grad_new=True,
    weight_surface_projection=0.,  # 0.001
    projection_dis_error=0.001,
    weight_near_surface=0.1,
    N_iters= 1, # 3:12k  6:15k 9: 20k view
    N_iters_bg= 44000,  # 3:70K   6: 140k  9: 210 k
    lrate_decay=10,  # lr decay by 0.1 after every lrate_decay*1000 steps
    weight_tv_k0= 0, #0.01, # 0.01
    weight_tv_density=0.005, # 0.001
    weight_sdf_delta=1,
    tv_terms=dict(
        sdf_tv=0.1,
        grad_tv=0,
        smooth_grad_tv=0.05,
    ),
    tv_updates={
        4000:dict(
            sdf_tv=0.1,
            # grad_tv=10,
            smooth_grad_tv=0.2
        ),
    },
    tv_dense_before=20000,

    lr_pose= 0.,
    lr_pose_end= 0.,
    sched_pose='ExponentialLR',

    lrate_sdf=0.1,
    lrate_sdf_delta=1e-3,
    lrate_sdf_alpha=1e-2,
    lrate_sdf_beta=1e-2,
    lrate_point_deform = 2e-5,

    decay_step_module={
        5000:dict(sdf=0.1,sdf_beta=0.1,sdf_alpha=0.1), # 1000 ,5000
        10000:dict(sdf=0.5,sdf_beta=0.5,sdf_alpha=0.5),
    },

    lrate_k0=1e-1, #1e-1,                # lr of color/feature voxel grid
    lrate_rgbnet=1e-3, # 1e-3,           # lr of the mlp to preduct view-dependent color
    lrate_sdf_delta_conv=1e-3,
    lrate_k_rgbnet=1e-3, # 1e-3,           # lr of the mlp to preduct view-dependent color
    lrate_rgb_addnet=1e-3, # 1e-3,
    lrate_warp_network=1e-3,
    matcher_lrate_sg_matching=1e-4,
)

surf_model_and_render=dict(
    optimize_sdf=False,
    load_sdf=True,
    num_voxels= 96**3, # 84**3, #
    num_voxels_base=96**3, #
    rgbnet_full_implicit=False, # by using a full mlp without local feature for rgb, the info for the geometry would be better
    posbase_pe=5,
    viewbase_pe=1, # 4
    rgb_add_res=True,
    rgbnet_depth=4,
    geo_rgb_dim=3,
    smooth_ksize=0, # 5
    smooth_sigma=0.8,
    s_ratio=50,
    s_start=0.2,
)
