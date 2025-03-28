import os
import time

import einops
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import segment_coo
from torch.utils.cpp_extension import load
from einops import rearrange
from . import grid
from lib import camera
from lib.dvgo_ori import extract_geometry
from external.SuperGlue.models.matching import Matching
from lib.warp import SE3Field
from lib.deformation.deform_net import DeformedImplicitField
parent_dir = os.path.dirname(os.path.abspath(__file__))
render_utils_cuda = load(
    name='render_utils_cuda',
    sources=[
        os.path.join(parent_dir, path)
        for path in ['cuda/render_utils.cpp', 'cuda/render_utils_kernel.cu']],
    verbose=True)


class pose_model(torch.nn.Module):
    def __init__(self, i_train=[],camera_noise=0.05):
        super(pose_model, self).__init__()
        self.i_train = i_train
        self.camera_noise = camera_noise
        self.se3_refine = torch.nn.Parameter(torch.zeros((len(self.i_train), 6), dtype=torch.float32),
                                             requires_grad=True)
        self.se3_align_refine = torch.nn.Parameter(torch.zeros((1, 6), dtype=torch.float32),
                                                   requires_grad=True)
        # self.se3_refine.requires_grad = False
        # self.se3_align_refine.requires_grad = False
        se3_noise = torch.randn(len(i_train), 6) * self.camera_noise  # * 0.
        self.pose_noise = camera.lie.se3_to_SE3(se3_noise)


'''Model'''
class Voxurf(torch.nn.Module):
    """
    This module is modified from DirectVoxGO https://github.com/sunset1995/DirectVoxGO/blob/main/lib/dvgo.py
    """
    def __init__(self, xyz_min, xyz_max,
                 num_voxels=0, num_voxels_base=0,
                 alpha_init=None,
                 nearest=False,
                 mask_cache_path=None, mask_cache_thres=1e-3,
                 fast_color_thres=0,
                 rgbnet_dim=0, rgbnet_direct=False, rgbnet_full_implicit=False,
                 rgbnet_depth=3, rgbnet_width=128,
                 posbase_pe=5, viewbase_pe=4, geo_rgb_dim=3,
                 grad_mode='interpolate',
                 s_ratio=2000, s_start=0.2, s_learn=False, step_start=0,
                 smooth_ksize=0, smooth_sigma=1, camera_noise=0., barf_c2f=None,i_train=[],
                 i_test = [], N_iters=20000, flow_ckpt_path='',flow_backbone='', HW=[[512,512]],sg_config=None,
                 optimize_sdf=False,range_shape=None, rect_size=None,
                 **kwargs):
        super(Voxurf, self).__init__()
        self.i_train = i_train
        self.N_iters = N_iters
        self.camera_noise = camera_noise
        self.barf_c2f = barf_c2f
        self.use_explict_deform = False
        self.use_implict_deform = True
        self.start_deform_net = -1
        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))
        self.device = self.xyz_min.device
        self.flow_ckpt_path = flow_ckpt_path
        self.flow_backbone = flow_backbone
        self.HW = HW
        self.activation = nn.Softplus(beta=10)
        self.range_shape = range_shape
        self.rect_size = rect_size
        self.sg_config = sg_config
        H, W = HW[0,0], HW[0,1]
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        self.grid = torch.stack((xx, yy), dim=-1).to(self.device).float()
        self.grid_flat = self.grid[:, :, 1] * W + self.grid[:, :,0]  # (H, W), corresponds to index in flattedned array (in H*W)
        self.grid_flat = self.grid_flat.to(self.device).long()
        self.warp_network = DeformedImplicitField(range_shape = self.range_shape)  # 修改为输出一个xyz的offset + correction（DIFNet）

        self.fast_color_thres = fast_color_thres
        self.nearest = nearest

        self.s_ratio = s_ratio
        self.s_start = s_start
        self.s_learn = s_learn
        self.step_start = step_start
        self.s_val = nn.Parameter(torch.ones(1), requires_grad=s_learn).cuda()

        self.s_val.data *= s_start
        self.sdf_init_mode = 'cube_init' # ball_init
        self.optimize_sdf = optimize_sdf
        # determine based grid resolution
        self.num_voxels_base = num_voxels_base
        self.voxel_size_base = ((self.xyz_max - self.xyz_min).prod() / self.num_voxels_base).pow(1/3)
        self.diagonal_length = torch.sqrt(torch.sum(self.xyz_max - self.xyz_min ** 2))
        # mapping the bbox to a [-1,1] space
        self.bool_mapping_deformspace = False
        self.mapping_sacle = (2. / (self.xyz_max - self.xyz_min).max())


        # determine the density bias shift
        self.alpha_init = alpha_init
        self.act_shift = np.log(1/(1-alpha_init) - 1)
        print('set density bias shift to', self.act_shift)

        # determine init grid resolution
        self._set_grid_resolution(num_voxels)
        # use Parameter so it could be checkpointed
        self.progress = torch.nn.Parameter(torch.tensor(0.))
        # sdf scale shift
        self.use_mapping_fun = True

        # self.sdf_alpha = torch.nn.Parameter(torch.Tensor([3.]), requires_grad=True) # 3.0
        # self.sdf_beta = torch.nn.Parameter(torch.Tensor([-0.5]), requires_grad=True) # -0.5

        self.sdf_alpha = torch.nn.Parameter(torch.Tensor([10.0]), requires_grad=True)
        self.sdf_beta = torch.nn.Parameter(torch.Tensor([2.0]), requires_grad=True)


        self.sdf = grid.create_grid(
            'DenseGrid', channels=1, world_size=self.world_size,
            xyz_min=self.xyz_min, xyz_max=self.xyz_max)

        if self.use_explict_deform:
            self.point_deform = grid.create_grid(
                'DenseGrid', channels=3, world_size=self.world_size,
                xyz_min=self.xyz_min, xyz_max=self.xyz_max)
            self.grid_xyz = torch.stack(torch.meshgrid(
                torch.linspace(self.xyz_min[0], self.xyz_max[0], self.sdf.grid.shape[2]),
                torch.linspace(self.xyz_min[1], self.xyz_max[1], self.sdf.grid.shape[3]),
                torch.linspace(self.xyz_min[2], self.xyz_max[2], self.sdf.grid.shape[4]),
            ), -1)
            for param in self.sdf.parameters():
                param.requires_grad = False
        elif self.use_implict_deform:
            for param in self.sdf.parameters():
                param.requires_grad = False
        else:
            print("No DeFormation Network")

        if self.sdf_init_mode == "ball_init":
            x, y, z = np.mgrid[-1.0:1.0:self.world_size[0].item() * 1j, -1.0:1.0:self.world_size[1].item() * 1j,
                      -1.0:1.0:self.world_size[2].item() * 1j]
            self.sdf.grid.data = torch.from_numpy((x ** 2 + y ** 2 + z ** 2) ** 0.5 - 0.3).float()[None, None, ...].to(self.device)
        elif self.sdf_init_mode == "random":
            self.sdf.grid = torch.nn.Parameter(torch.rand([1, 1, *self.world_size]) * 0.05) # random initialization
            torch.nn.init.normal_(self.sdf.grid, 0.0, 0.5)
        elif self.sdf_init_mode == "cube_init":
            x, y, z = np.mgrid[self.xyz_min[0].item():self.xyz_max[0].item():self.world_size[0].item() * 1j,
                      self.xyz_min[1].item():self.xyz_max[1].item():self.world_size[1].item() * 1j,
                      self.xyz_min[2].item():self.xyz_max[2].item():self.world_size[2].item() * 1j]


            center_point = ((self.xyz_max+self.xyz_min)/2).tolist()
            dist_x = np.minimum(np.abs(x-(rect_size[0]/2-center_point[0])), np.abs(x-(rect_size[0]/2+center_point[0])))
            dist_y = np.minimum(np.abs(y-(rect_size[1]/2-center_point[1])), np.abs(y-(rect_size[1]/2+center_point[1])))
            dist_z = np.minimum(np.abs(z-(rect_size[2]/2-center_point[2])), np.abs(z-(rect_size[2]/2+center_point[2])))
            sdf = torch.from_numpy((dist_x ** 2 + dist_y ** 2 + dist_z ** 2) ** 0.5)
            insize_rect = (x>=(center_point[0]-rect_size[0]/2))&(x<=(center_point[0]+rect_size[0]/2)) &\
                          (y>=(center_point[1]-rect_size[1]/2))&(y<=(center_point[1]+rect_size[1]/2)) &\
                          (z>=(center_point[2]-rect_size[2]/2))&(z<=(center_point[2]+rect_size[2]/2))
            sdf[insize_rect] *= -1
            self.sdf.grid.data = sdf.float()[None, None, ...].to(self.device)

        else:
            raise NotImplementedError
        self.init_smooth_conv(smooth_ksize, smooth_sigma)

        # init color representation
        self.rgbnet_kwargs = {
            'rgbnet_dim': rgbnet_dim, 'rgbnet_direct': rgbnet_direct,
            'rgbnet_full_implicit': rgbnet_full_implicit,
            'rgbnet_depth': rgbnet_depth, 'rgbnet_width': rgbnet_width,
            'posbase_pe': posbase_pe, 'viewbase_pe': viewbase_pe,
        }
        self.rgbnet_full_implicit = rgbnet_full_implicit
        if rgbnet_dim <= 0:
            # color voxel grid  (dvgo coarse stage)
            self.k0_dim = 3
            self.k0 = torch.nn.Parameter(torch.zeros([1, self.k0_dim, *self.world_size]))
            self.rgbnet = None
        else:
            if self.rgbnet_full_implicit:
                self.k0_dim = 0
            else:
                self.k0_dim = rgbnet_dim
            self.k0 = grid.create_grid(
                'DenseGrid', channels=self.k0_dim, world_size=self.world_size,
                xyz_min=self.xyz_min, xyz_max=self.xyz_max)
            self.rgbnet_direct = rgbnet_direct
            self.register_buffer('posfreq', torch.FloatTensor([(2**i) for i in range(posbase_pe)]))
            self.register_buffer('viewfreq', torch.FloatTensor([(2**i) for i in range(viewbase_pe)]))
            dim0 = (3+3*posbase_pe*2) + (3+3*viewbase_pe*2)
            if rgbnet_direct:
                dim0 += self.k0_dim
            else:
                dim0 += self.k0_dim-3
            self.geo_rgb_dim = geo_rgb_dim
            if self.geo_rgb_dim:
                dim0 += self.geo_rgb_dim
            self.rgbnet = nn.Sequential(
                nn.Linear(dim0, rgbnet_width), nn.ReLU(inplace=True),
                *[
                    nn.Sequential(nn.Linear(rgbnet_width, rgbnet_width), nn.ReLU(inplace=True))
                    for _ in range(rgbnet_depth-2)
                ],
                nn.Linear(rgbnet_width, 3),
            )
            nn.init.constant_(self.rgbnet[-1].bias, 0)
            print('feature voxel grid', self.k0.grid.shape)
            print('mlp', self.rgbnet)
        # Using the coarse geometry if provided (used to determine known free space and unknown space)

        mask_cache_thres = 1e9 # sdf mask cache
        self.mask_cache_path = mask_cache_path  # None #
        self.mask_cache_thres = mask_cache_thres

        # if self.mask_cache_path is not None:  # set mask to false
        #     self.mask_cache = MaskCache(
        #         path=mask_cache_path,
        #         mask_cache_thres=mask_cache_thres).to(self.device)
        #     self._set_nonempty_mask()
        # else:
        self.mask_cache = None
        self.nonempty_mask = None

        # grad conv to calculate gradient
        self.init_gradient_conv()
        self.grad_mode = grad_mode

        # sdf_grid = self.sdf.grid
        # sdf_grid = self.smooth_conv(sdf_grid) if self.smooth_sdf else sdf_grid


    def init_gradient_conv(self, sigma = 0):
        self.grad_conv = nn.Conv3d(1,3,(3,3,3),stride=(1,1,1), padding=(1, 1, 1), padding_mode='replicate')
        kernel = np.asarray([
            [[1,2,1],[2,4,2],[1,2,1]],
            [[2,4,2],[4,8,4],[2,4,2]],
            [[1,2,1],[2,4,2],[1,2,1]],
        ])
        # sigma controls the difference between naive [-1,1] and sobel kernel
        distance = np.zeros((3,3,3))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    distance[i,j,k] = ((i-1)**2 + (j-1)**2 + (k-1)**2 - 1)
        kernel0 = kernel * np.exp(-distance * sigma)

        kernel1 = kernel0 / ( kernel0[0].sum() * 2 * self.voxel_size.item())
        weight = torch.from_numpy(np.concatenate([kernel1[None] for _ in range(3)])).float()
        weight[0,1,:,:] *= 0
        weight[0,0,:,:] *= -1
        weight[1,:,1,:] *= 0
        weight[1,:,0,:] *= -1
        weight[2,:,:,1] *= 0
        weight[2,:,:,0] *= -1
        self.grad_conv.weight.data = weight.unsqueeze(1).float()
        self.grad_conv.bias.data = torch.zeros(3)
        for param in self.grad_conv.parameters():
            param.requires_grad = False

        # smooth conv for TV
        self.tv_smooth_conv = nn.Conv3d(1, 1, (3, 3, 3), stride=1, padding=1, padding_mode='replicate')
        weight = torch.from_numpy(kernel0 / kernel0.sum()).float()
        self.tv_smooth_conv.weight.data = weight.unsqueeze(0).unsqueeze(0).float()
        self.tv_smooth_conv.bias.data = torch.zeros(1)
        for param in self.tv_smooth_conv.parameters():
            param.requires_grad = False



    def _gaussian_3dconv(self, ksize=3, sigma=1):
        x = np.arange(-(ksize//2),ksize//2 + 1,1)
        y = np.arange(-(ksize//2),ksize//2 + 1,1)
        z = np.arange(-(ksize//2),ksize//2 + 1,1)
        xx, yy, zz = np.meshgrid(x,y,z)
        kernel = np.exp(-(xx**2 + yy**2 + zz**2)/(2*sigma**2))
        kernel = torch.from_numpy(kernel).to(self.sdf.grid)
        m = nn.Conv3d(1,1,ksize,stride=1,padding=ksize//2, padding_mode='replicate')
        m.weight.data = kernel[None, None, ...] / kernel.sum()
        m.bias.data = torch.zeros(1)
        for param in m.parameters():
            param.requires_grad = False
        return m

    def init_smooth_conv(self, ksize=3, sigma=1):
        self.smooth_sdf = ksize > 0
        if self.smooth_sdf:
            self.smooth_conv = self._gaussian_3dconv(ksize, sigma)
            print("- "*10 + "init smooth conv with ksize={} and sigma={}".format(ksize, sigma) + " -"*10)

    def init_sdf_from_sdf(self, sdf0=None, smooth=False, reduce=1., ksize=3, sigma=1., zero2neg=True):
        print("\n", "- "*3 + "initing sdf from sdf" + " -"*3, "\n")
        if sdf0.shape != self.sdf.grid.shape:
            sdf0 = F.interpolate(sdf0, size=tuple(self.world_size), mode='trilinear', align_corners=True)
        if smooth:
            m = self._gaussian_3dconv(ksize, sigma)
            sdf_data = m(sdf0 / reduce)
            self.sdf.grid = torch.nn.Parameter(sdf_data).to(self.sdf.grid) / reduce
        else:
            self.sdf.grid.data = sdf0.to(self.sdf.grid) / reduce # + self.act_shift
        if self.mask_cache is not None:
            self._set_nonempty_mask()
        self.gradient = self.neus_sdf_gradient()



    def init_sdf_from_density(self, smooth=False, reduce=1., ksize=3, sigma=1., zero2neg=True):
        print("\n", "- "*3 + "initing sdf from density" + " -"*3, "\n")
        self.s = torch.nn.Parameter(torch.ones(1)) * 10
        if zero2neg:
            self.density.data[self.density.data==0] = -100
        if self.density.shape != self.sdf.grid.shape:
            self.density.data = F.interpolate(self.density.data, size=tuple(self.world_size), mode='trilinear', align_corners=True)
        if smooth:
            # first build the smoothing kernel
            m = self._gaussian_3dconv(ksize, sigma)
            sdf_data = m(-torch.tanh(self.density.data) / reduce)
            self.sdf.grid = torch.nn.Parameter(sdf_data)
        else:
            self.sdf.grid.data = -torch.tanh(self.density.data) / reduce # + self.act_shift

        self.gradient = self.neus_sdf_gradient()

    def _set_grid_resolution(self, num_voxels):
        # Determine grid resolution
        self.num_voxels = num_voxels
        self.voxel_size = ((self.xyz_max - self.xyz_min).prod() / num_voxels).pow(1/3)
        self.world_size = ((self.xyz_max - self.xyz_min) / self.voxel_size).long()
        self.voxel_size_ratio = self.voxel_size / self.voxel_size_base
        print('voxel_size      ', self.voxel_size)
        print('world_size      ', self.world_size)
        print('voxel_size_base ', self.voxel_size_base)
        print('voxel_size_ratio', self.voxel_size_ratio)

    def get_kwargs(self):
        return {
            'xyz_min': self.xyz_min.cpu().numpy(),
            'xyz_max': self.xyz_max.cpu().numpy(),
            'num_voxels': self.num_voxels,
            'num_voxels_base': self.num_voxels_base,
            'alpha_init': self.alpha_init,
            'nearest': self.nearest,
            'mask_cache_path': self.mask_cache_path,
            'mask_cache_thres': self.mask_cache_thres,
            'fast_color_thres': self.fast_color_thres,
            'geo_rgb_dim':self.geo_rgb_dim,
             'flow_ckpt_path':self.flow_ckpt_path,
             'flow_backbone':self.flow_backbone,
              'sg_config':self.sg_config,
            'HW':self.HW,
            'i_train':self.i_train,
            'N_iters':self.N_iters,
            'camera_noise':self.camera_noise,
            'range_shape':self.range_shape,
            'rect_size':self.rect_size,
            # 's_start': self.s_start,
            # 's_ratio': self.s_ratio,
            **self.rgbnet_kwargs,
        }

    def get_MaskCache_kwargs(self):
        return {
            'xyz_min': self.xyz_min.cpu().numpy(),
            'xyz_max': self.xyz_max.cpu().numpy(),
            'act_shift': self.act_shift,
            'voxel_size_ratio': self.voxel_size_ratio,
            'nearest': self.nearest
        }

    @torch.no_grad()
    def _set_nonempty_mask(self):
        # Find grid points that is inside nonempty (occupied) space
        self_grid_xyz = torch.stack(torch.meshgrid(
            torch.linspace(self.xyz_min[0], self.xyz_max[0], self.sdf.grid.shape[2]),
            torch.linspace(self.xyz_min[1], self.xyz_max[1], self.sdf.grid.shape[3]),
            torch.linspace(self.xyz_min[2], self.xyz_max[2], self.sdf.grid.shape[4]),
        ), -1)
        nonempty_mask = self.mask_cache(self_grid_xyz)[None,None].contiguous()
        if hasattr(self, 'nonempty_mask'):
            self.nonempty_mask = nonempty_mask
        else:
            self.register_buffer('nonempty_mask', nonempty_mask)
        # self.density[~self.nonempty_mask] = -100
        self.sdf.grid[~self.nonempty_mask] = 10


    @torch.no_grad()
    def maskout_near_cam_vox(self, cam_o, near):
        self_grid_xyz = torch.stack(torch.meshgrid(
            torch.linspace(self.xyz_min[0], self.xyz_max[0], self.sdf.grid.shape[2]),
            torch.linspace(self.xyz_min[1], self.xyz_max[1], self.sdf.grid.shape[3]),
            torch.linspace(self.xyz_min[2], self.xyz_max[2], self.sdf.grid.shape[4]),
        ), -1)
        nearest_dist = torch.stack([
            (self_grid_xyz.unsqueeze(-2) - co).pow(2).sum(-1).sqrt().amin(-1)
            for co in cam_o.split(100)  # for memory saving
        ]).amin(0)
        # self.density[nearest_dist[None,None] <= near] = -100
        self.sdf.grid[nearest_dist[None,None] <= near] = 1

    @torch.no_grad()
    def scale_volume_grid(self, num_voxels):
        print('scale_volume_grid start')
        ori_world_size = self.world_size
        self._set_grid_resolution(num_voxels)
        print('scale_volume_grid scale world_size from', ori_world_size, 'to', self.world_size)

        self.sdf.scale_volume_grid(self.world_size)
        if self.k0_dim > 0:
            self.k0.scale_volume_grid(self.world_size)
        if self.mask_cache is not None:
            self._set_nonempty_mask()
        print('scale_volume_grid finish')

    def density_total_variation(self, sdf_tv=0, smooth_grad_tv=0, sdf_thrd=0.999):
        tv = 0
        if sdf_tv > 0:
            tv += total_variation(self.sdf.grid, self.nonempty_mask) / 2 / self.voxel_size * sdf_tv
        if smooth_grad_tv > 0:
            smooth_tv_error = (self.tv_smooth_conv(self.gradient.permute(1,0,2,3,4)).detach() - self.gradient.permute(1,0,2,3,4))
            if self.nonempty_mask is None:
                smooth_tv_error = smooth_tv_error ** 2
            else:
                smooth_tv_error = smooth_tv_error[self.nonempty_mask.repeat(3,1,1,1,1)] ** 2
            tv += smooth_tv_error.mean() * smooth_grad_tv
        return tv

    def deform_smooth(self, smooth_grad_tv=0):
        sm = 0
        if smooth_grad_tv> 0 and self.point_deform is not None:
            self.gradient_deform = self.neus_sdf_gradient(sdf=self.point_deform.grid)
            sm = self.gradient_deform.norm(dim=1).mean() * smooth_grad_tv
        return sm

    def sdf_delta_weight_regularization(self, sdf_delta_rg=0):
        weight_sum=0
        if sdf_delta_rg>0:
            weight_sum =torch.abs(self.sdf_delta.grid) *sdf_delta_rg
        return weight_sum

    def sdf_deform_regularization(self, sdf_deform_rg=0):
        weight_sum = 0
        if sdf_deform_rg>0 and self.point_deform is not None:
            weight_sum = torch.mean(self.point_deform.grid.norm(dim=1))*sdf_deform_rg
        return weight_sum

    def k0_total_variation(self, k0_tv=1., k0_grad_tv=0.):
        if self.rgbnet is not None:
            v = self.k0.grid
        else:
            v = torch.sigmoid(self.k0.grid)
        tv = 0
        if k0_tv > 0:
            if self.nonempty_mask is None:
                tv += total_variation(v)
            else:
                tv += total_variation(v, self.nonempty_mask.repeat(1,v.shape[1],1,1,1))
        if k0_grad_tv > 0:
            raise NotImplementedError
        return tv


    def neus_sdf_gradient(self, mode=None, sdf=None):
        if sdf is None:
            sdf = self.sdf.grid
        if mode is None:
            mode = self.grad_mode
        if mode == 'interpolate':
            gradient = torch.zeros([1, 3] + [*self.sdf.grid.shape[-3:]]).to(self.sdf.grid.device)
            gradient[:,0,1:-1,:,:] = (sdf[:,0,2:,:,:] - sdf[:,0,:-2,:,:]) / 2 / self.voxel_size
            gradient[:,1,:,1:-1,:] = (sdf[:,0,:,2:,:] - sdf[:,0,:,:-2,:]) / 2 / self.voxel_size
            gradient[:,2,:,:,1:-1] = (sdf[:,0,:,:,2:] - sdf[:,0,:,:,:-2]) / 2 / self.voxel_size
        elif mode == 'grad_conv':
            # use sobel operator for gradient seems basically the same as the naive solution
            for param in self.grad_conv.parameters():
                assert not param.requires_grad
                pass
            gradient = self.grad_conv(sdf)
        elif mode == 'raw':
            gradient = torch.zeros([1, 3] + [*self.sdf.grid.shape[-3:]]).to(self.sdf.grid.device)
            gradient[:,0,:-1,:,:] = (sdf[:,0,1:,:,:] - sdf[:,0,:-1,:,:]) / self.voxel_size
            gradient[:,1,:,:-1,:] = (sdf[:,0,:,1:,:] - sdf[:,0,:,:-1,:]) / self.voxel_size
            gradient[:,2,:,:,:-1] = (sdf[:,0,:,:,1:] - sdf[:,0,:,:,:-1]) / self.voxel_size
        else:
            raise NotImplementedError
        return gradient
    
    def neus_alpha_from_sdf_scatter(self, viewdirs, ray_id, dist, sdf, gradients, global_step,
                            is_train, use_mid=True):
        if is_train:
            if not self.s_learn:
                s_val = 1. / (global_step + self.s_ratio / self.s_start - self.step_start) * self.s_ratio
                self.s_val.data = torch.ones_like(self.s_val) * s_val
            else:
                s_val = self.s_val.item()
        else:
            s_val = 0

        dirs = viewdirs[ray_id]
        inv_s = torch.ones(1).cuda() / self.s_val
        assert use_mid
        if use_mid:
            true_cos = (dirs * gradients).sum(-1, keepdim=True)
            cos_anneal_ratio = 1.0
            iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                         F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive (M, 1)

            sdf = sdf.unsqueeze(-1) # (M, 1)

            # Estimate signed distances at section points
            estimated_next_sdf = sdf + iter_cos * dist.reshape(-1, 1) * 0.5 # (M, 1)
            estimated_prev_sdf = sdf - iter_cos * dist.reshape(-1, 1) * 0.5 # (M, 1)
        else:
            estimated_next_sdf = torch.cat([sdf[..., 1:], sdf[..., -1:]], -1).reshape(-1, 1)
            estimated_prev_sdf = torch.cat([sdf[..., :1], sdf[..., :-1]], -1).reshape(-1, 1)

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s.reshape(-1, 1))
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s.reshape(-1, 1))
        p = prev_cdf - next_cdf
        c = prev_cdf
        alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0).squeeze()
        # if alpha.dim() != 1:
        #     alpha = torch.tensor([alpha])
        return s_val, alpha

    def neus_alpha_from_sdf(self, viewdirs, steps, sdf, gradients, global_step, is_train, use_mid=True):

        ori_shape = viewdirs.shape
        n_samples = steps.shape[-1]
        # force s_val value to change with global step
        if is_train:
            batch_size = steps.shape[0]
            if not self.s_learn:
                s_val = 1. / (global_step + self.s_ratio / self.s_start - self.step_start) * self.s_ratio
                self.s_val.data = torch.ones_like(self.s_val) * s_val
            else:
                s_val = self.s_val.item()
        else:
            dirs = viewdirs.reshape(-1, 3)
            steps = steps.reshape(-1, n_samples)
            batch_size = dirs.shape[0]
            s_val = 0
        if steps.shape[0] == 1:
            steps = steps.repeat(batch_size,1)
        dirs = viewdirs.unsqueeze(-2)
        inv_s = torch.ones(1).cuda() / self.s_val # * torch.exp(-inv_s)
        inv_s = inv_s.expand(batch_size * n_samples, 1)

        if use_mid:
            true_cos = (dirs * gradients).sum(-1, keepdim=True)
            cos_anneal_ratio = 1.0
            iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                         F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive
            iter_cos = iter_cos.reshape(-1,1)

            sdf = sdf.reshape(-1, 1)

            # calculate dist from steps / z_vals
            dists = steps[..., 1:] - steps[..., :-1]
            dists = torch.cat([dists, torch.Tensor([dists.mean()]).expand(dists[..., :1].shape)], -1)

            # Estimate signed distances at section points
            estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5
            estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5
        else:
            estimated_next_sdf = torch.cat([sdf[...,1:], sdf[...,-1:]], -1).reshape(-1, 1)
            estimated_prev_sdf = torch.cat([sdf[...,:1], sdf[...,:-1]], -1).reshape(-1, 1)

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)
        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, n_samples).clip(0.0, 1.0)
        return s_val, alpha

    def grid_sampler(self, xyz, *grids, mode=None, align_corners=True, smooth=False, displace=0., use_custom=False):
        '''Wrapper for the interp operation'''
        if mode is None:
            # bilinear is actually trilinear if 5D input is given to grid_sample
            mode = 'nearest' if self.nearest else 'bilinear'
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1,1,1,-1,3)
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
        if displace !=0:
            ind_norm[...,:] += displace * self.voxel_size
        # TODO: use `rearrange' to make it readable
        if smooth:
            grid = self.smooth_conv(grids[0])
        else:
            grid = grids[0]
        if use_custom:
            ret_lst = self.grid_sample_3d(grid, ind_norm).reshape(grid.shape[1], -1).T.reshape(*shape,
                                                                                               grid.shape[1]).squeeze()
        else:
            ret_lst = F.grid_sample(grid, ind_norm, mode=mode, align_corners=align_corners,padding_mode="border"
                                ).reshape(grid.shape[1],-1).T.reshape(*shape,grid.shape[1]).squeeze()

        return ret_lst

    def grid_sample_3d(self, image, optical):
        N, C, ID, IH, IW = image.shape
        _, D, H, W, _ = optical.shape

        ix = optical[..., 0]
        iy = optical[..., 1]
        iz = optical[..., 2]

        ix = ((ix + 1) / 2) * (IW - 1);
        iy = ((iy + 1) / 2) * (IH - 1);
        iz = ((iz + 1) / 2) * (ID - 1);
        with torch.no_grad():

            ix_tnw = torch.floor(ix);
            iy_tnw = torch.floor(iy);
            iz_tnw = torch.floor(iz);

            ix_tne = ix_tnw + 1;
            iy_tne = iy_tnw;
            iz_tne = iz_tnw;

            ix_tsw = ix_tnw;
            iy_tsw = iy_tnw + 1;
            iz_tsw = iz_tnw;

            ix_tse = ix_tnw + 1;
            iy_tse = iy_tnw + 1;
            iz_tse = iz_tnw;

            ix_bnw = ix_tnw;
            iy_bnw = iy_tnw;
            iz_bnw = iz_tnw + 1;

            ix_bne = ix_tnw + 1;
            iy_bne = iy_tnw;
            iz_bne = iz_tnw + 1;

            ix_bsw = ix_tnw;
            iy_bsw = iy_tnw + 1;
            iz_bsw = iz_tnw + 1;

            ix_bse = ix_tnw + 1;
            iy_bse = iy_tnw + 1;
            iz_bse = iz_tnw + 1;

        tnw = (ix_bse - ix) * (iy_bse - iy) * (iz_bse - iz);
        tne = (ix - ix_bsw) * (iy_bsw - iy) * (iz_bsw - iz);
        tsw = (ix_bne - ix) * (iy - iy_bne) * (iz_bne - iz);
        tse = (ix - ix_bnw) * (iy - iy_bnw) * (iz_bnw - iz);
        bnw = (ix_tse - ix) * (iy_tse - iy) * (iz - iz_tse);
        bne = (ix - ix_tsw) * (iy_tsw - iy) * (iz - iz_tsw);
        bsw = (ix_tne - ix) * (iy - iy_tne) * (iz - iz_tne);
        bse = (ix - ix_tnw) * (iy - iy_tnw) * (iz - iz_tnw);


        with torch.no_grad():

            torch.clamp(ix_tnw, 0, IW - 1, out=ix_tnw)
            torch.clamp(iy_tnw, 0, IH - 1, out=iy_tnw)
            torch.clamp(iz_tnw, 0, ID - 1, out=iz_tnw)

            torch.clamp(ix_tne, 0, IW - 1, out=ix_tne)
            torch.clamp(iy_tne, 0, IH - 1, out=iy_tne)
            torch.clamp(iz_tne, 0, ID - 1, out=iz_tne)

            torch.clamp(ix_tsw, 0, IW - 1, out=ix_tsw)
            torch.clamp(iy_tsw, 0, IH - 1, out=iy_tsw)
            torch.clamp(iz_tsw, 0, ID - 1, out=iz_tsw)

            torch.clamp(ix_tse, 0, IW - 1, out=ix_tse)
            torch.clamp(iy_tse, 0, IH - 1, out=iy_tse)
            torch.clamp(iz_tse, 0, ID - 1, out=iz_tse)

            torch.clamp(ix_bnw, 0, IW - 1, out=ix_bnw)
            torch.clamp(iy_bnw, 0, IH - 1, out=iy_bnw)
            torch.clamp(iz_bnw, 0, ID - 1, out=iz_bnw)

            torch.clamp(ix_bne, 0, IW - 1, out=ix_bne)
            torch.clamp(iy_bne, 0, IH - 1, out=iy_bne)
            torch.clamp(iz_bne, 0, ID - 1, out=iz_bne)

            torch.clamp(ix_bsw, 0, IW - 1, out=ix_bsw)
            torch.clamp(iy_bsw, 0, IH - 1, out=iy_bsw)
            torch.clamp(iz_bsw, 0, ID - 1, out=iz_bsw)

            torch.clamp(ix_bse, 0, IW - 1, out=ix_bse)
            torch.clamp(iy_bse, 0, IH - 1, out=iy_bse)
            torch.clamp(iz_bse, 0, ID - 1, out=iz_bse)

        image = image.view(N, C, ID * IH * IW)

        tnw_val = torch.gather(image, 2, (iz_tnw * IW * IH + iy_tnw * IW + ix_tnw).long().view(N, 1, D * H * W).repeat(1, C, 1))
        tne_val = torch.gather(image, 2, (iz_tne * IW * IH + iy_tne * IW + ix_tne).long().view(N, 1, D * H * W).repeat(1, C, 1))
        tsw_val = torch.gather(image, 2, (iz_tsw * IW * IH + iy_tsw * IW + ix_tsw).long().view(N, 1, D * H * W).repeat(1, C, 1))
        tse_val = torch.gather(image, 2, (iz_tse * IW * IH + iy_tse * IW + ix_tse).long().view(N, 1, D * H * W).repeat(1, C, 1))
        bnw_val = torch.gather(image, 2, (iz_bnw * IW * IH + iy_bnw * IW + ix_bnw).long().view(N, 1, D * H * W).repeat(1, C, 1))
        bne_val = torch.gather(image, 2, (iz_bne * IW * IH + iy_bne * IW + ix_bne).long().view(N, 1, D * H * W).repeat(1, C, 1))
        bsw_val = torch.gather(image, 2, (iz_bsw * IW * IH + iy_bsw * IW + ix_bsw).long().view(N, 1, D * H * W).repeat(1, C, 1))
        bse_val = torch.gather(image, 2, (iz_bse * IW * IH + iy_bse * IW + ix_bse).long().view(N, 1, D * H * W).repeat(1, C, 1))

        out_val = (tnw_val.view(N, C, D, H, W) * tnw.view(N, 1, D, H, W) +
                   tne_val.view(N, C, D, H, W) * tne.view(N, 1, D, H, W) +
                   tsw_val.view(N, C, D, H, W) * tsw.view(N, 1, D, H, W) +
                   tse_val.view(N, C, D, H, W) * tse.view(N, 1, D, H, W) +
                   bnw_val.view(N, C, D, H, W) * bnw.view(N, 1, D, H, W) +
                   bne_val.view(N, C, D, H, W) * bne.view(N, 1, D, H, W) +
                   bsw_val.view(N, C, D, H, W) * bsw.view(N, 1, D, H, W) +
                   bse_val.view(N, C, D, H, W) * bse.view(N, 1, D, H, W))

        return out_val

    def sample_ray_cuda(self, rays_o, rays_d, near, far, stepsize, maskout=True, use_bg=False, **render_kwargs):
        '''Sample query points on rays.
        All the output points are sorted from near to far.
        Input:
            rays_o, rayd_d:   both in [N, 3] indicating ray configurations.
            near, far:        the near and far distance of the rays.
            stepsize:         the number of voxels of each sample step.
        Output:
            ray_pts:          [M, 3] storing all the sampled points.
            ray_id:           [M]    the index of the ray of each point.
            step_id:          [M]    the i'th step on a ray of each point.
        '''
        far = 1e9  # the given far can be too small while rays stop when hitting scene bbox
        rays_o = rays_o.contiguous()
        rays_d = rays_d.contiguous()
        if not use_bg:
            stepdist = stepsize * self.voxel_size
        else:
            stepdist = stepsize * self.voxel_size_bg
        ray_pts, mask_outbbox, ray_id, step_id, N_steps, t_min, t_max,rays_start,rays_dir = render_utils_cuda.sample_pts_on_rays(
            rays_o, rays_d, self.xyz_min, self.xyz_max, near, far, stepdist)
        rays_view=rays_d / rays_d.norm(dim=-1, keepdim=True)
        ray_pts = rays_start[ray_id] + rays_view[ray_id]* step_id[..., None]*stepdist
        # correct the cuda output N_steps, which could have a bias of 1 randomly
        N_steps = ray_id.unique(return_counts=True)[1]
        if maskout:
            if not use_bg:
                mask_inbbox = ~mask_outbbox
            else:
                mask_inbbox = mask_outbbox
            ray_pts = ray_pts[mask_inbbox]
            ray_id = ray_id[mask_inbbox]
            step_id = step_id[mask_inbbox]

        return ray_pts, ray_id, step_id, mask_outbbox, N_steps,t_min,rays_start,rays_dir

    def sample_ray_ori(self, rays_o, rays_d, near, far, stepsize, is_train=False, **render_kwargs):
        '''Sample query points on rays'''
        # 1. determine the maximum number of query points to cover all possible rays
        N_samples = int(np.linalg.norm(np.array(self.sdf.grid.shape[2:])+1) / stepsize) + 1
        # 2. determine the two end-points of ray bbox intersection
        vec = torch.where(rays_d==0, torch.full_like(rays_d, 1e-6), rays_d)
        rate_a = (self.xyz_max - rays_o) / vec
        rate_b = (self.xyz_min - rays_o) / vec
        t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)
        t_max = torch.maximum(rate_a, rate_b).amin(-1).clamp(min=near, max=far)
        # 3. check wheter a raw intersect the bbox or not
        mask_outbbox = (t_max <= t_min)
        # 4. sample points on each ray
        rng = torch.arange(N_samples)[None].float()
        rng = rng.repeat(rays_d.shape[-2], 1)
        if is_train:
            rng += torch.rand_like(rng[:,[0]])
        step = stepsize * self.voxel_size * rng
        interpx = (t_min[...,None] + step/rays_d.norm(dim=-1,keepdim=True))
        rays_pts = rays_o[...,None,:] + rays_d[...,None,:] * interpx[...,None]
        # 5. update mask for query points outside bbox
        mask_outbbox = mask_outbbox[...,None] | ((self.xyz_min>rays_pts) | (rays_pts>self.xyz_max)).any(dim=-1)
        return rays_pts, mask_outbbox, step,t_min, t_max

    def positional_encoding_barf(self,input_enc,L,barf_c2f=None): # [B,...,N]
        # coarse-to-fine: smoothly mask positional encoding for BARF
        if barf_c2f is not None:
            # set weights for different frequency bands
            start,end = barf_c2f
            alpha = (self.progress.data-start)/(end-start)*L
            k = torch.arange(L,dtype=torch.float32,device=self.device)
            weight = (1-(alpha-k).clamp_(min=0, max=1).mul_(np.pi).cos_())/2
            # apply weights
            shape = input_enc.shape
            input_enc = (input_enc.view(-1,L)*weight).view(*shape)
        return input_enc

    def query_sdf_depth(self,rays_o, rays_d, global_step=None, **render_kwargs):
        N = len(rays_o)
        ray_pts, ray_id, step_id, mask_outbbox, N_steps,t_min,rays_start,rays_dir = self.sample_ray_cuda(
            rays_o=rays_o, rays_d=rays_d, is_train=global_step is not None, **render_kwargs)
        if self.mask_cache is not None:
            mask = self.mask_cache(ray_pts)
            ray_pts = ray_pts[mask]
            # ray_pts_defor = ray_pts_defor[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            mask_outbbox[~mask_outbbox] |= ~mask

        sdf_grid = self.smooth_conv(self.sdf.grid) if self.smooth_sdf else self.sdf.grid
        sdf = self.grid_sampler(ray_pts, sdf_grid.detach()) #ray_pts
        n_samples = step_id.max() + 1
        sdf_d = torch.ones([N, n_samples]).to(sdf.device) * 10  # max is set to 10
        sdf_d[ray_id, step_id] = sdf
        prev_sdf, next_sdf = sdf_d[:, :-1], sdf_d[:, 1:]
        sign = prev_sdf * next_sdf
        sign = torch.where(sign <= 0, torch.ones_like(sign), torch.zeros_like(sign))
        prev_idx = torch.argmax(sign, 1, keepdim=True)
        next_idx = prev_idx + 1
        sdf1 = torch.gather(sdf_d, 1, prev_idx).squeeze(-1)
        sdf2 = torch.gather(sdf_d, 1, next_idx).squeeze(-1)
        dist = render_kwargs['stepsize'] * self.voxel_size
        z_vals1 = prev_idx.squeeze(-1) * dist + dist * 0.5  # +t_min
        z_vals2 = next_idx.squeeze(-1) * dist + dist * 0.5

        z_vals_sdf0 = (sdf1 * z_vals2 - sdf2 * z_vals1) / (sdf1 - sdf2 + 1e-10)
        z_vals_sdf0 = torch.where(z_vals_sdf0 < z_vals1, torch.zeros_like(z_vals_sdf0), z_vals_sdf0)
        z_vals_sdf0 = torch.where(z_vals_sdf0 > z_vals2, torch.zeros_like(z_vals_sdf0), z_vals_sdf0)
        valid_through = (sdf1 * sdf2) <0
        valid_depth_ray = z_vals_sdf0 > 1e-10
        # t_min is for rays_d   z_vals_sdf0 is for rays_dir(norm)
        interpx = z_vals_sdf0 + t_min / rays_d.norm(dim=-1, keepdim=False)
        return interpx, valid_depth_ray*valid_through

    def query_sdf_point_wocuda(self,rays_o, rays_d, global_step=None,keep_dim=False,return_depth=False, **render_kwargs):
        rays_pts, mask_outbbox, step,t_min, t_max = self.sample_ray_ori(rays_o=rays_o, rays_d=rays_d,
                                                           is_train=global_step is not None, **render_kwargs)

        # if self.mask_cache is not None:
        #     mask_outbbox[~mask_outbbox] |= (~self.mask_cache(rays_pts[~mask_outbbox]))
        sdf_grid = self.smooth_conv(self.sdf.grid) if self.smooth_sdf else self.sdf.grid
        num_rays, num_samples = rays_pts.shape[0], rays_pts.shape[1]
        # filte the point out of box
        i_index, j_index = torch.where(mask_outbbox==False)
        rays_pts_inbbox = rays_pts[i_index,j_index]
        sdf_d = torch.ones_like(step)
        if self.bool_mapping_deformspace:
            deform, correction = self.warp_network(rays_pts_inbbox*self.mapping_sacle)
            new_coords = deform/self.mapping_sacle + rays_pts_inbbox
            correction = correction/self.mapping_sacle
        else:
            deform, correction = self.warp_network(rays_pts_inbbox)
            new_coords = rays_pts_inbbox + deform
        # new_coords, correction = new_coords.detach(), correction.detach()
        sdf = self.grid_sampler(new_coords, sdf_grid)
        sdf = sdf + correction.squeeze()  # whether backward?  .detach()
        sdf_d[i_index,j_index] = sdf
        # sdf_d = sdf_d.reshape(num_rays,num_samples)
        prev_sdf, next_sdf = sdf_d[:, :-1], sdf_d[:, 1:]
        sign = prev_sdf * next_sdf
        sign = torch.where(sign <= 0, torch.ones_like(sign), torch.zeros_like(sign))
        prev_idx = torch.argmax(sign, 1, keepdim=True)
        next_idx = prev_idx + 1
        sdf1 = torch.gather(sdf_d, 1, prev_idx).squeeze(-1)
        sdf2 = torch.gather(sdf_d, 1, next_idx).squeeze(-1)
        dist = render_kwargs['stepsize'] * self.voxel_size
        z_vals1 = prev_idx.squeeze(-1) * dist + dist * 0.5  # +t_min
        z_vals2 = next_idx.squeeze(-1) * dist + dist * 0.5
        z_vals_sdf0 = (sdf1 * z_vals2 - sdf2 * z_vals1) / (sdf1 - sdf2 + 1e-10)
        z_vals_sdf0 = torch.where(z_vals_sdf0 < z_vals1, torch.zeros_like(z_vals_sdf0), z_vals_sdf0)
        z_vals_sdf0 = torch.where(z_vals_sdf0 > z_vals2, torch.zeros_like(z_vals_sdf0), z_vals_sdf0)
        mask_idx_ray = z_vals_sdf0 > 1e-10
        valid_through = (sdf1 * sdf2) < 0
        mask_idx_ray = valid_through*mask_idx_ray

        interpx = (t_min+ z_vals_sdf0 / rays_d.norm(dim=-1, keepdim=False))
        if keep_dim == True:
            rays_pts0 = rays_o + rays_d * interpx[..., None]
            return rays_pts0, mask_idx_ray, sdf_d
        else:
            pts_sdf0 = rays_o[mask_idx_ray] + rays_d[mask_idx_ray] * interpx[mask_idx_ray][..., None]
            if return_depth:
                interpx = z_vals_sdf0 + t_min / rays_d.norm(dim=-1, keepdim=False)
                return pts_sdf0,interpx[mask_idx_ray]
            else:
                return pts_sdf0

    def query_sdf_point_wocuda_wodeform(self, rays_o, rays_d, global_step=None, keep_dim=False, return_depth=False,
                               **render_kwargs):
        rays_pts, mask_outbbox, step, t_min, t_max = self.sample_ray_ori(rays_o=rays_o, rays_d=rays_d,
                                                                         is_train=global_step is not None,
                                                                         **render_kwargs)

        # if self.mask_cache is not None:
        #     mask_outbbox[~mask_outbbox] |= (~self.mask_cache(rays_pts[~mask_outbbox]))
        sdf_grid = self.smooth_conv(self.sdf.grid) if self.smooth_sdf else self.sdf.grid
        num_rays, num_samples = rays_pts.shape[0], rays_pts.shape[1]
        sdf_d = self.grid_sampler(rays_pts, sdf_grid)

        sdf_d = sdf_d.reshape(num_rays, num_samples)
        prev_sdf, next_sdf = sdf_d[:, :-1], sdf_d[:, 1:]
        sign = prev_sdf * next_sdf
        sign = torch.where(sign <= 0, torch.ones_like(sign), torch.zeros_like(sign))
        prev_idx = torch.argmax(sign, 1, keepdim=True)
        next_idx = prev_idx + 1
        sdf1 = torch.gather(sdf_d, 1, prev_idx).squeeze(-1)
        sdf2 = torch.gather(sdf_d, 1, next_idx).squeeze(-1)
        dist = render_kwargs['stepsize'] * self.voxel_size
        z_vals1 = prev_idx.squeeze(-1) * dist + dist * 0.5  # +t_min
        z_vals2 = next_idx.squeeze(-1) * dist + dist * 0.5
        z_vals_sdf0 = (sdf1 * z_vals2 - sdf2 * z_vals1) / (sdf1 - sdf2 + 1e-10)
        z_vals_sdf0 = torch.where(z_vals_sdf0 < z_vals1, torch.zeros_like(z_vals_sdf0), z_vals_sdf0)
        z_vals_sdf0 = torch.where(z_vals_sdf0 > z_vals2, torch.zeros_like(z_vals_sdf0), z_vals_sdf0)
        mask_idx_ray = z_vals_sdf0 > 1e-10
        valid_through = (sdf1 * sdf2) < 0
        mask_idx_ray = valid_through * mask_idx_ray

        interpx = (t_min + z_vals_sdf0 / rays_d.norm(dim=-1, keepdim=False))
        if keep_dim == True:
            rays_pts0 = rays_o + rays_d * interpx[..., None]
            return rays_pts0, mask_idx_ray, sdf_d
        else:
            pts_sdf0 = rays_o[mask_idx_ray] + rays_d[mask_idx_ray] * interpx[mask_idx_ray][..., None]
            if return_depth:
                interpx = z_vals_sdf0 + t_min / rays_d.norm(dim=-1, keepdim=False)
                return pts_sdf0, interpx[mask_idx_ray]
            else:
                return pts_sdf0

    def query_sdf_point_wocuda_render(self,rays_o, rays_d, global_step=None,keep_dim=False,
                                      return_depth=False, wo_deform=False, **render_kwargs):
        if global_step is None:
            self.progress.data.fill_(1.)
            self.warp_network.progress.data.fill_(1.)
        else:
            self.progress.data.fill_(global_step/self.N_iters)
            self.warp_network.progress.data.fill_(global_step / self.N_iters)
        N = len(rays_o)
        ray_pts, mask_outbbox, step, t_min, t_max = self.sample_ray_ori(rays_o=rays_o, rays_d=rays_d,
                                                                        is_train=global_step is not None,
                                                                        **render_kwargs)
        ray_id = torch.arange(step.shape[0] * step.shape[1]) // step.shape[1]
        ray_pts = ray_pts.reshape(-1, 3)
        mask_outbbox = mask_outbbox.flatten()
        step = step.flatten()

        mask = ~mask_outbbox
        ray_pts = ray_pts[mask]
        ray_id = ray_id[mask]
        step = step[mask]

        sdf_grid = self.sdf.grid
        sdf_grid = self.smooth_conv(sdf_grid) if self.smooth_sdf else sdf_grid
        sdf_grid_mapping = self.activation(self.sdf_alpha) * \
                           (1 / (1 + torch.exp(-self.activation(self.sdf_beta) * sdf_grid)) - 0.5)

        if wo_deform:
            sdf_final = self.grid_sampler(ray_pts, sdf_grid)
            sdf_final = self.activation(self.sdf_alpha) * \
                        (1 / (1 + torch.exp(-self.activation(self.sdf_beta) * sdf_final)) - 0.5)
            self.gradient = self.neus_sdf_gradient(sdf=sdf_grid_mapping)  # sdf_grid
            gradient = self.grid_sampler(ray_pts, self.gradient)
        elif self.use_explict_deform:
            deform_vec = self.grid_sampler(ray_pts, self.point_deform.grid)
            ray_pts = ray_pts + deform_vec
            deform_xyz = self.grid_xyz + rearrange(self.point_deform.grid[0], 'c h w l -> h w l c', c=3)
            sdf_grid = self.grid_sampler(deform_xyz, sdf_grid)[None, None, ...]
            sdf_final = self.grid_sampler(ray_pts, sdf_grid)
            self.gradient = self.neus_sdf_gradient(sdf=sdf_grid_mapping)  # sdf_grid
            gradient = self.grid_sampler(ray_pts, self.gradient)
        elif self.use_implict_deform:
            if self.bool_mapping_deformspace:
                deform, correction = self.warp_network(ray_pts * self.mapping_sacle)
                new_coords = deform / self.mapping_sacle + ray_pts
                correction = correction / self.mapping_sacle
            else:
                deform, correction = self.warp_network(ray_pts)
                new_coords = deform + ray_pts

            sdf = self.grid_sampler(new_coords, sdf_grid, use_custom=True)  # SDF value in template space
            sdf_final = sdf + correction.squeeze()  # add correction
            if self.use_mapping_fun:
                sdf_final = self.activation(self.sdf_alpha) * \
                            (1 / (1 + torch.exp(-self.activation(self.sdf_beta) * sdf_final)) - 0.5)


            if sdf_final.shape[0] == 0:
                gradient = torch.zeros((0, 3))
            else:
                gradient = \
                    torch.autograd.grad(sdf_final, [ray_pts], grad_outputs=torch.ones_like(sdf_final),
                                        create_graph=True)[0]



        dist = render_kwargs['stepsize'] * self.voxel_size
        viewdirs = rays_d / rays_d.norm(dim=-1, keepdim=True)
        s_val, alpha = self.neus_alpha_from_sdf_scatter(viewdirs, ray_id, dist, sdf_final, gradient,
                                                        global_step=global_step,
                                                        is_train=global_step is not None, use_mid=True)
        weights, alphainv_last = Alphas2Weights.apply(alpha, ray_id, N)


        n_step = segment_coo(
            src=(weights * step),
            index=ray_id,
            out=torch.zeros([N]),
            reduce='sum')
        depth = t_min[..., None] + n_step[..., None] / rays_d.norm(dim=-1, keepdim=True)
        mask = n_step>0.
        if keep_dim == True:
            rays_pts = rays_o + rays_d * depth
            return rays_pts, mask, depth
        else:
            pts_sdf0 = rays_o[mask] + rays_d[mask] * depth[mask][..., None]
            if return_depth:
                return pts_sdf0, depth[mask]
            else:
                return pts_sdf0

    def forward(self, rays_o, rays_d, viewdirs, global_step=None,**render_kwargs):
        if global_step is None:
            self.progress.data.fill_(1.)
            self.warp_network.progress.data.fill_(1.)
        else:
            self.progress.data.fill_(global_step/self.N_iters)
            self.warp_network.progress.data.fill_(global_step / self.N_iters)
        ret_dict = {}
        N = len(rays_o)

        ray_pts, mask_outbbox, step, t_min, t_max = self.sample_ray_ori(rays_o=rays_o, rays_d=rays_d,
                                                                         is_train=global_step is not None,
                                                                         **render_kwargs)
        ray_id = torch.arange(step.shape[0] * step.shape[1])// step.shape[1]
        ray_pts = ray_pts.reshape(-1,3)
        mask_outbbox = mask_outbbox.flatten()
        step = step.flatten()

        mask = ~mask_outbbox
        ray_pts = ray_pts[mask]
        ray_id = ray_id[mask]
        step = step[mask]
        mask_outbbox = mask_outbbox[mask]
        sdf_grid = self.sdf.grid
        sdf_grid = self.smooth_conv(sdf_grid) if self.smooth_sdf else sdf_grid
        sdf_grid_mapping = self.activation(self.sdf_alpha) * \
                           (1 / (1 + torch.exp(-self.activation(self.sdf_beta) * sdf_grid)) - 0.5)
        if  self.use_explict_deform:
            deform_vec = self.grid_sampler(ray_pts,self.point_deform.grid)
            ray_pts = ray_pts + deform_vec
            deform_xyz = self.grid_xyz + rearrange(self.point_deform.grid[0], 'c h w l -> h w l c', c=3)
            sdf_grid = self.grid_sampler(deform_xyz, sdf_grid)[None, None, ...]
            sdf_final = self.grid_sampler(ray_pts, sdf_grid)
            self.gradient = self.neus_sdf_gradient(sdf=sdf_grid_mapping)  # sdf_grid
            gradient = self.grid_sampler(ray_pts, self.gradient)
        elif  self.use_implict_deform:
            if self.bool_mapping_deformspace:
                deform, correction = self.warp_network(ray_pts * self.mapping_sacle)
                new_coords = deform/self.mapping_sacle + ray_pts
                correction = correction / self.mapping_sacle
            else:
                deform, correction = self.warp_network(ray_pts)
                new_coords = deform + ray_pts

            sdf = self.grid_sampler(new_coords, sdf_grid, use_custom=True)  # SDF value in template space
            u = new_coords[:, 0]
            v = new_coords[:, 1]
            w = new_coords[:, 2]
            grad_outputs = torch.ones_like(u)
            grad_u = torch.autograd.grad(u, [ray_pts], grad_outputs=grad_outputs, create_graph=True)[0]
            grad_v = torch.autograd.grad(v, [ray_pts], grad_outputs=grad_outputs, create_graph=True)[0]
            grad_w = torch.autograd.grad(w, [ray_pts], grad_outputs=grad_outputs, create_graph=True)[0]
            grad_deform = torch.stack([grad_u, grad_v, grad_w],dim=2)  # gradient of deformation wrt. input position

            sdf_final = sdf + correction.squeeze()  # add correction
            if self.use_mapping_fun:
                sdf_final = self.activation(self.sdf_alpha) * \
                           (1 / (1 + torch.exp(-self.activation(self.sdf_beta) * sdf_final)) - 0.5)
            # calculate gradient of the deformation field
            # grad_temp = \
            #     torch.autograd.grad(sdf, [new_coords], grad_outputs=torch.ones_like(sdf), create_graph=True)[
            #         0]

            if sdf_final.shape[0]==0:
                gradient = torch.zeros((0,3))
            else:
                gradient = \
                torch.autograd.grad(sdf_final, [ray_pts], grad_outputs=torch.ones_like(sdf_final), create_graph=True)[0]

        else:
            sdf_final = self.grid_sampler(ray_pts, sdf_grid)
            sdf_final = self.activation(self.sdf_alpha) * \
                        (1 / (1 + torch.exp(-self.activation(self.sdf_beta) * sdf_final)) - 0.5)
            self.gradient = self.neus_sdf_gradient(sdf=sdf_grid_mapping)  # sdf_grid
            gradient = self.grid_sampler(ray_pts, self.gradient)


        dist = render_kwargs['stepsize'] * self.voxel_size
        s_val, alpha = self.neus_alpha_from_sdf_scatter(viewdirs, ray_id, dist, sdf_final, gradient, global_step=global_step,
                                                        is_train=global_step is not None, use_mid=True)
        weights, alphainv_last = Alphas2Weights.apply(alpha, ray_id, N)
        if self.fast_color_thres > 0: # this can save to speed-up since the sampled-points with low_thres have no contribution
            mask = weights > self.fast_color_thres
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step = step[mask]
            alpha = alpha[mask]
            gradient = gradient[mask]
            weights, alphainv_last = Alphas2Weights.apply(alpha, ray_id, N)
        
        rgb_feat = []
        if not self.rgbnet_full_implicit:
            k0 = self.k0(ray_pts)
            rgb_feat.append(k0)
        rays_xyz = (ray_pts - self.xyz_min) / (self.xyz_max - self.xyz_min)

        xyz_emb = (rays_xyz.unsqueeze(-1) * self.posfreq).flatten(-2)


        xyz_emb = self.positional_encoding_barf(torch.cat([xyz_emb.sin(), xyz_emb.cos()],-1),
                                                L=self.rgbnet_kwargs['posbase_pe'], barf_c2f=self.barf_c2f)

        xyz_emb = torch.cat([rays_xyz, xyz_emb], dim=-1)

        rgb_feat.append(xyz_emb)

        viewdirs_emb = (viewdirs.unsqueeze(-1) * self.viewfreq).flatten(-2)
        viewdirs_emb = self.positional_encoding_barf(torch.cat([viewdirs_emb.sin(), viewdirs_emb.cos()],-1),
                                                L=self.rgbnet_kwargs['viewbase_pe'], barf_c2f=self.barf_c2f)
        viewdirs_emb = torch.cat([viewdirs, viewdirs_emb], dim=-1)

        rgb_feat.append(viewdirs_emb.flatten(0, -2)[ray_id])
        rgb_feat = torch.cat(rgb_feat, -1)

        if self.geo_rgb_dim == 3:
            normal = gradient / (gradient.norm(dim=-1, keepdim=True) + 1e-5)
            rgb_feat = torch.cat([rgb_feat, normal], -1)

        rgb_logit = self.rgbnet(rgb_feat)
        rgb = torch.sigmoid(rgb_logit)
        rgb_marched = segment_coo(
            src=(weights.unsqueeze(-1) * rgb),
            index=ray_id, out=torch.zeros([N, 3]), reduce='sum')
        cum_weights = segment_coo(
            src=(weights.unsqueeze(-1)),
            index=ray_id, out=torch.zeros([N, 1]), reduce='sum')
        # Ray marching
        rgb_marched = rgb_marched + (1 - cum_weights) * render_kwargs['bg']
        # rgb_marched = rgb_marched + alphainv_last.unsqueeze(-1) * render_kwargs['bg']
        rgb_marched = rgb_marched.clamp(0, 1)

        if gradient is not None and render_kwargs.get('render_grad', False):
            normal = gradient / (gradient.norm(2, -1, keepdim=True) + 1e-6)
            normal_marched = segment_coo(
                src=(weights.unsqueeze(-1) * normal),
                index=ray_id, out=torch.zeros([N, 3]), reduce='sum')
        else:
            normal_marched = None
        if getattr(render_kwargs, 'render_depth', True):
            n_step = segment_coo(
                src=(weights * step),
                index=ray_id,
                out=torch.zeros([N]),
                reduce='sum')
            depth = t_min/rays_d.norm(dim=-1, keepdim=False) + n_step


            disp = 1 / depth
        else:
            depth = None
            disp = None
        ret_dict.update({
            'alphainv_cum': alphainv_last,
            'weights': weights,
            'cum_weights':cum_weights,
            'rgb_marched': rgb_marched,
            'normal_marched': normal_marched,
            'raw_alpha': alpha,
            'raw_rgb': rgb,
            'depth': depth,
            'disp': disp,
            'mask': mask,
            'mask_outbbox':mask_outbbox,
            'gradient': gradient,
            # 'gradient_origin':gradient_origin,
            # "grad_temp": grad_temp,
            "s_val": s_val,
            'deform':deform,
            'grad_deform': grad_deform,
            'sdf_correct': correction
        })
        return ret_dict

    def inference(self, rays_o, rays_d, viewdirs, global_step=None, training=True, **render_kwargs):
        if global_step is None:
            self.progress.data.fill_(1.)
            self.warp_network.progress.data.fill_(1.)
        else:
            self.progress.data.fill_(global_step / self.N_iters)
            self.warp_network.progress.data.fill_(global_step / self.N_iters)
        ret_dict = {}
        N = len(rays_o)

        ray_pts, ray_id, step_id, mask_outbbox, N_steps, t_min, rays_start, rays_dir = self.sample_ray_cuda(
                rays_o=rays_o, rays_d=rays_d, is_train=global_step is not None, **render_kwargs)

        # skip known free space
        # if self.mask_cache is not None:
        # mask =  torch.(step_id)  #self.mask_cache(ray_pts)
        # ray_pts = ray_pts[mask]
        # ray_id = ray_id[mask]
        # step_id = step_id[mask]
        # if training:
        #     mask_outbbox = mask_outbbox[mask]
        # else:
        #     mask_outbbox[~mask_outbbox] |= ~mask
        # else:
        #     mask = ~mask_outbbox
        #     ray_pts = ray_pts[mask]
        #     ray_id = ray_id[mask]
        #     step_id = step_id[mask]
        #     mask_outbbox = mask_outbbox[mask]

        if ray_pts.shape[0]==1:  # especial case: dim error
            ray_pts = ray_pts.repeat(2,1)
            ray_id = ray_id.repeat(2)
            step_id = step_id.repeat(2)
        sdf_grid = self.sdf.grid
        sdf_grid = self.smooth_conv(sdf_grid) if self.smooth_sdf else sdf_grid
        sdf_grid_mapping = self.activation(self.sdf_alpha) * \
                           (1 / (1 + torch.exp(-self.activation(self.sdf_beta) * sdf_grid)) - 0.5)
        with torch.enable_grad():
            ray_pts.requires_grad_(True)
            if self.bool_mapping_deformspace:
                deform, correction = self.warp_network(ray_pts * self.mapping_sacle)
                new_coords = deform/self.mapping_sacle + ray_pts
                correction = correction/self.mapping_sacle
            else:
                deform, correction = self.warp_network(ray_pts)
                new_coords = deform + ray_pts
            sdf = self.grid_sampler(new_coords, sdf_grid, use_custom=True)  # SDF value in template space

            sdf_final = sdf + correction.squeeze()  # add correction
            if self.use_mapping_fun:
                sdf_final = self.activation(self.sdf_alpha) * \
                           (1 / (1 + torch.exp(-self.activation(self.sdf_beta) * sdf_final)) - 0.5)
            try:
                if sdf_final.shape[0] == 0:
                    gradient = torch.zeros((0, 3))
                else:
                    gradient = \
                        torch.autograd.grad(sdf_final, [ray_pts], grad_outputs=torch.ones_like(sdf_final),
                                            create_graph=True)[0]
            except:
                import pdb
                pdb.set_trace()


        gradient_error = ((torch.linalg.norm(gradient, ord=2, dim=-1) - 1.0) ** 2).mean()
        dist = render_kwargs['stepsize'] * self.voxel_size
        s_val, alpha = self.neus_alpha_from_sdf_scatter(viewdirs, ray_id, dist, sdf_final, gradient,
                                                        global_step=global_step,
                                                        is_train=global_step is not None, use_mid=True)
        weights, alphainv_last = Alphas2Weights.apply(alpha, ray_id, N)
        if self.fast_color_thres > 0:  # this can save to speed-up since the sampled-points with low_thres have no contribution
            mask = weights > self.fast_color_thres
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            alpha = alpha[mask]
            gradient = gradient[mask]
            weights, alphainv_last = Alphas2Weights.apply(alpha, ray_id, N)

        rgb_feat = []
        if not self.rgbnet_full_implicit:
            k0 = self.k0(ray_pts)
            rgb_feat.append(k0)
        rays_xyz = (ray_pts - self.xyz_min) / (self.xyz_max - self.xyz_min)

        xyz_emb = (rays_xyz.unsqueeze(-1) * self.posfreq).flatten(-2)
        xyz_emb = self.positional_encoding_barf(torch.cat([xyz_emb.sin(), xyz_emb.cos()], -1),
                                                L=self.rgbnet_kwargs['posbase_pe'], barf_c2f=self.barf_c2f)
        xyz_emb = torch.cat([rays_xyz, xyz_emb], dim=-1)
        rgb_feat.append(xyz_emb)

        viewdirs_emb = (viewdirs.unsqueeze(-1) * self.viewfreq).flatten(-2)
        viewdirs_emb = self.positional_encoding_barf(torch.cat([viewdirs_emb.sin(), viewdirs_emb.cos()], -1),
                                                     L=self.rgbnet_kwargs['viewbase_pe'], barf_c2f=self.barf_c2f)
        viewdirs_emb = torch.cat([viewdirs, viewdirs_emb], dim=-1)

        rgb_feat.append(viewdirs_emb.flatten(0, -2)[ray_id])
        rgb_feat = torch.cat(rgb_feat, -1)

        if self.geo_rgb_dim == 3:
            normal = gradient / (gradient.norm(dim=-1, keepdim=True) + 1e-5)
            rgb_feat = torch.cat([rgb_feat, normal], -1)

        rgb_logit = self.rgbnet(rgb_feat)
        rgb = torch.sigmoid(rgb_logit)
        rgb_marched = segment_coo(
            src=(weights.unsqueeze(-1) * rgb),
            index=ray_id, out=torch.zeros([N, 3]), reduce='sum')
        cum_weights = segment_coo(
            src=(weights.unsqueeze(-1)),
            index=ray_id, out=torch.zeros([N, 1]), reduce='sum')
        # Ray marching
        rgb_marched = rgb_marched + (1 - cum_weights) * render_kwargs['bg']
        # rgb_marched = rgb_marched + alphainv_last.unsqueeze(-1) * render_kwargs['bg']
        rgb_marched = rgb_marched.clamp(0, 1)

        if gradient is not None:
            normal = gradient / (gradient.norm(2, -1, keepdim=True) + 1e-6)
            normal_marched = segment_coo(
                src=(weights.unsqueeze(-1) * normal),
                index=ray_id, out=torch.zeros([N, 3]), reduce='sum')
        else:
            normal_marched = None
        if getattr(render_kwargs, 'render_depth', True):
            depth = segment_coo(
                src=(weights * step_id * dist),
                index=ray_id, out=torch.zeros([N]), reduce='sum')
            # depth = depth+t_min/rays_d.norm(dim=-1, keepdim=False)  # t_min is not normlized
            disp = 1 / depth
        else:
            depth = None
            disp = None
        ret_dict.update({
            'alphainv_cum': alphainv_last,
            'weights': weights,
            'cum_weights': cum_weights,
            'rgb_marched': rgb_marched,
            'normal_marched': normal_marched,
            'raw_alpha': alpha,
            'raw_rgb': rgb,
            'depth': depth,
            'disp': disp,
            'mask': torch.ones_like(step_id),
            'mask_outbbox': mask_outbbox,
            'gradient': gradient,
            "gradient_error": gradient_error,
            "s_val": s_val
        })
        return ret_dict


    def mesh_color_forward(self, ray_pts, **kwargs):

        ### coarse-stage geometry and texture are low in resolution

        sdf_grid = self.smooth_conv(self.sdf.grid) if self.smooth_sdf else self.sdf.grid
        self.gradient = self.neus_sdf_gradient(sdf=sdf_grid)
        gradient = self.grid_sampler(ray_pts, self.gradient).reshape(-1, 3)
        normal = gradient / (gradient.norm(dim=-1, keepdim=True) + 1e-5)
        viewdirs = -normal

        rgb_feat = []
        k0 = self.k0(ray_pts)
        rgb_feat.append(k0)

        rays_xyz = (ray_pts - self.xyz_min) / (self.xyz_max - self.xyz_min)
        xyz_emb = (rays_xyz.unsqueeze(-1) * self.posfreq).flatten(-2)
        xyz_emb = torch.cat([rays_xyz, xyz_emb.sin(), xyz_emb.cos()], -1)
        rgb_feat.append(xyz_emb)

        viewdirs_emb = (viewdirs.unsqueeze(-1) * self.viewfreq).flatten(-2)
        viewdirs_emb = torch.cat(
            [viewdirs, viewdirs_emb.sin(), viewdirs_emb.cos()], -1)
        rgb_feat.append(viewdirs_emb.flatten(0, -2))
        rgb_feat = torch.cat(rgb_feat, -1)
        if self.geo_rgb_dim == 3:
            rgb_feat = torch.cat([rgb_feat, normal], -1)
        rgb_logit = self.rgbnet(rgb_feat)
        rgb = torch.sigmoid(rgb_logit)

        return rgb

    def extract_deform_geometry(self, bound_min, bound_max, resolution=128, threshold=0.0, **kwargs):
        if self.smooth_sdf:
            sdf_grid = self.smooth_conv(self.sdf.grid)
        else:
            sdf_grid = self.sdf.grid

        # self._set_nonempty_mask()
        def query_func(pts):
            if self.bool_mapping_deformspace:
                deform, correction = self.warp_network(self.mapping_sacle * pts)
                new_coords = deform / self.mapping_sacle + pts
                correction = correction / self.mapping_sacle
            else:
                deform, correction = self.warp_network(pts)
                new_coords = pts + deform
            sdf = self.grid_sampler(new_coords, sdf_grid)
            sdf_final = sdf + correction.squeeze()
            return -sdf_final

        if resolution is None:
            resolution = self.world_size[0]
        return extract_geometry(bound_min,
                                bound_max,
                                resolution=resolution,
                                threshold=threshold,
                                query_func=query_func)

    def extract_geometry(self, bound_min, bound_max, resolution=128, threshold=0.0, **kwargs):
        if self.smooth_sdf:
            sdf_grid = self.smooth_conv(self.sdf.grid)
        else:
            sdf_grid = self.self.sdf.grid
        # self._set_nonempty_mask()
        query_func = lambda pts: self.grid_sampler(pts, -sdf_grid)
        if resolution is None:
            resolution = self.world_size[0]
        return extract_geometry(bound_min,
                                bound_max,
                                resolution=resolution,
                                threshold=threshold,
                                query_func=query_func)


''' Module for the searched coarse geometry
It supports query for the known free space and unknown space.
'''
class MaskCache(nn.Module):
    def __init__(self, path, mask_cache_thres, ks=3):
        super().__init__()
        sdf_dict = np.load(path, allow_pickle=True).tolist()
        sdf0 = torch.tensor(sdf_dict['sdf_grid_xyz'])
        sdf0 = rearrange(sdf0, 'd h w -> 1 1 d h w')
        xyz_min = torch.tensor(sdf_dict['xyz_min']).to(torch.float32)
        xyz_max = torch.tensor(sdf_dict['xyz_max']).to(torch.float32)
        self.mask_cache_thres = mask_cache_thres
        self.register_buffer('xyz_min', (xyz_min))
        self.register_buffer('xyz_max', (xyz_max))
        self.register_buffer('sdf', sdf0)
        self.nearest = False
    @torch.no_grad()
    def forward(self, xyz):
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1,1,1,-1,3)
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
        sdf = F.grid_sample(self.sdf, ind_norm, align_corners=True)
        sdf = sdf.reshape(*shape)
        return (sdf < self.mask_cache_thres)



''' Misc
'''

def total_variation(v, mask=None):

    if torch.__version__ == '1.10.0':
        tv2 = v.diff(dim=2).abs()
        tv3 = v.diff(dim=3).abs()
        tv4 = v.diff(dim=4).abs()
    else:
        tv2 = (v[:,:,1:,:,:] - v[:,:,:-1,:,:]).abs()
        tv3 = (v[:,:,:,1:,:] - v[:,:,:,:-1,:]).abs()
        tv4 = (v[:,:,:,:,1:] - v[:,:,:,:,:-1]).abs()
    if mask is not None:
        tv2 = tv2[mask[:,:,:-1] & mask[:,:,1:]]
        tv3 = tv3[mask[:,:,:,:-1] & mask[:,:,:,1:]]
        tv4 = tv4[mask[:,:,:,:,:-1] & mask[:,:,:,:,1:]]
    if mask is None:
        mask = torch.ones_like(v)
    return (tv2.sum() + tv3.sum() + tv4.sum()) / 3 / mask.sum()

class Alphas2Weights(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alpha, ray_id, N):
        weights, T, alphainv_last, i_start, i_end = render_utils_cuda.alpha2weight(alpha, ray_id, N)
        if alpha.requires_grad:
            ctx.save_for_backward(alpha, weights, T, alphainv_last, i_start, i_end)
            ctx.n_rays = N
        return weights, alphainv_last

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_weights, grad_last):
        alpha, weights, T, alphainv_last, i_start, i_end = ctx.saved_tensors
        grad = render_utils_cuda.alpha2weight_backward(
                alpha, weights, T, alphainv_last,
                i_start, i_end, ctx.n_rays, grad_weights, grad_last)
        return grad, None, None


''' Ray and batch
'''
def get_rays(H, W, K, c2w, inverse_y, flip_x, flip_y, mode='center'):
    i, j = torch.meshgrid(
        torch.linspace(0, W-1, W, device=c2w.device),
        torch.linspace(0, H-1, H, device=c2w.device))  # pytorch's meshgrid has indexing='ij'
    i = i.t().float()
    j = j.t().float()
    if mode == 'lefttop':
        pass
    elif mode == 'center':
        i, j = i+0.5, j+0.5
    elif mode == 'random':
        i = i+torch.rand_like(i)
        j = j+torch.rand_like(j)
    else:
        raise NotImplementedError

    if flip_x:
        i = i.flip((1,))
    if flip_y:
        j = j.flip((0,))
    if inverse_y:
        dirs = torch.stack([(i-K[0][2])/K[0][0], (j-K[1][2])/K[1][1], torch.ones_like(i)], -1)
    else:
        dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,3].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,3], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d

    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]

    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)

    return rays_o, rays_d


def get_rays_of_a_view(H, W, K, c2w, ndc, inverse_y, flip_x, flip_y, mode='center'):
    rays_o, rays_d = get_rays(H, W, K, c2w, inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y, mode=mode)
    rays_d = viewdirs = rays_d / rays_d.norm(dim=-1, keepdim=True)
    if ndc:
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)
    return rays_o, rays_d, viewdirs



def get_rays_of_a_view_semantic(H, W, K, c2w, ndc, inverse_y, flip_x, flip_y,background_sampler,
                                object_sampler,boundary_sampler, mode='center'):
    min_mun = min(boundary_sampler[0].shape[0], object_sampler[0].shape[0])
    bgnum, bdnum, objectnum = int(min_mun*0.2), int(min_mun*0.3),int(min_mun*0.5)
    background_random = torch.randint(low=0, high=background_sampler[0].shape[0], size=[bgnum])
    boundary_random = torch.randint(low=0, high=boundary_sampler[0].shape[0], size=[bdnum])
    object_random = torch.randint(low=0, high=object_sampler[0].shape[0], size=[objectnum])
    background_y = background_sampler[0][background_random]
    background_x = background_sampler[1][background_random]
    boundary_y = boundary_sampler[0][boundary_random]
    boundary_x = boundary_sampler[1][boundary_random]
    object_y = object_sampler[0][object_random]
    object_x = object_sampler[1][object_random]

    x = torch.cat([background_x, boundary_x, object_x])
    y = torch.cat([background_y, boundary_y, object_y])

    if mode == 'lefttop':
        i, j = x, y
    elif mode == 'center':
        i, j = x+0.5, y+0.5
    elif mode == 'random':
        i = x+torch.rand_like(x)
        j = y+torch.rand_like(y)
    else:
        raise NotImplementedError

    if flip_x:
        i = i.flip((1,))
    if flip_y:
        j = j.flip((0,))
    if inverse_y:
        dirs = torch.stack([(i-K[0][2])/K[0][0], (j-K[1][2])/K[1][1], torch.ones_like(i)], -1)
    else:
        dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,3].expand(rays_d.shape)
    viewdirs = rays_d / rays_d.norm(dim=-1, keepdim=True)
    if ndc:
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)
    return rays_o, rays_d, viewdirs, [x.long(),y.long()]



def get_training_rays_semantic(rgb_tr_ori,mask_tr_ori,train_poses, HW, Ks, ndc, inverse_y, flip_x, flip_y,samplers):
    eps_time = time.time()
    DEVICE = rgb_tr_ori[0].device
    N = sum(im.shape[0] * im.shape[1] for im in rgb_tr_ori)
    rgb_tr = torch.zeros([N, 3], device=DEVICE)
    mask_tr = torch.zeros([N, 1], device=DEVICE)
    rays_o_tr = torch.zeros_like(rgb_tr)
    rays_d_tr = torch.zeros_like(rgb_tr)
    viewdirs_tr = torch.zeros_like(rgb_tr)
    imsz = []
    top = 0
    idx = 0
    for c2w, img, mask, (H, W), K in zip(train_poses, rgb_tr_ori,mask_tr_ori, HW, Ks):
        background = samplers['background'][idx]
        boundary = samplers['boundary'][idx]
        object = samplers['object'][idx]
        rays_o, rays_d, viewdirs, xy_index = get_rays_of_a_view_semantic(H=H, W=W, K=K, c2w=c2w, ndc=ndc,
            inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y, background_sampler=background,
            object_sampler=object, boundary_sampler=boundary)
        n = rays_o.shape[0]
        rgb_tr[top:top+n].copy_(img[xy_index[1],xy_index[0]])
        mask_tr[top:top+n].copy_(mask[xy_index[1],xy_index[0]])
        rays_o_tr[top:top+n].copy_(rays_o.to(DEVICE))
        rays_d_tr[top:top+n].copy_(rays_d.to(DEVICE))
        viewdirs_tr[top:top+n].copy_(viewdirs.to(DEVICE))
        imsz.append(n)
        top += n
        idx += 1
    rgb_tr = rgb_tr[:top]
    mask_tr = mask_tr[:top]
    rays_o_tr = rays_o_tr[:top]
    rays_d_tr = rays_d_tr[:top]
    viewdirs_tr = viewdirs_tr[:top]
    return rgb_tr, mask_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz

def get_training_rays(rgb_tr,mask_tr, train_poses, HW, Ks, ndc, inverse_y, flip_x, flip_y):
    # print('get_training_rays: start')
    assert len(np.unique(HW, axis=0)) == 1
    # assert len(np.unique(Ks.reshape(len(Ks),-1), axis=0)) == 1
    assert len(rgb_tr) == len(train_poses) and len(rgb_tr) == len(Ks) and len(rgb_tr) == len(HW)
    H, W = HW[0]
    K = Ks[0]
    eps_time = time.time()
    rays_o_tr = torch.zeros([len(rgb_tr), H, W, 3], device=rgb_tr.device)
    rays_d_tr = torch.zeros([len(rgb_tr), H, W, 3], device=rgb_tr.device)
    viewdirs_tr = torch.zeros([len(rgb_tr), H, W, 3], device=rgb_tr.device)
    imsz = [1] * len(rgb_tr)
    for i, c2w in enumerate(train_poses):
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
            H=H, W=W, K=K, c2w=c2w, ndc=ndc, inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y)
        rays_o_tr[i].copy_(rays_o.to(rgb_tr.device))
        rays_d_tr[i].copy_(rays_d.to(rgb_tr.device))
        viewdirs_tr[i].copy_(viewdirs.to(rgb_tr.device))
        del rays_o, rays_d, viewdirs
    # eps_time = time.time() - eps_time
    # print('get_training_rays: finish (eps time:', eps_time, 'sec)')
    return rgb_tr, mask_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz


def get_training_rays_flatten(rgb_tr_ori,mask_tr_ori,train_poses, HW, Ks, ndc, inverse_y, flip_x, flip_y):
    # 0.004 sec for 8 image
    # print('get_training_rays_flatten: start')
    assert len(rgb_tr_ori) == len(train_poses) and len(rgb_tr_ori) == len(Ks) and len(rgb_tr_ori) == len(HW)
    eps_time = time.time()
    DEVICE = rgb_tr_ori[0].device
    N = sum(im.shape[0] * im.shape[1] for im in rgb_tr_ori)
    rgb_tr = torch.zeros([N,3], device=DEVICE)
    mask_tr = torch.zeros([N,1], device=DEVICE)
    rays_o_tr = torch.zeros_like(rgb_tr)
    rays_d_tr = torch.zeros_like(rgb_tr)
    viewdirs_tr = torch.zeros_like(rgb_tr)
    imsz = []
    top = 0
    for c2w, img, mask, (H, W), K in zip(train_poses, rgb_tr_ori,mask_tr_ori, HW, Ks):
        assert img.shape[:2] == (H, W)
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
            H=H, W=W, K=K, c2w=c2w, ndc=ndc,
            inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y)
        n = H * W
        rgb_tr[top:top+n].copy_(img.flatten(0,1))
        mask_tr[top:top+n].copy_(mask.flatten(0,1))
        rays_o_tr[top:top+n].copy_(rays_o.flatten(0,1).to(DEVICE))
        rays_d_tr[top:top+n].copy_(rays_d.flatten(0,1).to(DEVICE))
        viewdirs_tr[top:top+n].copy_(viewdirs.flatten(0,1).to(DEVICE))
        imsz.append(n)
        top += n

    assert top == N
    eps_time = time.time() - eps_time
    # print('get_training_rays_flatten: finish (eps time:', eps_time, 'sec)')
    return rgb_tr, mask_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz


def get_training_rays_in_maskcache_sampling_grad(rgb_tr_ori, train_poses, HW, Ks, ndc, inverse_y, flip_x, flip_y, model, render_kwargs):
    # print('get_training_rays_in_maskcache_sampling: start')
    assert len(rgb_tr_ori) == len(train_poses) and len(rgb_tr_ori) == len(Ks) and len(rgb_tr_ori) == len(HW)
    CHUNK = 64
    DEVICE = rgb_tr_ori[0].device
    eps_time = time.time()
    N = sum(im.shape[0] * im.shape[1] for im in rgb_tr_ori)
    rgb_tr = torch.zeros([N, 3], device=DEVICE)
    rays_o_tr = torch.zeros_like(rgb_tr)
    rays_d_tr = torch.zeros_like(rgb_tr)
    viewdirs_tr = torch.zeros_like(rgb_tr)
    imsz = []
    top = 0
    for c2w, img, (H, W), K in zip(train_poses, rgb_tr_ori, HW, Ks):
        assert img.shape[:2] == (H, W)
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
            H=H, W=W, K=K, c2w=c2w, ndc=ndc,
            inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y)
        mask = torch.ones(img.shape[:2], device=DEVICE, dtype=torch.bool)
        for i in range(0, img.shape[0], CHUNK):
            rays_pts, mask_outbbox, _ = model.sample_ray_ori(
                rays_o=rays_o[i:i + CHUNK], rays_d=rays_d[i:i + CHUNK], **render_kwargs)
            mask_outbbox[~mask_outbbox] |= (~model.mask_cache(rays_pts[~mask_outbbox]))
            mask[i:i + CHUNK] &= (~mask_outbbox).any(-1).to(DEVICE)
        n = mask.sum()
        rgb_tr[top:top + n].copy_(img[mask])
        rays_o_tr[top:top + n].copy_(rays_o[mask].to(DEVICE))
        rays_d_tr[top:top + n].copy_(rays_d[mask].to(DEVICE))
        viewdirs_tr[top:top + n].copy_(viewdirs[mask].to(DEVICE))
        imsz.append(n)
        top += n
    rgb_tr = rgb_tr[:top]
    rays_o_tr = rays_o_tr[:top]
    rays_d_tr = rays_d_tr[:top]
    viewdirs_tr = viewdirs_tr[:top]
    return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz



@torch.no_grad()
def get_training_rays_in_maskcache_sampling(rgb_tr_ori, train_poses, HW, Ks, ndc, inverse_y, flip_x, flip_y, model, render_kwargs):
    print('get_training_rays_in_maskcache_sampling: start')
    assert len(rgb_tr_ori) == len(train_poses) and len(rgb_tr_ori) == len(Ks) and len(rgb_tr_ori) == len(HW)
    CHUNK = 64
    DEVICE = rgb_tr_ori[0].device
    eps_time = time.time()
    N = sum(im.shape[0] * im.shape[1] for im in rgb_tr_ori)
    rgb_tr = torch.zeros([N,3], device=DEVICE)
    rays_o_tr = torch.zeros_like(rgb_tr)
    rays_d_tr = torch.zeros_like(rgb_tr)
    viewdirs_tr = torch.zeros_like(rgb_tr)
    imsz = []
    top = 0
    for c2w, img, (H, W), K in zip(train_poses, rgb_tr_ori, HW, Ks):
        assert img.shape[:2] == (H, W)
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
            H=H, W=W, K=K, c2w=c2w, ndc=ndc,
            inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y)
        mask = torch.ones(img.shape[:2], device=DEVICE, dtype=torch.bool)
        for i in range(0, img.shape[0], CHUNK):
            rays_pts, mask_outbbox, _ = model.sample_ray_ori(
                rays_o=rays_o[i:i+CHUNK], rays_d=rays_d[i:i+CHUNK], **render_kwargs)
            mask_outbbox[~mask_outbbox] |= (~model.mask_cache(rays_pts[~mask_outbbox]))
            mask[i:i+CHUNK] &= (~mask_outbbox).any(-1).to(DEVICE)
        n = mask.sum()
        rgb_tr[top:top+n].copy_(img[mask])
        rays_o_tr[top:top+n].copy_(rays_o[mask].to(DEVICE))
        rays_d_tr[top:top+n].copy_(rays_d[mask].to(DEVICE))
        viewdirs_tr[top:top+n].copy_(viewdirs[mask].to(DEVICE))
        imsz.append(n)
        top += n

    print('get_training_rays_in_maskcache_sampling: ratio', top / N)
    rgb_tr = rgb_tr[:top]
    rays_o_tr = rays_o_tr[:top]
    rays_d_tr = rays_d_tr[:top]
    viewdirs_tr = viewdirs_tr[:top]
    eps_time = time.time() - eps_time
    # print('get_training_rays_in_maskcache_sampling: finish (eps time:', eps_time, 'sec)')
    return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz



def batch_indices_generator(N, BS):
    # torch.randperm on cuda produce incorrect results in my machine
    idx, top = torch.LongTensor(np.random.permutation(N)), 0
    while True:
        if top + BS > N:
            idx, top = torch.LongTensor(np.random.permutation(N)), 0
        yield idx[top:top+BS]
        top += BS


