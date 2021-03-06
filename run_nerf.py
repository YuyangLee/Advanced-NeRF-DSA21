import os
import sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from utils.visualization import graph_volume

from models.nerf import *
from models.octree import *
from models.scene import *
from models.volume import *

from utils.sampling import *

from utils.load_blender import load_blender_data
from torchvision.utils import save_image
from PIL import Image
from datetime import datetime
# import wandb

# wandb.init(project="NeRF-DSA21", entity="ap0str0ph3")

writer = None

date_unique_tag = datetime.now().strftime("%Y-%m-%d")
time_unique_tag = datetime.now().strftime("%H-%M-%S")

device_id = 0
torch.cuda.set_device(f"cuda:{device_id}")
device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn

    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    
    Arguments:
    `inputs`: batch_size x num_sample x 6
    """
    inputs_flat = inputs.view([-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs.unsqueeze(-2).expand(inputs.shape)
        input_dirs_flat = input_dirs.reshape([-1, input_dirs.shape[-1]])
        # input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    outputs = outputs_flat.view(list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs

def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i: i + chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, focal, chunk=1024*32, rays=None, cam_to_world=None, ndc=True,
           near=0., far=1.,
           use_viewdirs=False, cam_to_world_staticcam=None,
           **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      cam_to_world: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      cam_to_world_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other cam_to_world argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if cam_to_world is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, focal, cam_to_world)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if cam_to_world_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, focal, cam_to_world_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

    sh = rays_d.shape  # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, focal, 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()

    near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]



def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(
            args.multires_views, args.i_embed)
    output_ch = 5
    skips = [4]
    model = NeRF(D=args.netdepth, W=args.netwidth, input_ch=input_ch, output_ch=output_ch, skips=skips, input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    model_fine = None
    
    # wandb.watch(model)
    
    grad_vars = list(model.parameters())

    def network_query_fn(inputs, viewdirs, network_fn):
        return run_network(inputs, viewdirs, network_fn,
                           embed_fn=embed_fn,
                           embeddirs_fn=embeddirs_fn,
                           netchunk=args.netchunk)

    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth, W=args.netwidth, input_ch=input_ch, output_ch=output_ch, skips=skips, input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        grad_vars += list(model_fine.parameters())

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))
    
    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    # if args.ft_path is not None and args.ft_path != 'None':
    #     ckpts = [args.ft_path]
    # else:
    #     ckpts = [os.path.join(basedir, expname, f) for f in sorted(
    #         os.listdir(os.path.join(basedir, expname))) if 'tar' in f]
    print("Loading from pretrained...")
    
    ckpt = torch.load(args.load_from_snapshot)
    start = ckpt['global_step']
    model.load_state_dict(ckpt['network_fn_state_dict'])
    
    if args.N_importance > 0:
        if 'network_fine_state_dict' in ckpt.keys():
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])
        else:
            model_fine.load_state_dict(ckpt['network_fn_state_dict'])
    if not args.reconstruct_only:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        
    # print('Found ckpts', ckpts)
    # if len(ckpts) > 0 and not args.no_reload:
    #     ckpt_path = ckpts[-1]
    #     print('Reloading from', ckpt_path)
    #     ckpt = torch.load(ckpt_path)

    #     start = ckpt['global_step']
    #     optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    #     # Load model
    #     model.load_state_dict(ckpt['network_fn_state_dict'])
    #     if model_fine is not None:
    #         model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################

    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'perturb': args.perturb,
        'N_importance': args.N_importance,
        'network_fine': model_fine,
        'N_samples': args.N_samples,
        'network_fn': model,
        'use_viewdirs': args.use_viewdirs,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        # print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = { k: render_kwargs_train[k] for k in render_kwargs_train }
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer

def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False, verbose=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """

    dists = z_vals[..., 1:] - z_vals[..., :-1]  # [num_rays x (num_samples - 1)]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 3].shape) * raw_noise_std

    alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1. - alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[..., None])

    if verbose:
        return [rgb_map, disp_map, acc_map, depth_map], weights
    else:
        return rgb_map, disp_map, acc_map, depth_map, weights


def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:
        z_vals = near * (1. - t_vals) + far * (t_vals)
    else:
        z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

    mi = .5 * (z_vals[..., 1:] + z_vals[..., :-1])    # batch_size x (n_samples - 1)
    if perturb > 0.:
        # NOTE: Why sample like this:?
        # get intervals between samples
        hi = torch.cat([mi, z_vals[..., -1:]], -1)     # batch_size x n_samples
        lo = torch.cat([z_vals[..., :1], mi], -1)      # batch_size x n_samples
        
        # stratified samples in those intervals: unified sampling in each interval
        t_rand = torch.rand(z_vals.shape)
        z_vals = lo + (hi - lo) * t_rand
        
    # NOTE: Why expand here?
    z_vals = z_vals.expand([N_rays, N_samples])

    # batch_size x n_samples
    points = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]

    raw = network_query_fn(points, viewdirs, network_fn)
    
    rgb_map, disp_map, acc_map, depth_map, weights = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    # Hierarchical Sampling
    if N_importance > 0:
        # inverse transform sampling N_importance points
        bounds = torch.cat([z_vals[..., :1], mi, z_vals[..., :1]], dim=-1)
        
        # Sample N_f points from PDF of weights via inverse transform sampling
        z_vals_fine = its_from_weights(bounds, weights, N_importance)
        z_vals_fine, _ = torch.cat([z_vals, z_vals_fine], dim=-1).sort(dim=-1)
        points_fine = (rays_o[..., None, :] + rays_d[..., None, :] * z_vals_fine[..., :, None])
        
        # network query
        raw_fine = network_query_fn(points_fine, viewdirs, network_fine)
        
        maps_fine, _ = raw2outputs(raw_fine, z_vals_fine, rays_d, raw_noise_std, white_bkgd, pytest=pytest, verbose=True)
        rgb_map_fine, disp_map_fine, acc_map_fine, depth_map_fine = maps_fine

    ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map}
    if retraw:
        ret['raw'] = raw
        
    if N_importance > 0:
        ret.update(
            {
                'rgb_map_fine': rgb_map_fine,
                'disp_map_fine': disp_map_fine,
                'acc_map_fine': acc_map_fine,
                'depth_map_fine': depth_map_fine
            }
        )

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    # parser.add_argument("--expname", type=str,
    parser.add_argument("--expname", type=str, default="just_train_it",
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/nerf_synthetic/multihuman/',
                        help='input data directory')
    parser.add_argument("--load_from_snapshot", type=str, default="/home/yuyang/dev/Advanced-NeRF-DSA21/logs/pretrained/ptr.tar",
                        help="snapshot path")

    # training options
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8,
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=1024,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=500,
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true',
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=64,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', default=True,
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # reconstruction options
    parser.add_argument("--reconstruct_only", type=bool, default=False,
                        help="do not optimize, relaod weights and reconstruct the 3D scene")
    parser.add_argument("--reconstruct_mode", type=str, default='octree_lite',
                        help="reconstruct the 3D scene with various modees: direct | octree | octree_lite")
    parser.add_argument("--rec_export_filename", type=str, default="reconstruct.html",
                        help="filename of exported file of reconstruction")
    parser.add_argument("--reconstruct_range", type=list, default=[[-1, 1], [0, 2], [-1, 1]],
                        help="range of reconstruction")
    parser.add_argument("--reconstruct_resolution", type=float, default=2**(-7),
                        help="resolution of reconstruction")
    parser.add_argument("--reconstruct_sparse_resolution", type=float, default=2**(-6),
                        help="resolution of sparse query in octree-reconstruction")
    parser.add_argument("--reconstruct_oct_init_depth", type=int, default=3,
                        help="threshold for mean of sparse query in octree-reconstruction")
    parser.add_argument("--reconstruct_oct_threshold", type=float, default=0.01,
                        help="threshold for mean of sparse query in octree-reconstruction")
    parser.add_argument("--reconstruct_min_chunk_size", type=float, default=0.25,
                        help="threshold for minimum chunk size in octree-reconstruction")
    
    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='blender',
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    # deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek',
                        help='options : armchair / cube / greek / vase')

    # blender flags
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true',
                        help='load blender synthetic data at 400x400 instead of 800x800')

    # llff flags
    parser.add_argument("--factor", type=int, default=8,
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', default=True,
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true',
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true',
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8,
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=10000,
                        help='frequency of render_poses video saving')
    parser.add_argument("--i_reconstruct",   type=int, default=50000,
                        help='frequency of reconstruct 3d scene')

    return parser


# def render_vids_debug(render_poses, hwf, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):
#     H, W, focal = hwf
#     if render_factor != 0:
#         # Render downsampled for speed
#         H = H//render_factor
#         W = W//render_factor
#         focal = focal/render_factor
        
#     rgbs_gt = []
#     # disps_gt = []
#     rgbs = []
#     # disps = []
    
#     t = time.time()
#     for i, cam_to_world in enumerate(tqdm(render_poses)):
#         print(i, time.time() - t)
#         t = time.time()
#         rays_o, rays_d = get_rays(H, W, focal, torch.Tensor(cam_to_world))
#         batch_rays = torch.stack([rays_o, rays_d], dim=0)
#         rgb, _, _, _ = render(H, W, focal, chunk=args.chunk, rays=batch_rays, verbose=i < 10, retraw=True, **render_kwargs)
        
#         rgbs_gt.append(gt_imgs[i])
#         rgbs.append(rgb.cpu().numpy())
#         # disps.append(disp.cpu().numpy())
#         if i == 0:
#             print(rgb.shape)
            
#         if savedir is not None:
#             rgb8 = to8b(rgbs[-1])
#             filename = os.path.join(savedir, '{:03d}.png'.format(i))
#             imageio.imwrite(filename, rgb8)
            
#     imageio.mimwrite(os.path.join(savedir, 'rgb.mp4'), to8b(rgbs), fps=30, quality=8)
#     rgbs = np.stack(rgbs, 0)

#     return rgbs
    
def render_vids(render_poses, hwf, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):
    H, W, focal = hwf

    if render_factor != 0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []

    t = time.time()
    for i, cam_to_world in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        rgb, disp, acc, _ = render(H, W, focal, chunk=chunk, cam_to_world=cam_to_world[:3, :4], **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        if i == 0:
            print(rgb.shape, disp.shape)

        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps

def train(args):
    
    # Load data
    images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir)
    print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
    i_train, i_val, i_test = i_split

    near = 3.25
    far = 4.75

    if args.white_bkgd:
        images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
    else:
        images = images[..., :3]

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    os.makedirs(os.path.join(basedir, expname, "img", date_unique_tag, time_unique_tag), exist_ok=True)
    os.makedirs(os.path.join(basedir, expname, "reconstruct", date_unique_tag, time_unique_tag), exist_ok=True)
    
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, _, optimizer = create_nerf(args)
    
    global_step = start

    bds_dict = {
        'near': near,
        'far': far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)
    
    global writer
    writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))
    
    # Short circuit if only reconstruct the scene from trained model
    if args.reconstruct_only:
        reconstruct_kwargs = {
            "mode": args.reconstruct_mode,
            "export_filename": os.path.join(basedir, expname, f"{ args.reconstruct_mode }_{ args.rec_export_filename }"),
            # "export_filename": args.rec_export_filename,
            "rec_network": render_kwargs_train['network_fine'] if (args.N_importance > 0) else render_kwargs_train['network_fn'],
            "network_query_fn": render_kwargs_train['network_query_fn'],
            "range": args.reconstruct_range,
            "resolution": args.reconstruct_resolution,
            "min_chunk_size": args.reconstruct_min_chunk_size,
            "sparse_resolution": args.reconstruct_sparse_resolution,
            "threshold": args.reconstruct_oct_threshold,
            "init_depth": args.reconstruct_oct_init_depth
        }
        reconstruct(reconstruct_kwargs=reconstruct_kwargs)
        return

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = images[i_test]
            else:
                # Default is smoother render_poses path
                images = None

            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            rgbs, _ = render_vids(render_poses, hwf, args.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

            return

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching
    if use_batching:
        # For random ray batching
        print('Sampling rays')
        rays = np.stack([ get_rays_np(H, W, focal, cam_to_world_trans) for cam_to_world_trans in poses[:, :3, :4] ], 0)  # [N, ro+rd, H, W, 3]
        # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.concatenate([rays, images[:, None]], 1)
        # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])
        
        rays_rgb = rays_rgb[i_train]
        # rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0)  # train images only
        
        # Merge into batched rays: [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = np.reshape(rays_rgb, [-1, 3, 3]).astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgb)

        print('done')
        i_batch = 0

    # Move training data to GPU
    images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb).to(device)

    N_iters = 300000 + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    start = start + 1
    for i in trange(start, N_iters):
        time0 = time.time()

        # Sample random ray batch
        if use_batching:
            # Random over all images
            batch = rays_rgb[i_batch: i_batch + N_rand]  # [B, 2+1, 3*?]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0

        else:
            # Random from one image
            img_i = np.random.choice(i_train)
            target = images[img_i]
            pose = poses[img_i, :3, :4]

            if N_rand is not None:
                rays_o, rays_d = get_rays(H, W, focal, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

                coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)
                coords = torch.reshape(coords, [-1, 2])  # (H * W, 2)
                select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0)
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

        #####  Core optimization loop  #####
        rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, rays=batch_rays, verbose=i < 10, retraw=True, **render_kwargs_train)

        # Backword pass
        optimizer.zero_grad()
        
        img_loss = img2mse(rgb, target_s)
        loss = img_loss
        psnr = mse2psnr(img_loss)
    
        writer.add_scalar("Image MSE Loss - Coarse", loss, i)
        writer.add_scalar("Image Peak SNR - Coarse", psnr, i)

        if args.N_importance > 0:
            rgb_fine = extras['rgb_map_fine']
            disp_fine = extras['disp_map_fine']
            acc_fine = extras['acc_map_fine']
            depth_fine = extras['depth_map_fine']
            
            loss_fine = img2mse(rgb_fine, target_s)
            psnr_fine = mse2psnr(loss_fine)
            loss += loss_fine
                
            writer.add_scalar("Image MSE Loss - Fine", loss_fine, i)
            writer.add_scalar("Image Peak SNR - Fine", psnr_fine, i)
        
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
            
        loss.backward()
        optimizer.step()
            
        render_path = os.path.join(basedir, expname, "img", date_unique_tag, time_unique_tag, f"coarse-step_{i}.jpg")
        fine_path = os.path.join(basedir, expname, "img", date_unique_tag, time_unique_tag, f"fine-step_{i}.jpg")
        gt_path = os.path.join(basedir, expname, "img", date_unique_tag, time_unique_tag, f"gt-step_{i}.jpg")
        rc_path = os.path.join(basedir, expname, "reconstruct", date_unique_tag, time_unique_tag)
            
        dt = time.time() - time0
        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####

        # Rest is logging
        # if i % args.i_weights == 0:
        #     path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
        #     torch.save({
        #         'global_step': global_step,
        #         'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict() if args.N_importance > 0 else None
        #     }, path)
        #     print('Saved checkpoints at', path)

        if i % args.i_print == 0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")
            # print(expname, i, psnr.numpy(), loss.numpy(), global_step.numpy())
            # print('iter time {:.05f}'.format(dt))

        if i % args.i_img==0:
            img_i=np.random.choice(i_val)
            target_demo = images[img_i]
            pose_demo = poses[img_i, :3,:4]
            with torch.no_grad():
                rgb_demo, disp_demo, acc_demo, extras_demo = render(H, W, focal, chunk=args.chunk, cam_to_world=pose_demo, **render_kwargs_test)

            # psnr_demo = mse2psnr(img2mse(rgb_demo, target_demo))

            # Log a rendered validation view to Tensorboard
            target_demo_img = to8b(target_demo.clone().cpu().numpy())
            rgb_demo_img = to8b(rgb_demo.clone().cpu().numpy())
            
            writer.add_image('Target', hwc2chw(target_demo_img), i)
            writer.add_image('RGB', hwc2chw(rgb_demo_img), i)
            # writer.add_image('Disparity', disp_demo.clone().cpu().numpy(), i)
            # writer.add_image('Accept', acc_demo.clone().cpu().numpy(), i)
            Image.fromarray(target_demo_img).save(gt_path)
            Image.fromarray(rgb_demo_img).save(render_path)

            if args.N_importance > 0:
                fine_demo = extras_demo['rgb_map_fine']
                rgb_fine_demo_img = to8b(fine_demo.clone().cpu().numpy())
                writer.add_image('RGB - Fine', hwc2chw(rgb_fine_demo_img), i)
                Image.fromarray(rgb_fine_demo_img).save(fine_path)
                # writer.add_image('Disparity - Fine', extras_demo['disp_fine'].clone().cpu().numpy(), i)
                # writer.add_image('Accept - Fine', extras_demo['acc_fine'].clone().cpu().numpy(), i)

        # if i % args.i_reconstruct == 0:
        #     reconstruct(args, render_kwargs_train, os.path.join(rc_path, f"rec-{ i }.html"), i)

        # Something wrong with this function...
        # if i % args.i_video == 0 and i > 0:
        #     with torch.no_grad():
        #         directory = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
        #         os.makedirs(directory, exist_ok=True)
        #         rgbs, disps = render_vids_debug(render_poses, hwf, args.chunk, render_kwargs_test, gt_imgs=images, savedir=directory)
        #         # rgbs, disps = render_vids(render_poses, hwf, args.chunk, render_kwargs_test)
        #     print('Done, saving', rgbs.shape, disps.shape)
        #     moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
        #     imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
        #     imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

        #     if args.use_viewdirs:
        #         render_kwargs_test['cam_to_world_staticcam'] = render_poses[0][:3,:4]
        #         with torch.no_grad():
        #             rgbs_still, _ = render_vids(render_poses, hwf, args.chunk, render_kwargs_test)
        #         render_kwargs_test['cam_to_world_staticcam'] = None
        #         imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)

        # if i % args.i_testset == 0 and i > 0:
            # testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            # os.makedirs(testsavedir, exist_ok=True)
            # print('test poses shape', poses[i_test].shape)
            # with torch.no_grad():
            #     render_path(torch.Tensor(poses[i_test]).to(device), hwf, args.chunk, render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir)
            # print('Saved test set')

        global_step += 1

def reconstruct(reconstruct_kwargs):
    scene = Scene(
        oct_default_depth=reconstruct_kwargs['init_depth'],
        range=reconstruct_kwargs['range'],
        resolution=reconstruct_kwargs['resolution'],
        mode=reconstruct_kwargs['mode']
    )
    
    volume, time_s, time_r, blocks = scene.reconstruct_volume(reconstruct_kwargs)
    wo = "w/o" if reconstruct_kwargs['mode'] == "direct" else "w/"
    
    print(f"""
---------------------- Report ----------------------
Reconstructed with resolution { int((1 / reconstruct_kwargs['resolution'])) } x { int((1 / reconstruct_kwargs['resolution'])) } x { int((1 / reconstruct_kwargs['resolution'])) }
{ wo } OCTree, in mode { reconstruct_kwargs['mode'] }
Search: Searched in { str(time_s)[:5] } s
Recon.: Inferred in { str(time_r)[:5] } s through { blocks } blocks
Total Time Consum.: { str(time_s + time_r)[:5] } s
----------------------------------------------------
""")
    fig = graph_volume(volume, reconstruct_kwargs['export_filename'])

def reconstruction_benchtest(args):
    print("Benchtest for 3D Scene Reconstruction")
    
    # Debug params
    args.expname = "reconstruct_benchtest"
    # args.load_from_snapshot = "/home/yuyang/dev/Advanced-NeRF-DSA21/logs/pretrained/100000.tar"
    args.load_from_snapshot = "/home/yuyang/dev/Advanced-NeRF-DSA21/logs/pretrained/ptr.tar"
    args.render_only = False
    args.reconstruct_only = True
    
    # Reconstruction params
    # args.reconstruct_resolution = 2**(-7)
    args.reconstruct_min_chunk_size = 2**(-2)
    args.reconstruction_sparse_resolution = 2**(-5)
    args.reconstruction_mean_threshold = 0.025
    args.reconstruct_oct_init_depth = 1
    
    # Special: for octree vs octree_lite comparison
    # args.reconstruct_resolution = 2**(-9)
    # args.reconstruct_min_chunk_size = 2**(-5)
    
    args.reconstruct_with_octree = True
    args.reconstruct_range = [[-1., 1.], [0., 2.], [-1., 1.]]
    
    args.reconstruct_mode = "direct"
    train(args)
    
    # Use OCTree
    args.reconstruct_mode = "octree"
    train(args)
    
    # Use OCTree Light
    args.reconstruct_mode = "octree_lite"
    train(args)

if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    parser = config_parser()
    args = parser.parse_args()
    
    # Debug and export arguments
    args.render_only = False
    args.expname = "THATS_NOT_MY_NAME"
    # args.load_from_snapshot = "/home/yuyang/dev/Advanced-NeRF-DSA21/logs/pretrained/100000.tar"
    args.load_from_snapshot = "/home/yuyang/dev/Advanced-NeRF-DSA21/logs/just_train_it/200000.tar"
    
    # Training arguments
    args.no_batching = True
    args.use_viewdirs = True
    args.white_bkgd = True
    args.lrate_decay = 500
    args.N_samples = 64
    args.N_importrance = 64
    args.N_rand = 1024
    args.half_res = True
    args.testskip = 1
    args.lrate=1e-4
    args.i_weights=1000
    args.i_reconstruct = 500
    args.half_res = True
    
    # reconstruction_benchtest(args)
    train(args)
