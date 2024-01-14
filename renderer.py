import os
import pdb
import sys

import imageio
import torch
from tqdm.auto import tqdm

from dataLoader.ray_utils import get_rays, ndc_rays_blender
from models.tensoRF import (AlphaGridMask, TensorCP, TensorVM, TensorVMSplit,
                            raw2alpha)
from utils import *

class SimpleSampler:
    def __init__(self, total, batch):
        self.total = total
        self.batch = batch
        self.curr = total
        self.ids = None

    def nextids(self):
        self.curr+=self.batch
        if self.curr + self.batch > self.total:
            self.ids = torch.LongTensor(np.random.permutation(self.total))
            self.curr = 0
        return self.ids[self.curr:self.curr+self.batch]

def OctreeRender_trilinear_fast(rays, tensorf, chunk=4096, N_samples=-1, ndc_ray=False, white_bg=True, is_train=False, device='cuda'):

    rgbs, alphas, depth_maps, weights, uncertainties, z_val = [], [], [], [], [], []
    N_rays_all = rays.shape[0]
    for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
        rays_chunk = rays[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(device)
    
        rgb_map, depth_map, z_vals, weight = tensorf(rays_chunk, is_train=is_train, white_bg=white_bg, ndc_ray=ndc_ray, N_samples=N_samples)

        rgbs.append(rgb_map)
        depth_maps.append(depth_map)
        weights.append(weight)
        z_val.append(z_vals)
    
    return torch.cat(rgbs), None, torch.cat(depth_maps), torch.cat(weights), torch.cat(z_val)

@torch.no_grad()
def evaluation(test_dataset, tensorf, args, renderer, savePath=None, N_vis=5, 
               prtx='', N_samples=-1, white_bg=True, ndc_ray=False, compute_extra_metrics=True, 
               device='cuda', video_gen=False, N_iter=-1, preview=False, stitching=True):
    """
    preview: if True, rendering in support views; if False, rendering in training views (inpainting views)
    """
    
    PSNRs, rgb_maps, depth_maps = [], [], []
    ssims,l_alex,l_vgg=[],[],[]
    stitching=False
    if video_gen:
        stitching=False
    if savePath is not None:
        os.makedirs(savePath, exist_ok=True)
        if not stitching:
            os.makedirs(os.path.join(savePath, 'rgbs'), exist_ok=True)
            os.makedirs(os.path.join(savePath, 'depths'), exist_ok=True)

    if test_dataset.split == 'train':
        if preview:
            all_rays = test_dataset.all_rays_sprt_split
            all_rgbs = None   # test_dataset.all_rgbs_sprt_split
        else:
            all_rays = test_dataset.all_rays_gen_split[:N_iter+1]
            all_rgbs = test_dataset.all_rgbs_gen_split[:N_iter+1]
    elif test_dataset.split == 'test' and N_iter>=0:
        all_rays = test_dataset.all_rays_split[:N_iter+1]
        all_rgbs = None
    else:
        all_rays = test_dataset.all_rays_split
        all_rgbs = None

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    img_eval_interval = 1 if N_vis < 0 else max(all_rays.shape[0] // N_vis,1)
    idxs = list(range(0, all_rays.shape[0], img_eval_interval))
    for idx, samples in tqdm(enumerate(all_rays[0::img_eval_interval]), file=sys.stdout):
        W, H = test_dataset.img_wh
        rays = samples.view(-1,samples.shape[-1])

        rgb_map, _, depth_map, _, _ = renderer(rays, tensorf, chunk=args.batch_size, N_samples=N_samples,
                                        ndc_ray=ndc_ray, white_bg = white_bg, device=device)
        
        rgb_map = rgb_map.clamp(0.0, 1.0)
        rgb_map, depth_map = rgb_map.reshape(H, W, 3).cpu(), depth_map.reshape(H, W).cpu()
        depth_map = depth_map - args.push_depth + 0.8
        depth_map = np.maximum(depth_map.numpy(), 0)
        depth_map, _ = visualize_depth_numpy(depth_map, near_far, colorize=True)

        if all_rgbs is not None and compute_extra_metrics:
            gt_rgb = all_rgbs[idxs[idx]].view(H, W, 3)
            loss = torch.mean((rgb_map - gt_rgb) ** 2)
            PSNRs.append(-10.0 * np.log(loss.item()) / np.log(10.0))

            if compute_extra_metrics:
                ssim = rgb_ssim(rgb_map, gt_rgb, 1)
                l_a = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'alex', tensorf.device)
                l_v = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'vgg', tensorf.device)
                ssims.append(ssim)
                l_alex.append(l_a)
                l_vgg.append(l_v)

        rgb_map = (rgb_map.numpy() * 255).astype('uint8')
        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)
        if savePath is not None:
            if stitching and all_rgbs is not None:
                gt_rgb = (gt_rgb.numpy() * 255).astype('uint8')
                rgb_map = np.concatenate((rgb_map, depth_map, gt_rgb), axis=1)
                imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', rgb_map)
            elif stitching and all_rgbs is None:
                rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
                imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', rgb_map)
            else:
                imageio.imwrite(f'{savePath}/rgbs/{prtx}{idx:03d}_rgb.png', rgb_map)
                imageio.imwrite(f'{savePath}/depths/{prtx}{idx:03d}_depth.png', depth_map)

    if video_gen:
        imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(rgb_maps), fps=30, quality=9)
        imageio.mimwrite(f'{savePath}/{prtx}depthvideo.mp4', np.stack(depth_maps), fps=30, quality=9)

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            # np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr, ssim, l_a, l_v]))
        # else:
            # np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr]))

    return PSNRs

@torch.no_grad()
def evaluation_path(test_dataset,tensorf, c2ws, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
                    white_bg=False, ndc_ray=False, compute_extra_metrics=True, device='cuda'):
    PSNRs, rgb_maps, depth_maps = [], [], []
    ssims,l_alex,l_vgg=[],[],[]
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath+"/rgbd", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    for idx, c2w in tqdm(enumerate(c2ws)):

        W, H = test_dataset.img_wh

        c2w = torch.FloatTensor(c2w)
        rays_o, rays_d = get_rays(test_dataset.directions, c2w)  # both (h*w, 3)
        if ndc_ray:
            rays_o, rays_d = ndc_rays_blender(H, W, test_dataset.focal[0], 1.0, rays_o, rays_d)
        rays = torch.cat([rays_o, rays_d], 1)  # (h*w, 6)

        rgb_map, _, depth_map, _, _ = renderer(rays, tensorf, chunk=8192, N_samples=N_samples,
                                        ndc_ray=ndc_ray, white_bg = white_bg, device=device)
        rgb_map = rgb_map.clamp(0.0, 1.0)

        rgb_map, depth_map = rgb_map.reshape(H, W, 3).cpu(), depth_map.reshape(H, W).cpu()

        depth_map, _ = visualize_depth_numpy(depth_map.numpy(),near_far)

        rgb_map = (rgb_map.numpy() * 255).astype('uint8')
        # rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)
        if savePath is not None:
            imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', rgb_map)
            rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
            imageio.imwrite(f'{savePath}/rgbd/{prtx}{idx:03d}.png', rgb_map)

    imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(rgb_maps), fps=30, quality=8)
    imageio.mimwrite(f'{savePath}/{prtx}depthvideo.mp4', np.stack(depth_maps), fps=30, quality=8)

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr, ssim, l_a, l_v]))
        else:
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr]))


    return PSNRs

