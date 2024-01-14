import cv2, os, sys
import imageio.v2 as imageio
import numpy as np
import random
import scipy.signal
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch import nn
import statsmodels.api as sm
from scripts.Warper import Warper

mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))


def set_seed(seed, base=0, is_set=True):
    seed += base
    assert seed >= 0, '{} >= {}'.format(seed, 0)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

class TransMittanceLoss(nn.Module):
    def __init__(self):
        super(TransMittanceLoss, self).__init__()
        print('Using TransMittance Loss ')

        self.init_eta = 0.88
        self.decay = 0.01
        self.loss = torch.nn.MSELoss(reduction='mean')

    def forward(self, transmittance, epoch):
        # transmittance (XXX, 128 + 64)
        print ("transmittance: ", transmittance.shape)
        # save npy
        with open("transmittance_recon_wangcan.npy", 'wb') as fw:
            np.save(fw, transmittance.detach().cpu().numpy())
        self.init_eta = self.init_eta - self.decay * epoch
        target = torch.tensor(self.init_eta).cuda()
        mean_trans = torch.mean(transmittance, dim=1) # XXX
        loss = self.loss(mean_trans, target)
        return loss

class TransMittanceLoss_const(nn.Module):
    def __init__(self, device):
        super(TransMittanceLoss_const, self).__init__()
        self.init_eta = 0.8
        self.decay = 0.01
        self.device = device
        self.loss = torch.nn.MSELoss(reduction='mean')

    def forward(self, transmittance):
        # transmittance (XXX, 128 + 64)
        # self.init_eta = self.init_eta - self.decay * epoch
        target = torch.tensor(self.init_eta).repeat(transmittance.shape[0]).to(self.device)
        mean_trans = torch.mean(transmittance, dim=1) # XXX
        loss = self.loss(mean_trans, target)
        return loss
        # target2 = torch.tensor(.0).repeat(transmittance.shape[0]).to(self.device)
        # mean_trans2 = torch.mean(transmittance[:, :10], dim=1)
        # return loss + self.loss(mean_trans2, target2)

class TransMittanceLoss_mask(nn.Module):
    def __init__(self, device):
        super(TransMittanceLoss_mask, self).__init__()
        self.init_eta = .0
        self.device = device
        self.loss = torch.nn.MSELoss(reduction='mean')

    def forward(self, transmittance, mask):
        # transmittance (XXX, n_points)
        target = torch.tensor(self.init_eta).repeat(transmittance.shape[0]).to(self.device)
        mean_trans = torch.mean(transmittance*mask, dim=1) # XXX
        loss = self.loss(mean_trans, target)
       
        return loss

# warping: multiple view to one target view
def bilinear_splat_warping_multiview(rgbs, depths, poses, pose_tar, H, W, intrinsic, masks=None):
    """
    rgbs: list, range [0, 1]
    depths: list
    """
    num_views = len(rgbs)
    warper = Warper()
    transformation2 = np.linalg.inv(pose_tar)
    intrinsic_mtx = np.eye(3).astype(np.float32)
    intrinsic_mtx[0, 0] = intrinsic[0]
    intrinsic_mtx[1, 1] = intrinsic[1]
    intrinsic_mtx[0, 2] = intrinsic[2]
    intrinsic_mtx[1, 2] = intrinsic[3]
    
    mask_final = np.zeros((H, W), dtype=np.uint8)
    output_image = np.zeros((H, W, 3), dtype=np.uint8)
    output_depth = np.zeros((H, W))
    for vv in range(num_views):
        rgb_src = (rgbs[vv] * 255).astype(np.uint8)
        depth_src = depths[vv]
        mask_src = masks[vv] if masks is not None else masks
        transformation1 = np.linalg.inv(poses[vv])
        warped_frame2, mask2, warped_depth2, flow12 = warper.forward_warp(rgb_src, mask_src, depth_src, transformation1, transformation2, intrinsic_mtx, None)

        mask_new = mask2.copy()
        mask_new[mask_final>0] = 0
        for i in range(3):
            output_image[:,:,i] = output_image[:,:,i] * mask_final + warped_frame2[:,:,i] * mask_new
        output_depth = output_depth * mask_final + warped_depth2 * mask_new
        mask_merge = mask_final + mask2
        mask_final = (mask_merge>0) * 1   # update mask_final
        
    # obtain white background
    for i in range(3):
        output_image[:,:,i] = np.array(output_image[:,:,i]*mask_final + 255*(1-mask_final))
    output_image = (output_image/255).astype(np.float32)
    return mask_final, output_image, output_depth

# warping: one view to multiple views
def gt_warping(rgb_gt, depth_gt, pose_gt, poses_tar, H, W, intrinsic, logpath=None, mask_gt=None, warp_depth=False, bilinear_splat=False):
    """
    poses_tar: [n_views_tar, 4, 4]
    """
    if logpath is not None:
        # save path of warped image
        save_path_warp = os.path.join(logpath, "DIBR_gt")
        os.makedirs(os.path.join(save_path_warp, 'warped'), exist_ok=True)
        os.makedirs(os.path.join(save_path_warp, 'mask'), exist_ok=True)
        os.makedirs(os.path.join(save_path_warp, 'mask_inv'), exist_ok=True)
        if warp_depth:
            os.makedirs(os.path.join(save_path_warp, 'warped_depth'), exist_ok=True)

    rgbs_warp, masks_warp, depths_warp = [], [], []
    if bilinear_splat:
        warper = Warper()
        transformation1 = np.linalg.inv(pose_gt)
        intrinsic_mtx = np.eye(3).astype(np.float32)
        intrinsic_mtx[0, 0] = intrinsic[0]
        intrinsic_mtx[1, 1] = intrinsic[1]
        intrinsic_mtx[0, 2] = intrinsic[2]
        intrinsic_mtx[1, 2] = intrinsic[3]
        rgb_src = (rgb_gt * 255).astype(np.uint8)
        poses_np = poses_tar
        for vv in range(poses_np.shape[0]):
            transformation2 = np.linalg.inv(poses_np[vv])
            warped_frame2, mask2, warped_depth2, flow12 = warper.forward_warp(rgb_src, mask_gt, depth_gt, transformation1, transformation2, intrinsic_mtx, None)
            # obtain white background
            for i in range(3):
                warped_frame2[:,:,i] = np.array(warped_frame2[:,:,i]*mask2+255*(1-mask2))
            rgbs_warp.append((warped_frame2/255).astype(np.float32))
            masks_warp.append(mask2*1)
            if warp_depth:
                depths_warp.append(warped_depth2)
                if logpath is not None:
                    imageio.imwrite(os.path.join(save_path_warp, 'warped_depth', '%05d.png' % (vv+1)), depths_warp[-1])
            mask_image = (mask2 * 255).astype(np.uint8)
            mask_inv = ((1 - mask2) * 255).astype(np.uint8)
            if logpath is not None:
                imageio.imwrite(os.path.join(save_path_warp, 'warped', '%05d.png' % (vv+1)), warped_frame2)
                imageio.imwrite(os.path.join(save_path_warp, 'mask', '%05d.png' % (vv+1)), mask_image)
                imageio.imwrite(os.path.join(save_path_warp, 'mask_inv', '%05d.png' % (vv+1)), mask_inv)
    else:
        use_filter_filling = True
        # move data to cpu & numpy
        poses_np = poses_tar
        fx, fy, cx, cy = intrinsic
        img = rgb_gt
        depth = depth_gt
        pose_src = pose_gt
        
        for vv in range(poses_np.shape[0]):
            myMap = np.zeros((H, W), dtype=np.uint8)
            output_image = np.ones((H, W, 3))
            points_warped = np.zeros((H, W, 3))

            ## warp existing views to target view through DIBR
            tar_w2c = np.linalg.inv(poses_np[vv])
            # project
            y = np.linspace(0, H - 1, H)
            x = np.linspace(0, W - 1, W)
            xx, yy = np.meshgrid(x, y)
            # coordinate in cam1
            x = (xx - cx) / fx * depth
            y = (yy - cy) / fy * depth
            coords = np.zeros((H, W, 4))
            coords[:, :, 0] = x
            coords[:, :, 1] = y
            coords[:, :, 2] = depth
            coords[:, :, 3] = 1
            coords_c1 = coords.transpose(2, 0, 1).reshape(4, -1)
            coords_c2 = np.matmul(np.dot(tar_w2c, pose_src), coords_c1)
            coords_c2 = coords_c2.reshape(4, H, W).transpose(1, 2, 0)
            z_tar = coords_c2[:, :, 2]
            x = coords_c2[:, :, 0] / (1e-8 + z_tar) * fx + cx
            y = coords_c2[:, :, 1] / (1e-8 + z_tar) * fy + cy

            # Round off the pixels in new virutal image and fill cracks with white
            x = (np.round(x)).astype(np.int16)
            y = (np.round(y)).astype(np.int16)

            # Keeping track of already seen points, a better way would have been use of HashMap for saving space
            for i in range(H):
                for j in range(W):
                    x_o = x[i, j]
                    y_o = y[i, j]
                    z_o = z_tar[i, j]
                    if (x_o >= 0 and x_o < W and y_o >= 0 and y_o < H and z_o > 0):# and coords_c2[i, j, 2]>0:
                        if (myMap[y_o, x_o] == 0):
                            output_image[y_o, x_o, :] = img[i, j, :]
                            points_warped[y_o, x_o, :] = coords_c2[i, j, :3]
                            myMap[y_o, x_o] = 1
            if use_filter_filling and not warp_depth:
                output_image, myMap_filt = dibr_filter_mask(output_image, myMap)
            else:
                myMap_filt = myMap
            rgbs_warp.append(output_image)
            masks_warp.append(myMap_filt)
        
            if warp_depth:
                depths_warp.append(points_warped[:,:,2])
                # output_depth, myMap_filt = dibr_filter_mask(depths_warp[-1][:,:,np.newaxis], myMap)
                if logpath is not None:
                    imageio.imwrite(os.path.join(save_path_warp, 'warped_depth', '%05d.png' % (vv+1)), depths_warp[-1])

            output_image = (output_image * 255).astype(np.uint8)
            mask_image = (myMap_filt * 255).astype(np.uint8)
            mask_inv = ((1 - myMap_filt) * 255).astype(np.uint8)
            if logpath is not None:
                imageio.imwrite(os.path.join(save_path_warp, 'warped', '%05d.png' % (vv+1)), output_image)
                imageio.imwrite(os.path.join(save_path_warp, 'mask', '%05d.png' % (vv+1)), mask_image)
                imageio.imwrite(os.path.join(save_path_warp, 'mask_inv', '%05d.png' % (vv+1)), mask_inv)

    if warp_depth:
        return np.stack(rgbs_warp), np.stack(masks_warp), np.stack(depths_warp)
    else:
        return np.stack(rgbs_warp), np.stack(masks_warp)


def visualize_depth_numpy(depth, minmax=None, colorize=True, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """

    x = np.nan_to_num(depth) # change nan to 0
    if minmax is None:
        mi = np.min(x[x>0]) # get minimum positive depth (ignore background)
        ma = np.max(x)
    else:
        mi,ma = minmax

    x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
    x = np.maximum(x, 0)
    x = (255*x).astype(np.uint8)
    if colorize:
        x_ = cv2.applyColorMap(x, cmap)
    else:
        x = np.nan_to_num(depth)
        mi = np.min(x[x>0])
        ma = np.max(x)
        x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
        x = (255*x).astype(np.uint8)
        x_ = x[:,:,np.newaxis].repeat(3, 2)
    return x_, [mi,ma]

def init_log(log, keys):
    for key in keys:
        log[key] = torch.tensor([0.0], dtype=float)
    return log

def visualize_depth(depth, minmax=None, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    if type(depth) is not np.ndarray:
        depth = depth.cpu().numpy()

    x = np.nan_to_num(depth) # change nan to 0
    if minmax is None:
        mi = np.min(x[x>0]) # get minimum positive depth (ignore background)
        ma = np.max(x)
    else:
        mi,ma = minmax

    x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
    x = (255*x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_)  # (3, H, W)
    return x_, [mi,ma]

def N_to_reso(n_voxels, bbox):
    xyz_min, xyz_max = bbox
    dim = len(xyz_min)
    voxel_size = ((xyz_max - xyz_min).prod() / n_voxels).pow(1 / dim)
    return ((xyz_max - xyz_min) / voxel_size).long().tolist()

def cal_n_samples(reso, step_ratio=0.5):
    return int(np.linalg.norm(reso)/step_ratio)

def is_not_in_expected_distribution(depth_mean, depth_var, depth_measurement_mean, depth_measurement_std):
    delta_greater_than_expected = ((depth_mean - depth_measurement_mean).abs() - depth_measurement_std) > 0.
    var_greater_than_expected = depth_measurement_std**2 < depth_var
    return torch.logical_or(delta_greater_than_expected, var_greater_than_expected)

def compute_depth_loss(depth_map, z_vals, weights, target_depth, target_std=0.1):
    pred_mean = depth_map
    if pred_mean.shape[0] == 0:
        return torch.zeros((1,), device=depth_map.device, requires_grad=True)
    pred_var = ((z_vals - pred_mean.unsqueeze(-1)).pow(2) * weights).sum(-1) + 1e-8
    target_mean = target_depth
    apply_depth_loss = is_not_in_expected_distribution(pred_mean, pred_var, target_mean, target_std)
    pred_mean = pred_mean[apply_depth_loss]
    if pred_mean.shape[0] == 0:
        return torch.zeros((1,), device=depth_map.device, requires_grad=True)
    pred_var = pred_var[apply_depth_loss]
    target_mean = target_mean[apply_depth_loss]
    # https://pytorch.org/docs/stable/generated/torch.nn.GaussianNLLLoss.html
    f = nn.GaussianNLLLoss(eps=0.001)
    return torch.abs(f(pred_mean, target_mean, pred_var))
    # return f(pred_mean, target_mean, pred_var)


def compute_depth_loss_scale_invariant(depth_map, target_depth):
    num = target_depth.shape[0]
    log_pred = torch.log(depth_map)
    log_target = torch.log(target_depth)
    alpha = (log_target - log_pred).sum()/num
    log_diff = torch.abs((log_pred - log_target + alpha))
    # d = 0.05*0.2*(log_diff.sum()/num)
    return log_diff.sum()/num

def scale_shift_invariant_loss(z_vals, weights, target_depth):
    target_depth = target_depth[:,None].repeat([1, z_vals.shape[1]])
    z_vals_ = sm.add_constant(z_vals.view(-1).cpu().numpy())
    w_ = weights.view(-1).detach().cpu().numpy()
    tar_ = target_depth.view(-1).detach().cpu().numpy()
    mod_wls = sm.WLS(tar_, z_vals_, weights=w_)
    res_wls = mod_wls.fit()
    t, s = res_wls.params
    loss = torch.mean(weights*(s*z_vals+t - target_depth)**2)
    return loss, s, t


def dibr_filter_mask(output_image, myMap):
    H, W, C = output_image.shape
    weights0 = np.array([[1,1,1.5,1,1],[1,1.5,3,1.5,1],[1.5,3,0,3,1.5],[1,1.5,3,1.5,1],[1,1,1.5,1,1]], dtype=np.float32)
    sum_weight0 = np.sum(weights0)
    for i in range(2, H-2):
        for j in range(2, W-2):
            if myMap[i, j] == 0 and np.sum(myMap[i-2:i+3, j-2:j+3]*weights0)/sum_weight0>0.6:
                for cc in range(C):
                    output_image[i, j, cc] = np.sum(output_image[i-1:i+2, j-1:j+2, cc] * myMap[i-1:i+2, j-1:j+2])/np.sum(myMap[i-1:i+2, j-1:j+2])
                myMap[i, j] = 1
    
    weights = np.array([[1,3,1],[3,0,3],[1,3,1]], dtype=np.uint8)
    sum_weight = np.sum(weights)
    for i in range(1, H-1):
        for j in range(1, W-1):
            if myMap[i, j] == 0 and np.sum(myMap[i-1:i+2, j-1:j+2]*weights)/sum_weight>0.5:
                for cc in range(C):
                    output_image[i, j, cc] = np.sum(output_image[i-1:i+2, j-1:j+2, cc] * myMap[i-1:i+2, j-1:j+2])/np.sum(myMap[i-1:i+2, j-1:j+2])
                myMap[i, j] = 1
    i = 0
    for j in range(W):
        if myMap[i, j] == 0 and myMap[i+1, j] > 0:
            output_image[i, j, :] = output_image[i+1, j, :].copy()
            myMap[i, j] = 1
    i = H-1
    for j in range(W):
        if myMap[i, j] == 0 and myMap[i-1, j] > 0:
            output_image[i, j, :] = output_image[i-1, j, :].copy()
            myMap[i, j] = 1
    j = 0
    for i in range(H):
        if myMap[i, j] == 0 and myMap[i, j+1] > 0:
            output_image[i, j, :] = output_image[i, j+1, :].copy()
            myMap[i, j] = 1
    j = W-1
    for i in range(H):
        if myMap[i, j] == 0 and myMap[i, j-1] > 0:
            output_image[i, j, :] = output_image[i, j-1, :].copy()
            myMap[i, j] = 1
    for i in range(1, H-1):
        for j in range(1, W-1):
            if myMap[i, j] == 1 and np.sum(myMap[i-1:i+2, j-1:j+2]*weights)/sum_weight<0.45:
                for cc in range(C):
                    output_image[i, j, cc] = 255
                myMap[i, j] = 0
    
    return output_image, myMap


def dibr_filter_mask2(output_image, myMap, output_depth=None):
    H, W, C = output_image.shape
    weights0 = np.array([[1,1,1.5,1,1],[1,1.5,3,1.5,1],[1.5,3,0,3,1.5],[1,1.5,3,1.5,1],[1,1,1.5,1,1]], dtype=np.float32)
    sum_weight0 = np.sum(weights0)
    for i in range(2, H-2):
        for j in range(2, W-2):
            if myMap[i, j] == 0 and np.sum(myMap[i-2:i+3, j-2:j+3]*weights0)/sum_weight0>0.65:
                for cc in range(C):
                    output_image[i, j, cc] = np.sum(output_image[i-1:i+2, j-1:j+2, cc] * myMap[i-1:i+2, j-1:j+2])/np.sum(myMap[i-1:i+2, j-1:j+2])
                if output_depth is not None:
                    output_depth[i, j] = np.sum(output_depth[i-1:i+2, j-1:j+2] * myMap[i-1:i+2, j-1:j+2])/np.sum(myMap[i-1:i+2, j-1:j+2])
                myMap[i, j] = 1
    if output_depth is not None:
        return output_image, myMap, output_depth
    else:
        return output_image, myMap


__LPIPS__ = {}
def init_lpips(net_name, device):
    assert net_name in ['alex', 'vgg']
    import lpips
    print(f'init_lpips: lpips_{net_name}')
    return lpips.LPIPS(net=net_name, version='0.1').eval().to(device)

def rgb_lpips(np_gt, np_im, net_name, device):
    if net_name not in __LPIPS__:
        __LPIPS__[net_name] = init_lpips(net_name, device)
    gt = torch.from_numpy(np_gt).permute([2, 0, 1]).contiguous().to(device)
    im = torch.from_numpy(np_im).permute([2, 0, 1]).contiguous().to(device)
    return __LPIPS__[net_name](gt, im, normalize=True).item()


def findItem(items, target):
    for one in items:
        if one[:len(target)]==target:
            return one
    return None


''' Evaluation metrics (ssim, lpips)
'''
def rgb_ssim(img0, img1, max_val,
             filter_size=11,
             filter_sigma=1.5,
             k1=0.01,
             k2=0.03,
             return_map=False):
    # Modified from https://github.com/google/mipnerf/blob/16e73dfdb52044dcceb47cda5243a686391a6e0f/internal/math.py#L58
    assert len(img0.shape) == 3
    assert img0.shape[-1] == 3
    assert img0.shape == img1.shape

    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((np.arange(filter_size) - hw + shift) / filter_sigma)**2
    filt = np.exp(-0.5 * f_i)
    filt /= np.sum(filt)

    # Blur in x and y (faster than the 2D convolution).
    def convolve2d(z, f):
        return scipy.signal.convolve2d(z, f, mode='valid')

    filt_fn = lambda z: np.stack([
        convolve2d(convolve2d(z[...,i], filt[:, None]), filt[None, :])
        for i in range(z.shape[-1])], -1)
    mu0 = filt_fn(img0)
    mu1 = filt_fn(img1)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(img0**2) - mu00
    sigma11 = filt_fn(img1**2) - mu11
    sigma01 = filt_fn(img0 * img1) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    sigma00 = np.maximum(0., sigma00)
    sigma11 = np.maximum(0., sigma11)
    sigma01 = np.sign(sigma01) * np.minimum(
        np.sqrt(sigma00 * sigma11), np.abs(sigma01))
    c1 = (k1 * max_val)**2
    c2 = (k2 * max_val)**2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom
    ssim = np.mean(ssim_map)
    return ssim_map if return_map else ssim


import torch.nn as nn


class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]



import plyfile
import skimage.measure


def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    ply_filename_out,
    bbox,
    level=0.5,
    offset=None,
    scale=None,
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()
    voxel_size = list((bbox[1]-bbox[0]) / np.array(pytorch_3d_sdf_tensor.shape))

    verts, faces, normals, values = skimage.measure.marching_cubes(
        numpy_3d_sdf_tensor, level=level, spacing=voxel_size
    )
    faces = faces[...,::-1] # inverse face orientation

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = bbox[0,0] + verts[:, 0]
    mesh_points[:, 1] = bbox[0,1] + verts[:, 1]
    mesh_points[:, 2] = bbox[0,2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    print("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)
