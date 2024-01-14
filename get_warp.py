import cv2, os
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
