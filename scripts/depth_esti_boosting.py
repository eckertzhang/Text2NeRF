import argparse
import json
import os
import pdb
import sys
import time
import warnings
from operator import getitem

import cv2
import numpy as np
import torch
import yaml
from torchvision.transforms import Compose, transforms

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from tools.config import Config

sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '../third-parties'))
# OUR
from BoostingMonocularDepth.utils import (ImageandPatchs, ImageDataset,
                                          applyGridpatch,
                                          calculateprocessingres, generatemask,
                                          getGF_fromintegral, read_image,
                                          rgb2gray)

sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '../third-parties/BoostingMonocularDepth'))
# MIDAS
import midas.utils
#AdelaiDepth
from lib.multi_depth_model_woauxi import RelDepthModel
from lib.net_tools import strip_prefix_if_present
from midas.models.midas_net import MidasNet
from midas.models.transforms import NormalizeImage, PrepareForNet, Resize
from pix2pix.models.pix2pix4depth_model import Pix2Pix4DepthModel
# PIX2PIX : MERGE NET
from pix2pix.options.test_options import TestOptions

warnings.simplefilter('ignore', np.RankWarning)


# Global variables
pix2pixmodel = None
midasmodel = None
srlnet = None
leresmodel = None
factor = None
whole_size_threshold = 3000  # R_max from the paper
GPU_threshold = 1600 - 32 # Limit for the GPU (NVIDIA RTX 2080), can be adjusted


def options():
    # Adding necessary input arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=str, default='data/scene_gen_sdmxx/rgbs', 
                        help='input files directory, Images can be .png .jpg .tiff')
    parser.add_argument('--output_dir', type=str, default='results/00_test_img', 
                        help='result dir. result depth will be png. vides are JMPG as avi')
    parser.add_argument('--savepatchs', type=int, default=0, required=False,
                        help='Activate to save the patch estimations')
    parser.add_argument('--savewholeest', type=int, default=0, required=False,
                        help='Activate to save the base estimations')
    parser.add_argument('--output_resolution', type=int, default=1, required=False,
                        help='0 for results in maximum resolution 1 for resize to input size')
    parser.add_argument('--net_receptive_field_size', type=int, required=False)  # Do not set the value here
    parser.add_argument('--pix2pixsize', type=int, default=1024, required=False)  # Do not change it
    parser.add_argument('--depthNet', type=int, default=2, required=False,
                        help='use to select different base depth networks 0:midas 1:strurturedRL 2:LeRes')
    parser.add_argument('--colorize_results', action='store_true')
    parser.add_argument('--R0', action='store_true')
    parser.add_argument('--R20', action='store_true')
    parser.add_argument('--Final', action='store_true')
    parser.add_argument('--max_res', type=float, default=np.inf)
    parser.add_argument('--checkpoints_dir', type=str, default='./third-parties/BoostingMonocularDepth')

    # Check for required input
    option_, _ = parser.parse_known_args()
    option_.Final = True

    print(option_)
    if int(option_.R0) + int(option_.R20) + int(option_.Final) == 0:
        assert False, 'Please activate one of the [R0, R20, Final] options using --[R0]'
    elif int(option_.R0) + int(option_.R20) + int(option_.Final) > 1:
        assert False, 'Please activate only ONE of the [R0, R20, Final] options'

    if option_.depthNet == 1:
        from structuredrl.models import DepthNet

    # Setting each networks receptive field and setting the patch estimation resolution to twice the receptive
    # field size to speed up the local refinement as described in the section 6 of the main paper.
    if option_.depthNet == 0:
        option_.net_receptive_field_size = 384
        option_.patch_netsize = 2*option_.net_receptive_field_size
    elif option_.depthNet == 1:
        option_.net_receptive_field_size = 448
        option_.patch_netsize = 2*option_.net_receptive_field_size
    elif option_.depthNet == 2:
        option_.net_receptive_field_size = 448
        option_.patch_netsize = 2 * option_.net_receptive_field_size
    else:
        assert False, 'depthNet can only be 0,1 or 2'
    return option_


def write_depth(path, depth, bits=1 , colored=False):
    """Write depth map to pfm and png file.

    Args:
        path (str): filepath without extension
        depth (array): depth
    """
    # write_pfm(path + ".pfm", depth.astype(np.float32))
    if colored == True:
        bits = 1

    depth_min = depth.min()
    depth_max = depth.max()

    max_val = (2**(8*bits))-1
    # if depth_max>max_val:
    #     print('Warning: Depth being clipped')
    #
    # if depth_max - depth_min > np.finfo("float").eps:
    #     out = depth
    #     out [depth > max_val] = max_val
    # else:
    #     out = 0

    if depth_max - depth_min > np.finfo("float").eps:
        out = max_val * (depth - depth_min) / (depth_max - depth_min)
    else:
        out = 0

    if bits == 1 or colored:
        out = out.astype("uint8")
        if colored:
            out = cv2.applyColorMap(out,cv2.COLORMAP_INFERNO)
    elif bits == 2:
        out = out.astype("uint16")
    cv2.imwrite(path+'.png', out)

    return out


def depth_esti_boosting(image_dir, result_dir, image_name, image=None, depthNet=2, output_resolution=1, savepatchs=False, colorize_results=False, postfix='.png', device='cuda'):
    """
    image: [0, 1], [H, W, 3]
    image_name: str without postfix, e.g., '00000'
    output_resolution: 0 for results in maximum resolution 1 for resize to input size
    """
    global DEVICE
    DEVICE = device
    # Generating required directories
    os.makedirs(result_dir, exist_ok=True)
    if savepatchs:
        patchped_est_outputpath = result_dir + '/depth_patchest'
        os.makedirs(patchped_est_outputpath, exist_ok=True)
    if depthNet == 0:
        net_receptive_field_size = 384
    elif depthNet == 1:
        net_receptive_field_size = 448
    elif depthNet == 2:
        net_receptive_field_size = 448
    else:
        assert False, 'depthNet can only be 0, 1 or 2'
        
    patch_netsize = 2 * net_receptive_field_size

    # Load merge network
    # opt = TestOptions().parse()   
    opt = Config('./third-parties/BoostingMonocularDepth/TestOptionsBoosting.yml')
    opt.checkpoints_dir = 'weights/weights_BoostingMonocularDepth'
    global pix2pixmodel
    pix2pixmodel = Pix2Pix4DepthModel(opt)
    pix2pixmodel.save_dir = opt.checkpoints_dir+'/pix2pix/checkpoints/mergemodel'
    pix2pixmodel.load_networks('latest')
    pix2pixmodel.eval()

    # Decide which depth estimation network to load
    if depthNet == 0:
        midas_model_path = opt.checkpoints_dir+"/midas/model.pt"
        global midasmodel
        midasmodel = MidasNet(midas_model_path, non_negative=True)
        midasmodel.to(device)
        midasmodel.eval()
    elif depthNet == 1:
        global srlnet
        srlnet = DepthNet.DepthNet()
        srlnet = torch.nn.DataParallel(srlnet, device_ids=[0]).cuda()
        checkpoint = torch.load('structuredrl/model.pth.tar')
        srlnet.load_state_dict(checkpoint['state_dict'])
        srlnet.eval()
    elif depthNet == 2:
        global leresmodel
        leres_model_path = opt.checkpoints_dir+"/pix2pix/checkpoints/res101.pth"
        checkpoint = torch.load(leres_model_path)
        leresmodel = RelDepthModel(backbone='resnext101')
        leresmodel.load_state_dict(strip_prefix_if_present(checkpoint['depth_model'], "module."),
                                    strict=True)
        del checkpoint
        torch.cuda.empty_cache()
        leresmodel.to(device)
        leresmodel.eval()
    
    mask_org = generatemask((3000, 3000))
    mask = mask_org.copy()

    print("start processing")
    # Value x of R_x defined in the section 5 of the main paper.
    r_threshold_value = 0.2
    if image is not None:
        img = image
    else:
        img = read_image(os.path.join(image_dir, image_name+postfix))
    input_resolution = img.shape
    scale_threshold = 3  # Allows up-scaling with a scale up to 3
    pix2pixsize = 1024  # Do not change it

    # Find the best input resolution R-x. The resolution search described in section 5-double estimation of the main paper and section B of the
    # supplementary material.
    whole_image_optimal_size, patch_scale = calculateprocessingres(img, net_receptive_field_size,
                                                                    r_threshold_value, scale_threshold,
                                                                    whole_size_threshold)
    print('\t wholeImage being processed in :', whole_image_optimal_size)
    # Generate the base estimate using the double estimation.
    whole_estimate = doubleestimate(img, net_receptive_field_size, whole_image_optimal_size,
                                    pix2pixsize, depthNet)


    # Compute the multiplier described in section 6 of the main paper to make sure our initial patch can select
    # small high-density regions of the image.
    global factor
    factor = max(min(1, 4 * patch_scale * whole_image_optimal_size / whole_size_threshold), 0.2)
    print('Adjust factor is:', 1/factor)

    # Compute the default target resolution.
    if img.shape[0] > img.shape[1]:
        a = 2 * whole_image_optimal_size
        b = round(2 * whole_image_optimal_size * img.shape[1] / img.shape[0])
    else:
        a = round(2 * whole_image_optimal_size * img.shape[0] / img.shape[1])
        b = 2 * whole_image_optimal_size
    b = int(round(b / factor))
    a = int(round(a / factor))

    img = cv2.resize(img, (b, a), interpolation=cv2.INTER_CUBIC)

    # Extract selected patches for local refinement
    base_size = net_receptive_field_size*2
    patchset = generatepatchs(img, base_size)
    print('Target resolution: ', img.shape)

    mergein_scale = input_resolution[0] / img.shape[0]
    print('Dynamicly change merged-in resolution; scale:', mergein_scale)


    imageandpatchs = ImageandPatchs(image_dir, image_name, patchset, img, mergein_scale)
    whole_estimate_resized = cv2.resize(whole_estimate, (round(img.shape[1]*mergein_scale),
                                        round(img.shape[0]*mergein_scale)), interpolation=cv2.INTER_CUBIC)
    imageandpatchs.set_base_estimate(whole_estimate_resized.copy())
    imageandpatchs.set_updated_estimate(whole_estimate_resized.copy())
    print('\t Resulted depthmap res will be :', whole_estimate_resized.shape[:2])
    print('patchs to process: '+str(len(imageandpatchs)))

    # Enumerate through all patches, generate their estimations and refining the base estimate.
    for patch_ind in range(len(imageandpatchs)):
        
        # Get patch information
        patch = imageandpatchs[patch_ind] # patch object
        patch_rgb = patch['patch_rgb'] # rgb patch
        patch_whole_estimate_base = patch['patch_whole_estimate_base'] # corresponding patch from base
        rect = patch['rect'] # patch size and location
        patch_id = patch['id'] # patch ID
        org_size = patch_whole_estimate_base.shape # the original size from the unscaled input
        print('\t processing patch', patch_ind, '|', rect)

        # We apply double estimation for patches. The high resolution value is fixed to twice the receptive
        # field size of the network for patches to accelerate the process.
        patch_estimation = doubleestimate(patch_rgb, net_receptive_field_size, patch_netsize,
                                            pix2pixsize, depthNet)

        # Output patch estimation if required
        if savepatchs:
            path = os.path.join(patchped_est_outputpath, imageandpatchs.name + '_{:04}'.format(patch_id))
            midas.utils.write_depth(path, patch_estimation, bits=2, colored=colorize_results)
        patch_estimation = cv2.resize(patch_estimation, (pix2pixsize, pix2pixsize),
                                        interpolation=cv2.INTER_CUBIC)
        patch_whole_estimate_base = cv2.resize(patch_whole_estimate_base, (pix2pixsize, pix2pixsize),
                                                interpolation=cv2.INTER_CUBIC)

        # Merging the patch estimation into the base estimate using our merge network:
        # We feed the patch estimation and the same region from the updated base estimate to the merge network
        # to generate the target estimate for the corresponding region.
        pix2pixmodel.set_input(patch_whole_estimate_base, patch_estimation)

        # Run merging network
        pix2pixmodel.test()
        visuals = pix2pixmodel.get_current_visuals()
        prediction_mapped = visuals['fake_B']
        prediction_mapped = (prediction_mapped+1)/2
        prediction_mapped = prediction_mapped.squeeze().cpu().numpy()
        mapped = prediction_mapped

        # We use a simple linear polynomial to make sure the result of the merge network would match the values of
        # base estimate
        p_coef = np.polyfit(mapped.reshape(-1), patch_whole_estimate_base.reshape(-1), deg=1)
        merged = np.polyval(p_coef, mapped.reshape(-1)).reshape(mapped.shape)
        merged = cv2.resize(merged, (org_size[1],org_size[0]), interpolation=cv2.INTER_CUBIC)

        # Get patch size and location
        w1 = rect[0]
        h1 = rect[1]
        w2 = w1 + rect[2]
        h2 = h1 + rect[3]

        # To speed up the implementation, we only generate the Gaussian mask once with a sufficiently large size
        # and resize it to our needed size while merging the patches.
        if mask.shape != org_size:
            mask = cv2.resize(mask_org, (org_size[1],org_size[0]), interpolation=cv2.INTER_LINEAR)

        tobemergedto = imageandpatchs.estimation_updated_image

        # Update the whole estimation:
        # We use a simple Gaussian mask to blend the merged patch region with the base estimate to ensure seamless
        # blending at the boundaries of the patch region.
        tobemergedto[h1:h2, w1:w2] = np.multiply(tobemergedto[h1:h2, w1:w2], 1 - mask) + np.multiply(merged, mask)
        imageandpatchs.set_updated_estimate(tobemergedto)

    # Output the result
    path = os.path.join(result_dir, imageandpatchs.name)
    if output_resolution == 1:
        depth = cv2.resize(imageandpatchs.estimation_updated_image,
                                            (input_resolution[1], input_resolution[0]),
                                            interpolation=cv2.INTER_CUBIC)
    else:
        depth = imageandpatchs.estimation_updated_image
    depth_out = write_depth(path, depth, bits=2, colored=colorize_results)
    return depth_out
    

    # print("finished")


# Generating local patches to perform the local refinement described in section 6 of the main paper.
def generatepatchs(img, base_size):
    
    # Compute the gradients as a proxy of the contextual cues.
    img_gray = rgb2gray(img)
    whole_grad = np.abs(cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)) +\
        np.abs(cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3))

    threshold = whole_grad[whole_grad > 0].mean()
    whole_grad[whole_grad < threshold] = 0

    # We use the integral image to speed-up the evaluation of the amount of gradients for each patch.
    gf = whole_grad.sum()/len(whole_grad.reshape(-1))
    grad_integral_image = cv2.integral(whole_grad)

    # Variables are selected such that the initial patch size would be the receptive field size
    # and the stride is set to 1/3 of the receptive field size.
    blsize = int(round(base_size/2))
    stride = int(round(blsize*0.75))

    # Get initial Grid
    patch_bound_list = applyGridpatch(blsize, stride, img, [0, 0, 0, 0])

    # Refine initial Grid of patches by discarding the flat (in terms of gradients of the rgb image) ones. Refine
    # each patch size to ensure that there will be enough depth cues for the network to generate a consistent depth map.
    print("Selecting patchs ...")
    patch_bound_list = adaptiveselection(grad_integral_image, patch_bound_list, gf)

    # Sort the patch list to make sure the merging operation will be done with the correct order: starting from biggest
    # patch
    patchset = sorted(patch_bound_list.items(), key=lambda x: getitem(x[1], 'size'), reverse=True)
    return patchset


# Adaptively select patches
def adaptiveselection(integral_grad, patch_bound_list, gf):
    patchlist = {}
    count = 0
    height, width = integral_grad.shape

    search_step = int(32/factor)

    # Go through all patches
    for c in range(len(patch_bound_list)):
        # Get patch
        bbox = patch_bound_list[str(c)]['rect']

        # Compute the amount of gradients present in the patch from the integral image.
        cgf = getGF_fromintegral(integral_grad, bbox)/(bbox[2]*bbox[3])

        # Check if patching is beneficial by comparing the gradient density of the patch to
        # the gradient density of the whole image
        if cgf >= gf:
            bbox_test = bbox.copy()
            patchlist[str(count)] = {}

            # Enlarge each patch until the gradient density of the patch is equal
            # to the whole image gradient density
            while True:

                bbox_test[0] = bbox_test[0] - int(search_step/2)
                bbox_test[1] = bbox_test[1] - int(search_step/2)

                bbox_test[2] = bbox_test[2] + search_step
                bbox_test[3] = bbox_test[3] + search_step

                # Check if we are still within the image
                if bbox_test[0] < 0 or bbox_test[1] < 0 or bbox_test[1] + bbox_test[3] >= height \
                        or bbox_test[0] + bbox_test[2] >= width:
                    break

                # Compare gradient density
                cgf = getGF_fromintegral(integral_grad, bbox_test)/(bbox_test[2]*bbox_test[3])
                if cgf < gf:
                    break
                bbox = bbox_test.copy()

            # Add patch to selected patches
            patchlist[str(count)]['rect'] = bbox
            patchlist[str(count)]['size'] = bbox[2]
            count = count + 1
    
    # Return selected patches
    return patchlist


# Generate a double-input depth estimation
def doubleestimate(img, size1, size2, pix2pixsize, net_type):
    # Generate the low resolution estimation
    estimate1 = singleestimate(img, size1, net_type)
    # Resize to the inference size of merge network.
    estimate1 = cv2.resize(estimate1, (pix2pixsize, pix2pixsize), interpolation=cv2.INTER_CUBIC)

    # Generate the high resolution estimation
    estimate2 = singleestimate(img, size2, net_type)
    # Resize to the inference size of merge network.
    estimate2 = cv2.resize(estimate2, (pix2pixsize, pix2pixsize), interpolation=cv2.INTER_CUBIC)

    # Inference on the merge model
    pix2pixmodel.set_input(estimate1, estimate2)
    pix2pixmodel.test()
    visuals = pix2pixmodel.get_current_visuals()
    prediction_mapped = visuals['fake_B']
    prediction_mapped = (prediction_mapped+1)/2
    prediction_mapped = (prediction_mapped - torch.min(prediction_mapped)) / (
                torch.max(prediction_mapped) - torch.min(prediction_mapped))
    prediction_mapped = prediction_mapped.squeeze().cpu().numpy()

    return prediction_mapped


# Generate a single-input depth estimation
def singleestimate(img, msize, net_type):
    if msize > GPU_threshold:
        print(" \t \t DEBUG| GPU THRESHOLD REACHED", msize, '--->', GPU_threshold)
        msize = GPU_threshold

    if net_type == 0:
        return estimatemidas(img, msize)
    elif net_type == 1:
        return estimatesrl(img, msize)
    elif net_type == 2:
        return estimateleres(img, msize)


# Inference on SGRNet
def estimatesrl(img, msize):
    # SGRNet forward pass script adapted from https://github.com/KexianHust/Structure-Guided-Ranking-Loss
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img_resized = cv2.resize(img, (msize, msize), interpolation=cv2.INTER_CUBIC).astype('float32')
    tensor_img = img_transform(img_resized)

    # Forward pass
    input_img = torch.autograd.Variable(tensor_img.cuda().unsqueeze(0), volatile=True)
    with torch.no_grad():
        output = srlnet(input_img)

    # Normalization
    depth = output.squeeze().cpu().data.numpy()
    min_d, max_d = depth.min(), depth.max()
    depth_norm = (depth - min_d) / (max_d - min_d)

    depth_norm = cv2.resize(depth_norm, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

    return depth_norm

# Inference on MiDas-v2
def estimatemidas(img, msize):
    # MiDas -v2 forward pass script adapted from https://github.com/intel-isl/MiDaS/tree/v2
    device = DEVICE
    transform = Compose(
        [
            Resize(
                msize,
                msize,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="upper_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ]
    )

    img_input = transform({"image": img})["image"]

    # Forward pass
    with torch.no_grad():
        sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
        prediction = midasmodel.forward(sample)

    prediction = prediction.squeeze().cpu().numpy()
    prediction = cv2.resize(prediction, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

    # Normalization
    depth_min = prediction.min()
    depth_max = prediction.max()

    if depth_max - depth_min > np.finfo("float").eps:
        prediction = (prediction - depth_min) / (depth_max - depth_min)
    else:
        prediction = 0

    return prediction


def scale_torch(img):
    """
    Scale the image and output it in torch.tensor.
    :param img: input rgb is in shape [H, W, C], input depth/disp is in shape [H, W]
    :param scale: the scale factor. float
    :return: img. [C, H, W]
    """
    if len(img.shape) == 2:
        img = img[np.newaxis, :, :]
    if img.shape[2] == 3:
        transform = transforms.Compose([transforms.ToTensor(),
		                                transforms.Normalize((0.485, 0.456, 0.406) , (0.229, 0.224, 0.225) )])
        img = transform(img.astype(np.float32))
    else:
        img = img.astype(np.float32)
        img = torch.from_numpy(img)
    return img

# Inference on LeRes
def estimateleres(img, msize):
    # LeReS forward pass script adapted from https://github.com/aim-uofa/AdelaiDepth/tree/main/LeReS

    rgb_c = img[:, :, ::-1].copy()
    A_resize = cv2.resize(rgb_c, (msize, msize))
    img_torch = scale_torch(A_resize)[None, :, :, :]

    # Forward pass
    with torch.no_grad():
        prediction = leresmodel.inference(img_torch)

    prediction = prediction.squeeze().cpu().numpy()
    prediction = cv2.resize(prediction, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

    return prediction


## python 00_run_test.py --data_dir 'imgs_scene_indoor' --output_dir 'imgs_scene_indoor'

if __name__ == "__main__":
    # select device
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device("cuda")
    print("device: %s" % device)
    option = options()
    
    option.data_dir = '/apdcephfs/share_1330077/eckertzhang/Dataset/data_for_text2nerf/debug_a_cozy_living_room/rgbs'
    option.output_dir = './test_depth'
    
    dataset = ImageDataset(option.data_dir, 'test_d')
    # dataset = ImageDataset(option.data_dir, 'test')
    for image_ind, images in enumerate(dataset):
        img = images.rgb_image
        image_name = images.name
        depth_out = depth_esti_boosting(option.data_dir, option.output_dir, image_name, device=device)


    