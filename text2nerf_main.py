import cv2
import os
import random
import shutil
import sys
# from sympy import false
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from dataLoader import dataset_dict
from e_opt import config_parser
from renderer import *
from utils import *

from PIL import Image
from scripts.inpaint_sdm import text2inpainting_sdm
from dataLoader.scene_util import get_local_fixed_poses2
from dataLoader.scene_gen import produce_formatted_data
from dataLoader.bilateral_filtering import sparse_bilateral_filtering
from scripts.depth_esti_boosting import depth_esti_boosting
from transformers import CLIPProcessor, CLIPModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
renderer = OctreeRender_trilinear_fast
clip_model = CLIPModel.from_pretrained("weights/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("weights/clip-vit-base-patch32")

@torch.no_grad()
def render_test(args):
    # init dataset
    dataset = dataset_dict[args.dataset_name]
    if not os.path.isfile(os.path.join(args.datadir, 'rgbs/%05d.png' % 0)):
        args.datadir = args.datadir+'_'+args.prompt.replace(' ','_')
    test_dataset = dataset(args, split='test', is_stack=True)
    white_bg = test_dataset.white_bg
    ndc_ray = args.ndc_ray

    if not os.path.exists(args.ckpt):
        print('the ckpt path does not exists!!')
        return

    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.load(ckpt)

    logfolder = os.path.dirname(args.ckpt)
    if args.render_train:
        os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
        train_dataset = dataset(args, split='train', is_stack=True)
        PSNRs_test = evaluation(train_dataset,tensorf, args, renderer, f'{logfolder}/imgs_train_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        print(f'======> {args.expname} train all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_test:
        os.makedirs(f'{logfolder}/imgs_test_all', exist_ok=True)
        evaluation(test_dataset,tensorf, args, renderer, f'{logfolder}/imgs_test_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device, video_gen=True)

    if args.render_path:
        c2ws = test_dataset.render_path
        os.makedirs(f'{logfolder}/imgs_path_all', exist_ok=True)
        evaluation_path(test_dataset,tensorf, c2ws, renderer, f'{logfolder}/imgs_path_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)


def render_warping_inapinting(N_iter, tensorf, renderer, dataset, poses, H, W, intrinsic, args, logpath, N_samples=-1, white_bg=False, ndc_ray=False, device='cuda'):
    """
    allrays: [n_views, H*W, 6]
    allrgbs: [n_views, H, W, 3]
    alldepth: [n_views, H, W, 1]
    poses: [n_views, 4, 4]
    """
    # load data from dataset
    use_support_set = dataset.use_support_set
    all_rays_gen_split, all_rgbs_gen_split, all_depth_gen_split = dataset.all_rays_gen_split, dataset.all_rgbs_gen_split, dataset.all_depth_gen_split

    # save path of warped image
    save_path_warp = os.path.join(logpath, "DIBR")
    os.makedirs(os.path.join(save_path_warp, 'warped'), exist_ok=True)
    os.makedirs(os.path.join(save_path_warp, 'rendered'), exist_ok=True)
    os.makedirs(os.path.join(save_path_warp, 'mask'), exist_ok=True)
    os.makedirs(os.path.join(save_path_warp, 'mask_inv'), exist_ok=True)
    os.makedirs(os.path.join(save_path_warp, 'depth'), exist_ok=True)
    os.makedirs(os.path.join(save_path_warp, 'rgbs_support'), exist_ok=True)
    os.makedirs(os.path.join(save_path_warp, 'rgbs'), exist_ok=True)

    #
    update_known_views = args.update_known_views
    only_update_initial_view = False
    prompt = args.prompt
    negative_prompt = 'blurry, bad art, blurred, text, watermark'
    use_filter_filling = args.use_filter_filling_holes
    use_rendered_img_to_warp = args.use_rendered_img_to_warp
    push_depth = args.push_depth
    select_type = args.frame_select_type  # manual/auto_cos

    # render existing training views
    rgbs_pre, depths_pre = [], []
    rgbs_gt, depths_gt = [], []
    for n in range(N_iter):
        print("Pre-processing of Warp: Render image {}/{}".format(n + 1, N_iter))
        rgb_tar, depth_tar, rays_tar = all_rgbs_gen_split[n], all_depth_gen_split[n], all_rays_gen_split[n]
        rgb_tar, depth_tar = rgb_tar.reshape(H, W, 3), depth_tar.reshape(H, W)
        rgbs_gt.append(rgb_tar.numpy())
        depths_gt.append(depth_tar.numpy())

        with torch.no_grad():
            rgb_map, _, depth_map, _, _ = renderer(rays_tar, tensorf, chunk=args.batch_size, N_samples=N_samples,
                                            ndc_ray=ndc_ray, white_bg = white_bg, device=device, is_train=False)
        rgb_map = rgb_map.clamp(0.0, 1.0)
        rgb_map, depth_map = rgb_map.reshape(H, W, 3).cpu().numpy(), depth_map.reshape(H, W).cpu().numpy()

        vis_photos, vis_depths = sparse_bilateral_filtering(
                depth_map.copy(), rgb_map.copy(), filter_size=[7, 5, 5, 3, 3], 
                depth_threshold=0.02, num_iter=5, HR=False, mask=None)
        depth_map = vis_depths[-1]
        rgb_map = vis_photos[-1]
        
        rgbs_pre.append(rgb_map)
        depths_pre.append(depth_map)

    # move data to cpu & numpy
    poses_np = poses.cpu().numpy()

    ## warp existing views to target view through DIBR
    if use_rendered_img_to_warp:
        myMap, output_image_warp, output_depth = bilinear_splat_warping_multiview(rgbs_pre, depths_pre, poses_np, poses_np[N_iter], H, W, intrinsic, masks=None)
    else:
        myMap, output_image_warp, output_depth = bilinear_splat_warping_multiview(rgbs_gt, depths_gt, poses_np, poses_np[N_iter], H, W, intrinsic, masks=None)

    if use_filter_filling:
        output_image_warp, myMap_filt, output_depth = dibr_filter_mask2(output_image_warp, myMap, output_depth=output_depth)
    else:
        myMap_filt = myMap
    
    output_image_warp = (output_image_warp * 255).astype(np.uint8)
    mask_image = (myMap_filt * 255).astype(np.uint8)
    mask_inv = ((1 - myMap_filt) * 255).astype(np.uint8)
    imageio.imwrite(os.path.join(save_path_warp, 'warped', '%05d.png' % N_iter), output_image_warp)
    imageio.imwrite(os.path.join(save_path_warp, 'warped', '%05d_depth.png' % N_iter), output_depth)
    imageio.imwrite(os.path.join(save_path_warp, 'mask', '%05d.png' % N_iter), mask_image)
    imageio.imwrite(os.path.join(save_path_warp, 'mask_inv', '%05d.png' % N_iter), mask_inv)

    # mask expansion
    if update_known_views:
        myMap_filt_blur0 = cv2.blur(myMap_filt.astype(np.float32), (5,5))  # 10
        myMap_filt_blur0 = (myMap_filt_blur0>0.99) * 1
        mask_ex = myMap_filt-myMap_filt_blur0
        mask_ex = np.concatenate([mask_ex[:,:,np.newaxis], mask_ex[:,:,np.newaxis], mask_ex[:,:,np.newaxis]], -1)
        myMap_filt = myMap_filt_blur0
    else:
        mask_ex = np.concatenate([myMap_filt[:,:,np.newaxis], myMap_filt[:,:,np.newaxis], myMap_filt[:,:,np.newaxis]], -1)

    for i in range(3):
        output_image_warp[:,:,i] *= myMap_filt.astype(np.uint8)
    mask_image = (myMap_filt * 255).astype(np.uint8)
    mask_inv = ((1 - myMap_filt) * 255).astype(np.uint8)
    imageio.imwrite(os.path.join(save_path_warp, 'warped', '%05d_expand.png' % N_iter), output_image_warp)
    imageio.imwrite(os.path.join(save_path_warp, 'mask', '%05d_expand.png' % N_iter), mask_image)
    imageio.imwrite(os.path.join(save_path_warp, 'mask_inv', '%05d_expand.png' % N_iter), mask_inv)
    
    ## render from target view
    rays_tar = all_rays_gen_split[N_iter]
    with torch.no_grad():
        rgb_render, _, depth_rendered0, _, _ = renderer(rays_tar, tensorf, chunk=args.batch_size, N_samples=N_samples,
                                        ndc_ray=ndc_ray, white_bg = white_bg, device=device, is_train=False)
    rgb_render = rgb_render.clamp(0.0, 1.0).reshape(H, W, 3).cpu().numpy()
    rgb_render = (rgb_render * 255).astype(np.uint8)
    depth_rendered = depth_rendered0.reshape(H, W).cpu().numpy() * myMap_filt
    imageio.imwrite(os.path.join(save_path_warp, 'rendered', '%05d_ori.png' % N_iter), rgb_render)
    imageio.imwrite(os.path.join(save_path_warp, 'rendered', '%05d_depth.png' % N_iter), depth_rendered)
    rgb_render_ = rgb_render.copy()
    for i in range(3):
        rgb_render_[:,:,i] = rgb_render_[:,:,i]*myMap_filt+255*(1 - myMap_filt)
    rgb_render_ = (rgb_render_).astype(np.uint8)
    imageio.imwrite(os.path.join(save_path_warp, 'rendered', '%05d.png' % N_iter), rgb_render_)
    ## use rendered img replace warped one
    use_rendered_img_with_mask = False
    if use_rendered_img_with_mask:
        output_image = rgb_render_.copy()
    else:
        output_image = rgb_render.copy()

    ## Stable Diffusion Inpainting
    if select_type == 'auto_cos':
        init_image = Image.fromarray(output_image).convert("RGB")
        mask_image = Image.fromarray(mask_inv).convert("RGB")
        sub_idx = 0
        n_stable = 20
        num_per_circle = 5
        num_circle = int(n_stable/num_per_circle)
        sdm_inpaint = text2inpainting_sdm(device=device)
        # calculate cosine similarity
        cos_similarity = []
        img0 = all_rgbs_gen_split[0].numpy()
        img0 = Image.fromarray((img0*255).astype(np.uint8)).convert("RGB")
        init_data = clip_processor(text=[prompt], images=[img0], return_tensors="pt", padding=True)
        init_img_embeds = clip_model(**init_data).image_embeds
        logit_scale = clip_model.logit_scale.exp()

        for jjj in range(num_circle):
            with torch.no_grad():
                images = sdm_inpaint.sdm(init_image=init_image, mask_image=mask_image, prompt=prompt, negative_prompt=negative_prompt, guidance_scale=7.5, num_images_per_prompt=num_per_circle)
                inputs4similarity = clip_processor(text=[prompt], images=images, return_tensors="pt", padding=True)
                # logits_per_image = model(**inputs4similarity).logits_per_image
                image_embeds = clip_model(**inputs4similarity).image_embeds
                logits_per_image = torch.matmul(image_embeds, init_img_embeds.t()) * logit_scale
            
            for j_i in range(len(images)):
                image = np.array(images[j_i])
                cos_similarity.append((logits_per_image[j_i, 0], sub_idx))
                imageio.imwrite(os.path.join(save_path_warp, 'rgbs', '%05d_%03d.png' % (N_iter, sub_idx)), image)
                sub_idx += 1
        cos_similarity_sorted = sorted(
                                cos_similarity,
                                key=lambda t: t[0])  # small --> large
        selected_id = cos_similarity_sorted[-1][1]
        shutil.move(os.path.join(save_path_warp, 'rgbs', '%05d_%03d.png' % (N_iter, selected_id)), os.path.join(save_path_warp, 'rgbs', '%05d.png' % N_iter))

    ## depth estimation
    with torch.no_grad():
        depth_est = depth_esti_boosting(
            image_dir=os.path.join(save_path_warp, 'rgbs'), 
            result_dir=os.path.join(save_path_warp, 'depth'), 
            image_name='%05d' % N_iter,
            device=device
            )
    depth_est = depth_est / 12000 + push_depth
    
    ## depth alignment stage 1: global alignment
    pixel_filled = []
    for i in range(H):
        for j in range(W):
            if myMap_filt[j,i]>0:
                pixel_filled.append((j, i))
    num_max = min(len(pixel_filled), 10000)
    pixel_sample = random.sample(pixel_filled, num_max)
    thresh = (depth_rendered.max()-push_depth)/(depth_est.max()-push_depth)
    scales = []
    for ii in range(len(pixel_sample)-1):
        y_o1, x_o1 = pixel_sample[ii]
        y_o2, x_o2 = pixel_sample[ii+1]
        dd1 = depth_rendered[y_o1, x_o1]-depth_rendered[y_o2, x_o2]
        dd2 = depth_est[y_o1, x_o1]-depth_est[y_o2, x_o2]
        ss = dd1/(dd2+1e-8)
        if not np.isfinite(ss) or abs(ss-1)>5*abs(thresh-1) or ss < 0:
            continue
        scales.append(ss)
    if len(scales)==0:
        scales.append(thresh)
    scales = np.stack(scales)
    scale = np.average(scales)
    print('global scale:', scale)
    depth_scaled = depth_est * scale
    shifts = []
    thresh = depth_scaled.max()-depth_rendered.max()
    for ii in range(len(pixel_sample)):
        y_o1, x_o1 = pixel_sample[ii]
        ss = depth_scaled[y_o1, x_o1] - depth_rendered[y_o1, x_o1]
        if abs(ss) > 2*abs(thresh):
            continue
        shifts.append(ss)
    if len(shifts)==0:
        shifts.append(thresh)
    shifts = np.stack(shifts)
    shift = np.average(shifts)
    print('global shift:', shift)
    depth_shift = depth_scaled - shift

    ## depth alignment stage 2: local alignment
    from scripts.merge_model_single_channel import depth_merge_model
    merge_model = depth_merge_model(ckpt_name='epoch_00440', device=device)
    depth_ref = ((depth_rendered - push_depth) * 12000 / 32768. - 1.) * myMap_filt  #[-1,1]
    depth_src = ((depth_shift - push_depth) * 12000 / 32768. - 1.) #[-1,1]
    depth_merged = merge_model.run_finetune_numpy(depth_ref.astype(np.float32), depth_src.astype(np.float32), myMap_filt.copy(), outsize=512, lr=1e-5, iter=500)
    depth_new = (depth_merged.detach().cpu().numpy() + 1.) * 32768.
    imageio.imwrite(os.path.join(save_path_warp, 'depth', '%05d_depth_finetuning2.png' % N_iter), depth_new)


    depth_new = (depth_new / 12000 + push_depth).astype(np.float32)
    imageio.imwrite(os.path.join(save_path_warp, 'depth', '%05d_depth_finetuning_masked2.png' % N_iter), depth_rendered*myMap_filt+depth_new*(1-myMap_filt))
    img_new = imageio.imread(os.path.join(save_path_warp, 'rgbs', '%05d.png' % N_iter))
    img_new = (img_new/255.).astype(np.float32)

    vis_photos, vis_depths = sparse_bilateral_filtering(
            depth_new.copy(), img_new.copy(), filter_size=[5, 5, 3, 3], 
            depth_threshold=0.02, num_iter=4, HR=False, mask=None)
    depth_new = vis_depths[-1]
    img_new = vis_photos[-1]
    imageio.imwrite(os.path.join(save_path_warp, 'depth', '%05d_depth_merged_aft_filter.png' % N_iter), depth_new*(1-myMap_filt)+depth_rendered*myMap_filt)
    imageio.imwrite(os.path.join(save_path_warp, 'depth', '%05d_new.png' % N_iter), depth_new)

    ## update known views with the inpainted image   all_rgbs_gen_split, all_depth_gen_split, all_masks_gen_split
    current_mask_inpainted = 1-myMap_filt
    dataset.all_rgbs_gen_split = torch.cat([dataset.all_rgbs_gen_split, torch.tensor(img_new, dtype=torch.float32)[None]], 0)
    dataset.all_depth_gen_split = torch.cat([dataset.all_depth_gen_split, torch.tensor(depth_new, dtype=torch.float32)[None]], 0)
    dataset.all_masks_gen_split = torch.cat([dataset.all_masks_gen_split, torch.tensor(current_mask_inpainted)[None]], 0)
    if update_known_views:
        if only_update_initial_view:
            n_view_update = 1
        else:
            n_view_update = N_iter  # 
        inpaint_masks_split = dataset.all_masks_gen_split
        # avoid the foreground disappear
        rgbs_warp, masks_warp0, depth_warp = gt_warping(img_new, depth_new, poses_np[N_iter], poses_np[:n_view_update], H, W,
                                                        # logpath=os.path.join(save_path_warp, 'rgbs_support/%05d_warp2known'%N_iter), 
                                                        intrinsic=intrinsic, warp_depth=True, bilinear_splat=True)
        _, masks_warp, _ = gt_warping(img_new, depth_new, poses_np[N_iter], poses_np[:n_view_update], H, W, 
                                    #   logpath=os.path.join(save_path_warp, 'rgbs_support/%05d_warp2known_masked'%N_iter), 
                                      intrinsic=intrinsic, mask_gt=(1-myMap_filt), warp_depth=True, bilinear_splat=True)
        for ii in range(n_view_update):
            mask = masks_warp[ii]
            if mask.sum() < 1:
                continue
            mask3 = np.repeat(np.expand_dims(mask, -1), 3, axis=-1)
            pose_ref = poses_np[ii]
            img = all_rgbs_gen_split[ii].numpy()
            os.makedirs(os.path.join(save_path_warp, 'rgbs_support/%05d_warp2known'%N_iter), exist_ok=True)
            imageio.imwrite(os.path.join(save_path_warp, 'rgbs_support/%05d_warp2known'%N_iter, '%05d_image_pre.png'%ii), img)
            imageio.imwrite(os.path.join(save_path_warp, 'rgbs_support/%05d_warp2known'%N_iter, '%05d_image_warp.png'%ii), rgbs_warp[ii])
            img = img * (1-mask3) + rgbs_warp[ii] * mask3
            depth = all_depth_gen_split[ii].numpy()
            imageio.imwrite(os.path.join(save_path_warp, 'rgbs_support/%05d_warp2known'%N_iter, '%05d_depth_pre.png'%ii), depth)
            imageio.imwrite(os.path.join(save_path_warp, 'rgbs_support/%05d_warp2known'%N_iter, '%05d_depth_warp.png'%ii), depth_warp[ii])

            with torch.no_grad():
                depth_est = depth_esti_boosting(
                    image_dir=os.path.join(save_path_warp, 'rgbs_support/%05d_warp2known/%03d'%(N_iter, ii)), 
                    result_dir=os.path.join(save_path_warp, 'rgbs_support/%05d_warp2known/%03d/depth_est'%(N_iter, ii)), 
                    image_name='%05d_updated_image'%ii,
                    image=img,
                    device=device
                    )
            from scripts.merge_model_single_channel import depth_merge_model
            merge_model = depth_merge_model(ckpt_name='epoch_00440', device=device)
            depth_ref = ((depth - push_depth) * 12000 / 32768. - 1.) * (1-mask)  #[-1,1]
            depth_est = depth_est / 32768. - 1. #[-1,1]
            depth_update = merge_model.run_finetune_numpy(depth_ref.astype(np.float32), depth_est.astype(np.float32), (1-mask), outsize=512, lr=1e-5, iter=500)
            depth_update = (depth_update.detach().cpu().numpy() + 1.) * 32768.
            depth = (depth_update / 12000 + push_depth).astype(np.float32)
            imageio.imwrite(os.path.join(save_path_warp, 'rgbs_support/%05d_warp2known'%N_iter, '%05d_updated_image0.png'%ii), img)
            imageio.imwrite(os.path.join(save_path_warp, 'rgbs_support/%05d_warp2known'%N_iter, '%05d_updated_depth0.png'%ii), depth)
            dataset.all_rgbs_gen_split[ii] = torch.tensor(img, dtype=torch.float32)
            dataset.all_depth_gen_split[ii] = torch.tensor(depth, dtype=torch.float32)
            inpaint_mask = inpaint_masks_split[ii].numpy()
            if use_support_set:
                vis_photos, vis_depths = sparse_bilateral_filtering(
                        depth.copy(), img.copy(), filter_size=[5,5,3,3], 
                        depth_threshold=0.02, num_iter=4, HR=False, mask=None)
                depth = vis_depths[-1]
                img = vis_photos[-1]
                imageio.imwrite(os.path.join(save_path_warp, 'rgbs_support/%05d_warp2known'%N_iter, '%05d_updated_image.png'%ii), img)
                imageio.imwrite(os.path.join(save_path_warp, 'rgbs_support/%05d_warp2known'%N_iter, '%05d_updated_depth.png'%ii), depth)
                poses_support = get_local_fixed_poses2(pose_ref, angle=0, range_center=args.trans_range, range_yaw=0.6, range_pitch=0.2)
                rgbs_warp_temp, _, depth_warp_temp = gt_warping(img, depth, poses_support[0], poses_support[1:], H, W,
                                                                            #   logpath=os.path.join(save_path_warp, 'rgbs_support/%05d_%03d'%(N_iter, ii)), 
                                                                intrinsic=intrinsic, warp_depth=True, bilinear_splat=True)
                _, masks_warp_temp, _ = gt_warping(img, depth, poses_support[0], poses_support[1:], H, W,
                                                                            #   logpath=os.path.join(save_path_warp, 'rgbs_support/%05d_%03d'%(N_iter, ii)), 
                                                                intrinsic=intrinsic, mask_gt=inpaint_mask, warp_depth=True, bilinear_splat=True)
                imgs = np.concatenate([np.expand_dims(img, 0), rgbs_warp_temp], 0)
                masks = np.concatenate([np.expand_dims(inpaint_mask, 0), masks_warp_temp], 0)
                deps = np.concatenate([np.expand_dims(depth, 0), depth_warp_temp], 0)
                if args.pose_traj=='local_fixed' and ii==0:
                    imgs = np.concatenate([imgs[:1], imgs[N_iter+1:]] ,0)
                    masks = np.concatenate([masks[:1], masks[N_iter+1:]] ,0)
                    deps = np.concatenate([deps[:1], deps[N_iter+1:]] ,0)
                    poses_support = np.concatenate([poses_support[:1], poses_support[N_iter+1:]] ,0)
                all_rays, all_rgbs, all_depth, _, _, _, _ = produce_formatted_data(imgs, deps, masks, poses_support, intrinsic, H, W, mode='train')
            else:
                all_rays, all_rgbs, all_depth, _, _, _, _ = produce_formatted_data(np.expand_dims(img,0), np.expand_dims(depth,0), np.expand_dims(inpaint_mask, 0), 
                                                                                   np.expand_dims(pose_ref, 0), intrinsic, H, W, mode='train')
            dataset.all_rays_update['%05d'%ii] = all_rays
            dataset.all_rgbs_update['%05d'%ii] = all_rgbs
            dataset.all_depth_update['%05d'%ii] = all_depth

    ## get support set for the new inpainted view
    if use_support_set:
        poses_support = get_local_fixed_poses2(poses_np[N_iter], angle=0, range_center=args.trans_range, range_yaw=0.6, range_pitch=0.2)
        rgbs_warp, _, depth_warp = gt_warping(img_new, depth_new, poses_support[0], poses_support[1:], H, W, 
                                              logpath=os.path.join(save_path_warp, 'rgbs_support/%05d'%N_iter), 
                                              intrinsic=intrinsic, warp_depth=True, bilinear_splat=True)
        _, masks_warp, _ = gt_warping(img_new, depth_new, poses_support[0], poses_support[1:], H, W, 
                                    #   logpath=os.path.join(save_path_warp, 'rgbs_support/%05d_'%N_iter), 
                                      intrinsic=intrinsic, mask_gt=(1-myMap_filt), warp_depth=True, bilinear_splat=True)
        
        images = np.concatenate([np.expand_dims(img_new, 0), rgbs_warp], 0)
        masks = np.concatenate([np.expand_dims(current_mask_inpainted, 0), masks_warp], 0)
        depths = np.concatenate([np.expand_dims(depth_new, 0), depth_warp], 0)
        all_rays, all_rgbs, all_depth, _, _, _, _ = produce_formatted_data(images, depths, masks, poses_support, intrinsic, H, W, mode='train')

    else:
        all_rays, all_rgbs, all_depth, _, _, _, _ = produce_formatted_data(np.expand_dims(img_new, 0), np.expand_dims(depth_new, 0), np.expand_dims(current_mask_inpainted, 0), 
                                                                           np.expand_dims(poses_np[N_iter], 0), intrinsic, H, W, mode='train')
    dataset.all_rays_update['%05d'%N_iter] = all_rays
    dataset.all_rgbs_update['%05d'%N_iter] = all_rgbs
    dataset.all_depth_update['%05d'%N_iter] = all_depth
    
    return dataset



def reconstruction(args):
    # init dataset
    dataset = dataset_dict[args.dataset_name]
    if not os.path.isfile(os.path.join(args.datadir, 'rgbs/%05d.png' % 0)):
        args.datadir = args.datadir+'_'+args.prompt.replace(' ','_')
    train_dataset = dataset(args, split='train')
    test_dataset = dataset(args, split='test')
    white_bg = train_dataset.white_bg
    near_far = train_dataset.near_far
    ndc_ray = args.ndc_ray

    # initialize resolution
    n_lamb_sigma = args.n_lamb_sigma
    n_lamb_sh = args.n_lamb_sh
    type_depth_loss = args.type_depth_loss

    args.expname = args.expname+'_'+(args.prompt).replace(' ', '_')+'_'+str(args.angle)+'_'+str(args.trans_range)
    logfolder = f'{args.basedir}/{args.expname}'
    
    # init log file
    os.makedirs(logfolder, exist_ok=True)
    summary_writer = SummaryWriter(logfolder)

    # write args
    filename = open(os.path.join(logfolder, 'args.txt'),'w')
    dic_args = vars(args)
    for k,v in dic_args.items():
        filename.write(k+':'+str(v))
        filename.write('\n')
    filename.close()

    # init parameters
    aabb = train_dataset.scene_bbox.to(device)
    reso_cur = N_to_reso(args.N_voxel_init, aabb)
    nSamples = min(args.nSamples, int(cal_n_samples(reso_cur, args.step_ratio)/2))

    # load TensoRF Model & Optimizer
    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location=device)
        kwargs = ckpt['kwargs']
        kwargs.update({'device':device})
        tensorf = eval(args.model_name)(**kwargs)
        tensorf.load(ckpt)
    else:
        tensorf = eval(args.model_name)(aabb, reso_cur, device,
                    density_n_comp=n_lamb_sigma, appearance_n_comp=n_lamb_sh, app_dim=args.data_dim_color, near_far=near_far,
                    shadingMode=args.shadingMode, alphaMask_thres=args.alpha_mask_thre, density_shift=args.density_shift, distance_scale=args.distance_scale,
                    pos_pe=args.pos_pe, view_pe=args.view_pe, fea_pe=args.fea_pe, featureC=args.featureC, step_ratio=args.step_ratio, fea2denseAct=args.fea2denseAct)
    grad_vars = tensorf.get_optparam_groups(args.lr_init, args.lr_basis)
    optimizer = torch.optim.Adam(grad_vars, betas=(0.9,0.99))

    ### ----------- Training Prepare ------------
    torch.cuda.empty_cache()
    PSNRs,PSNRs_test = [],[0]
    Ortho_reg_weight = args.Ortho_weight
    print("initial Ortho_reg_weight", Ortho_reg_weight)
    L1_reg_weight = args.L1_weight_inital
    print("initial L1_reg_weight", L1_reg_weight)
    TV_weight_density, TV_weight_app = args.TV_weight_density, args.TV_weight_app
    tvreg = TVLoss()
    print(f"initial TV_weight density: {TV_weight_density} appearance: {TV_weight_app}")

    print('-'*10 + ' Begin training! ' + '-'*10)
    n_epoch_stage1, n_epoch_stage2_each = args.n_stage1, args.n_stage2
    global_step = 0
    global_epoch = 0

    ### ----------- Training ------------
    allrays, allrgbs, alldepth = train_dataset.all_rays, train_dataset.all_rgbs, train_dataset.all_depth
    poses_gen = train_dataset.poses
    vis_inpaint_views = False if train_dataset.poses.shape[0]==train_dataset.poses_support.shape[0] and (train_dataset.poses == train_dataset.poses_support).all() else True
    if not args.ndc_ray:
        allrays, allrgbs, alldepth = tensorf.filtering_rays(allrays, allrgbs, all_depth=alldepth, bbox_only=True)
    
    trainingSampler_pre = SimpleSampler(allrays.shape[0], args.batch_size)
    transmit_loss = TransMittanceLoss_mask(device=device)
    n_iters_eachepoch_stage1 = allrays.shape[0]//args.batch_size + (allrays.shape[0]%args.batch_size>0)
    n_iters_stage1 = n_epoch_stage1 * n_iters_eachepoch_stage1
    if args.lr_decay_iters > 0:
        lr_factor = args.lr_decay_target_ratio**(1/args.lr_decay_iters)
    else:
        args.lr_decay_iters = n_iters_stage1 if n_iters_stage1>0 else 1000
        lr_factor = args.lr_decay_target_ratio**(1/args.lr_decay_iters)
    print("lr decay", args.lr_decay_target_ratio, args.lr_decay_iters)

    N_iter = 0
    n_epoch_stage2 = n_epoch_stage2_each * (poses_gen.shape[0]-1)
    n_total = n_epoch_stage1+n_epoch_stage2
    for epoch in range(n_total+10):
        global_epoch += 1

        if epoch >= n_epoch_stage1 and (epoch-n_epoch_stage1)%n_epoch_stage2_each==0 and epoch < n_total:
            N_iter = (epoch-n_epoch_stage1)//n_epoch_stage2_each + 1

            W, H = train_dataset.img_wh
            intrinsic = (train_dataset.focal_x, train_dataset.focal_y, train_dataset.cx, train_dataset.cy)
            train_dataset = render_warping_inapinting(N_iter, tensorf, renderer, train_dataset, poses_gen, \
                H, W, intrinsic, args, logfolder, N_samples=nSamples, white_bg=white_bg, ndc_ray=ndc_ray, device=device)
            
            rays = train_dataset.all_rays_update['%05d'%N_iter]
            rgbs = train_dataset.all_rgbs_update['%05d'%N_iter]
            depths = train_dataset.all_depth_update['%05d'%N_iter]

            # update previous training rays
            allrays, allrgbs, alldepth = [], [], []
            for nn in range(N_iter):
                allrays.append(train_dataset.all_rays_update['%05d'%nn])
                allrgbs.append(train_dataset.all_rgbs_update['%05d'%nn])
                alldepth.append(train_dataset.all_depth_update['%05d'%nn])
            allrays = torch.cat(allrays, 0)
            allrgbs = torch.cat(allrgbs, 0)
            alldepth = torch.cat(alldepth, 0)
            trainingSampler_pre = SimpleSampler(allrays.shape[0], args.batch_size)

            # clamp num of iterations in each epoch
            n_iters_eachepoch_stage2 = rays.shape[0]//args.batch_size + (rays.shape[0]%args.batch_size>0) + n_iters_eachepoch_stage1

            # reset learning rate
            grad_vars = tensorf.get_optparam_groups(args.lr_init, args.lr_basis)
            optimizer = torch.optim.Adam(grad_vars, betas=(0.9,0.99))
            trainingSampler2 = SimpleSampler(rays.shape[0], args.batch_size)

            args.lr_decay_iters = n_iters_eachepoch_stage2 * n_epoch_stage2_each
            lr_factor = args.lr_decay_target_ratio**(1/args.lr_decay_iters)
        elif epoch == n_total:
            allrays = torch.cat([allrays, rays], 0)
            allrgbs = torch.cat([allrgbs, rgbs], 0)
            alldepth = torch.cat([alldepth, depths], 0)
            trainingSampler_pre = SimpleSampler(allrays.shape[0], args.batch_size)
            n_iters_eachepoch_stage3 = allrays.shape[0]//args.batch_size + (allrays.shape[0]%args.batch_size>0)
            args.lr_decay_iters = n_iters_eachepoch_stage3 * 10
            lr_factor = args.lr_decay_target_ratio**(1/args.lr_decay_iters)
        print('Epoch %d' % global_epoch)
        print('lr0 = ', optimizer.state_dict()['param_groups'][0]['lr'], '\nlr1 = ', optimizer.state_dict()['param_groups'][-1]['lr'])

        if N_iter == 0:
            pbar = tqdm(range(n_iters_eachepoch_stage1), miniters=args.progress_refresh_rate, file=sys.stdout)
        else:
            pbar = tqdm(range(n_iters_eachepoch_stage2), miniters=args.progress_refresh_rate, file=sys.stdout)
        if epoch >= n_total:
            pbar = tqdm(range(n_iters_eachepoch_stage3), miniters=args.progress_refresh_rate, file=sys.stdout)
        local_step = 0
        for iteration in pbar:
            local_step += 1
            global_step += 1
            if N_iter==0 or iteration % 5==0 or epoch >= n_total:
                ray_idx = trainingSampler_pre.nextids()
                rays_train, rgb_train, depth_train = allrays[ray_idx], allrgbs[ray_idx].to(device), alldepth[ray_idx].to(device)
            else:
                ray_idx = trainingSampler2.nextids()
                rays_train, rgb_train, depth_train = rays[ray_idx], rgbs[ray_idx].to(device), depths[ray_idx].to(device)
            rgb_map, alphas_map, depth_map, weights, z_vals = renderer(rays_train, tensorf, chunk=args.batch_size,
                                    N_samples=nSamples, white_bg = white_bg, ndc_ray=ndc_ray, device=device, is_train=True)
            
            if torch.isnan(depth_map).any():
                depth_map = torch.where(torch.isnan(depth_map), torch.full_like(depth_map, 0), depth_map)
            
            # losses
            loss = torch.mean((rgb_map - rgb_train) ** 2)
            depth_loss = torch.mean((depth_map - depth_train) ** 2)
            weight_depth_loss = 0.005
            summary_writer.add_scalar('train/depth_loss', depth_loss.detach().item(), global_step=global_step)
          
            # transmittance loss
            delta = 0.1
            weight_trans_loss = 1e3
            mask_rays = (z_vals - depth_train[:, None] + delta)<0
            trans_loss = transmit_loss(weights, mask_rays)
            summary_writer.add_scalar('train/transmit_loss', trans_loss.detach().item(), global_step=global_step)

            total_loss = loss + weight_depth_loss * depth_loss + weight_trans_loss * trans_loss

            if TV_weight_density>0:
                TV_weight_density *= lr_factor
                loss_tv = tensorf.TV_loss_density(tvreg) * TV_weight_density
                total_loss = total_loss + loss_tv
                summary_writer.add_scalar('train/reg_tv_density', loss_tv.detach().item(), global_step=global_step)
            if TV_weight_app>0:
                TV_weight_app *= lr_factor
                loss_tv = tensorf.TV_loss_app(tvreg)*TV_weight_app
                total_loss = total_loss + loss_tv
                summary_writer.add_scalar('train/reg_tv_app', loss_tv.detach().item(), global_step=global_step)
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            del total_loss

            loss = loss.detach().item()
            depth_loss = depth_loss.detach().item() * weight_depth_loss
            trans_loss = trans_loss.detach().item() * weight_trans_loss
            PSNRs.append(-10.0 * np.log(loss) / np.log(10.0))
            summary_writer.add_scalar('train/PSNR', PSNRs[-1], global_step=global_step)
            summary_writer.add_scalar('train/mse', loss, global_step=global_step)

            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * lr_factor

            # Print the current values of the losses.
            if local_step % args.progress_refresh_rate == 0:
                str_descrip = f'Epoch {global_epoch:03d} Step {local_step:05d}:' \
                    + f' psnr: {float(np.mean(PSNRs)):.2f}' \
                    + f' l_rgb: {loss:.6f}' \
                    + f' l_depth: {depth_loss:.6f}' \
                    + f' l_trans: {trans_loss:.6f}'
                if type_depth_loss == 'ssi':
                    str_descrip += f' s: {s:.3f} t: {t:.3f}'
                pbar.set_description(str_descrip)
                PSNRs = []

        # if global_epoch == n_epoch_stage1 or (global_epoch>n_epoch_stage1 and (global_epoch-n_epoch_stage1)%n_epoch_stage2_each==0):
        #     tensorf.save(f'{logfolder}/{args.expname}_epoch{global_epoch:04d}.th')
        # visulization after each epoch
        if global_epoch % 50 == 0 or (global_epoch-n_epoch_stage1)%n_epoch_stage2_each==0 or global_epoch==n_epoch_stage1:
            PSNRs = evaluation(train_dataset, tensorf, args, renderer, f'{logfolder}/imgs_vis_support_view/', N_vis=-1,
                                    prtx=f'epoch{global_epoch:04d}_', N_samples=nSamples, white_bg=white_bg, ndc_ray=ndc_ray, 
                                    compute_extra_metrics=False, device=device, preview=True)
            if vis_inpaint_views:
                PSNRs_train = evaluation(train_dataset, tensorf, args, renderer, f'{logfolder}/imgs_vis_inpaint_view/', N_vis=-1,
                                        prtx=f'epoch{global_epoch:04d}_', N_samples=nSamples, white_bg=white_bg, ndc_ray=ndc_ray, 
                                        compute_extra_metrics=False, device=device, N_iter=N_iter, preview=False)
    # Visualization after pre-training stage
    tensorf.save(f'{logfolder}/{args.expname}_final.th')

    if args.render_train:
        os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
        train_dataset = dataset(args.datadir, split='train', is_stack=True)
        PSNRs_test = evaluation(train_dataset, tensorf, args, renderer, f'{logfolder}/imgs_train_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device, video_gen=True)
        # print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_test:
        os.makedirs(f'{logfolder}/imgs_test_all', exist_ok=True)
        PSNRs_test = evaluation(test_dataset, tensorf, args, renderer, f'{logfolder}/imgs_test_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device, video_gen=True)
        summary_writer.add_scalar('test/psnr_all', np.mean(PSNRs_test), global_step=iteration)
        # print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_path:
        c2ws = test_dataset.render_path
        # c2ws = test_dataset.poses
        # print('========>',c2ws.shape)
        os.makedirs(f'{logfolder}/imgs_path_all', exist_ok=True)
        evaluation_path(test_dataset,tensorf, c2ws, renderer, f'{logfolder}/imgs_path_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)


        
if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    
    args = config_parser()
    args.use_support_set = True
    args.update_known_views = False
    args.use_filter_filling_holes = True
    args.use_rendered_img_to_inpaint = True
    args.use_rendered_img_to_warp = True
    args.N_voxel_init=27000000
    args.N_voxel_final=27000000
    args.batch_size = 1024*16
    args.regen_pose = True
    print(args)
    set_seed(args.seed)

    if args.render_only and (args.render_test or args.render_path):
        render_test(args)
    else:
        reconstruction(args)

