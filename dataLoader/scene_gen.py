import json
import os, pdb

import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from tqdm import tqdm

from .ray_utils import *
from .scene_util import *
from utils import gt_warping
from .bilateral_filtering import sparse_bilateral_filtering, sparse_bilateral_filtering_4imgedge

def depth_process(depth, depthNettype, disp_min=0.167, disp_rescale=5.0, max_depth=7.0, push_depth=1.0):
    if depthNettype==0:
        # depth = cv2.blur(depth, (3, 3))
        depth = depth / 32768. - 1.
        # np.save(os.path.join(self.depth_path, '00000.npy'), depth)
        depth = depth - depth.min()
        # depth = cv2.blur(depth / depth.max(), ksize=(3, 3)) * depth.max()
        depth = (depth / depth.max()) * disp_rescale
        depth = (1. / np.maximum(depth, disp_min)).astype(np.float32)
    elif depthNettype==2:
        # depth = cv2.blur(depth, (3, 3))
        depth = (depth / 12000 + push_depth).astype(np.float32)
        # depth[depth>max_depth] = max_depth
    return depth

def produce_formatted_data(images, depths, masks, poses, intrinsic, H, W, mode='train'):
    """
    produce tensor data including all_rays/all_rgbs/all_depths in the format [masked(N*H*W), C], 
    and all_rays_split/all_rgbs_split/all_depths_split in the format [N, H, W, C]
    images: [N, H, W, 3]
    depths: [N, H, W]
    masks: [N, H, W]
    poses: [N2, 4, 4]
    intrinsic: [fx, fy, cx, cy]
    """
    transform = T.ToTensor()
    # ray directions for all pixels, same for all images (same H, W, focal)
    fx, fy, cx, cy = intrinsic
    directions = get_ray_directions(H, W, [fx, fy], center=[cx, cy])  # (h, w, 3)
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)

    num = poses.shape[0]
    poses_tensor = []
    all_rays, all_rgbs, all_depths = [], [], []
    all_rays_split, all_rgbs_split, all_depths_split = [], [], []
    if mode == 'train':
        for i in range(num):
            img = (images[i]).astype(np.float32)
            depth = (depths[i]).astype(np.float32)
            img = transform(img) # (3, h, w)
            img = img.view(-1, W*H).permute(1, 0)  # --> (h*w, 3) RGB
            if img.shape[-1]==4:
                img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB
            depth = transform(depth)
            depth = depth.view(-1, W*H).permute(1, 0)  # --> (h*w, 1)
            mask = masks[i]
            mask = torch.tensor(mask).view(W*H)
            # add masked pixels
            all_rgbs += [img[mask>0.5]]
            all_depths += [depth[mask>0.5,0]]
            # add whole pixels
            all_rgbs_split += [img]
            all_depths_split += [depth[:,0]]

            c2w = torch.FloatTensor(poses[i])
            poses_tensor += [c2w]
            rays_o, rays_d = get_rays(directions, c2w)  # both (h*w, 3)
            # add masked pixels
            all_rays += [torch.cat([rays_o[mask>0.5], rays_d[mask>0.5]], 1)]
            # add whole pixels
            all_rays_split += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)

        poses_tensor = torch.stack(poses_tensor)
        all_rays = torch.cat(all_rays, 0)  # (mask(N*h*w), 6)
        all_rgbs = torch.cat(all_rgbs, 0)  # (mask(N*h*w), 3)
        all_depths = torch.cat(all_depths, 0)  # (mask(N*h*w), )
        all_rays_split = torch.stack(all_rays_split, 0)  # (N, h*w, 6)
        all_rgbs_split = torch.stack(all_rgbs_split, 0).reshape(-1, H, W, 3)
        all_depths_split = torch.stack(all_depths_split, 0).reshape(-1, H, W)

        return all_rays, all_rgbs, all_depths, all_rays_split, all_rgbs_split, all_depths_split, poses_tensor
    
    elif mode == 'test':
        for i in range(num):
            c2w = torch.FloatTensor(poses[i])
            poses_tensor += [c2w]
            rays_o, rays_d = get_rays(directions, c2w)  # both (h*w, 3)
            # add whole pixels
            all_rays_split += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)
        poses_tensor = torch.stack(poses_tensor)
        all_rays_split = torch.stack(all_rays_split, 0)  # (N, h*w, 6)

        return all_rays_split, poses_tensor
    



class SceneGenDataset(Dataset):
    def __init__(self, args, split='train', scene_bound=8. , is_stack=False, white_bg=True, N_vis=-1, hw=(512, 512), crop_square=False):

        self.args = args
        self.N_vis = N_vis
        self.root_dir = args.datadir
        self.prompt = args.prompt
        self.negative_prompt = 'blurry, bad art, blurred, text, watermark'
        self.split = split
        self.is_stack = is_stack
        self.crop_square = crop_square
        self.hw = hw
        self.scene_bound = scene_bound
        self.define_transforms()
        self.scene_bbox = torch.tensor([[-self.scene_bound, -self.scene_bound, -self.scene_bound], 
                                        [self.scene_bound, self.scene_bound, self.scene_bound]])
        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        self.white_bg = white_bg
        self.near_far = [0.5, 8.0]
        self.center = torch.mean(self.scene_bbox, axis=0).float().view(1, 1, 3)
        self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3)

        self.rgb_path = os.path.join(args.datadir, 'rgbs')
        self.depth_path = os.path.join(args.datadir, 'depth')
        self.cam_path = os.path.join(args.datadir, 'cam')
        self.img_id = 0
        if not os.path.isfile(os.path.join(self.rgb_path, '%05d.png' % self.img_id)):
            self.do_generation = True
        else:
            self.do_generation = False
        if (not os.path.isfile(os.path.join(self.depth_path, '%05d.png' % self.img_id))) and (not os.path.isfile(os.path.join(self.depth_path, '%05d.npy' % self.img_id))):
            self.do_depth_esti = True
        else:
            self.do_depth_esti = False
        if os.path.isfile(os.path.join(self.cam_path, '%05d_pose.npy' % self.img_id)) and \
        os.path.isfile(os.path.join(self.cam_path, 'intrinsic.npy')) and not args.regen_pose:
            self.do_pose_gen = False
        else:
            self.do_pose_gen = True
        os.makedirs(self.rgb_path, exist_ok=True)
        os.makedirs(self.depth_path, exist_ok=True)
        os.makedirs(self.cam_path, exist_ok=True)
        if split != 'train':
            self.do_pose_gen = False
        self.num_training = args.num_training
        self.pose_traj = args.pose_traj  # local:'local_fixed'/'local_double';  global: 'circle'/'rectangle'
        self.use_support_set = args.use_support_set
        self.read_meta()
        self.define_proj_mat()

    def scale_poses(self, poses, scale):
        for ii in range(poses.shape[0]):
            poses[ii, :3, 3] *= scale
        return poses

        
    def read_meta(self):
        # generate the first image accorrding to the prompt
        if self.do_generation:
            from scripts.text2img_sdm import text2img_sdm
            cmd_img_gen = 'gen'
            while True:
                if cmd_img_gen == 'gen' or cmd_img_gen == 'g' or cmd_img_gen == 'r':
                    image = text2img_sdm(self.prompt, negative_prompt=self.negative_prompt)
                    image.save(os.path.join(self.rgb_path, '%05d.png' % self.img_id))
                if cmd_img_gen == 'ok':
                    break
                print("path:", self.rgb_path)
                print("Current prompt:", self.prompt)
                print("Please confirm the next process!\n'gen/g/r': re-generation of rgb image; \n'ok': jump out of the loop and continue with subsequent processing.")
                cmd_img_gen = input("Input 'cmd_img_gen':")
            img_init = (np.array(image) / 255.).astype(np.float32)
        else:
            img_init = cv2.imread(os.path.join(self.rgb_path, '%05d.png' % self.img_id), cv2.IMREAD_UNCHANGED)
            convert_fn = cv2.COLOR_BGR2RGB
            img_init = (cv2.cvtColor(img_init, convert_fn) / 255.).astype(np.float32)
        H, W = img_init.shape[:2] # h,w,3
        if H == W:
            self.crop_square = False
        else:
            self.crop_square = True
        if self.crop_square:
            l_min = min(H, W)
            img_init = img_init[:l_min, :l_min]
            H, W = img_init.shape[:2] # h,w,3

        # estimate the depth 
        depthNet=2 # 0: MiDas, 2: LeRes
        push_depth = self.args.push_depth
        if self.do_depth_esti:
            from scripts.depth_esti_boosting import depth_esti_boosting
            depth_init = depth_esti_boosting(
                image_dir=self.rgb_path, 
                result_dir=self.depth_path, 
                image_name='%05d' % self.img_id,
                depthNet=depthNet,  # 0: MiDas, 2: LeRes
                )
            depth_init = depth_process(depth_init, depthNet, disp_min=0.14, disp_rescale=10.0, max_depth=7.2, push_depth=push_depth)
        else:
            depth_fname = os.path.join(self.depth_path, '%05d.png' % self.img_id)
            try:
                depth_init = cv2.imread(depth_fname, cv2.IMREAD_UNCHANGED).astype(np.float64)
                depth_init = depth_process(depth_init, depthNet, disp_min=0.14, disp_rescale=10.0, max_depth=7.2, push_depth=push_depth)
            except:
                depth_init = np.load(depth_fname.replace('png', 'npy'))
                depth_init = (depth_init / 2).astype(np.float32)
        imageio.imwrite(os.path.join(self.depth_path, '%05d_before_filter.png' % self.img_id), depth_init)
        if self.crop_square:
            depth_init = depth_init[:l_min, :l_min]
        self.scale = self.hw[0]/H
        if self.hw[0] != H or self.hw[1] != W:
            depth_init = cv2.resize(depth_init, (self.hw[1], self.hw[0]), interpolation=cv2.INTER_NEAREST)
            img_init = cv2.resize(img_init, (self.hw[1], self.hw[0]), interpolation=cv2.INTER_NEAREST)
            H, W = img_init.shape[:2] # h,w,3

        # depth process: bilateral_filtering, refer to <3D photo inpainting>
        vis_photos, vis_depths = sparse_bilateral_filtering(
            depth_init.copy(), img_init.copy(), filter_size=[5, 5, 3, 3], 
            depth_threshold=0.02, num_iter=4, HR=False, mask=None)
        depth_init = vis_depths[-1]
        img_init = vis_photos[-1]
        imageio.imwrite(os.path.join(self.depth_path, '%05d_after_filter.png' % self.img_id), depth_init)
        imageio.imwrite(os.path.join(self.rgb_path, '%05d_after_filter.png' % self.img_id), img_init)

        # generate training poses
        if self.do_pose_gen:
            ## intrinsic
            intri = np.array([[max(H, W), 0, W // 2], [0, max(H, W), H // 2],
                            [0, 0, 1]]).astype(np.float32)
            if intri.max() > 1:
                intri[0, :] = intri[0, :] / float(W)
                intri[1, :] = intri[1, :] / float(H)
            intri = intri * np.array([[W], [H], [1.]])
            np.save(os.path.join(self.cam_path, 'intrinsic.npy'), intri)
            fx, fy, cx, cy = intri[0, 0], intri[1, 1], intri[0, 2], intri[1, 2]
            
            ## extrinsic
            pose_ref = np.eye(4)
            if self.pose_traj == 'local_fixed':
                poses = get_local_fixed_poses2(pose_ref, angle=self.args.angle, range_center=self.args.trans_range, range_yaw=0.6, range_pitch=0.2)
            elif self.pose_traj == 'local_double':
                poses = get_double_circle_poses_from_center_pose(pose_ref, self.num_training, random_sample=True)
            elif self.pose_traj == 'local_circle':
                poses = get_local_poses3(pose_ref, range_center=self.args.trans_range)
            elif self.pose_traj == 'local_r2l':
                poses = get_r2l_pose(pose_ref, range_center=self.args.trans_range, num_frame=None)
            else:
                poses = cam_traj_gen(self.num_training, traj_type=self.pose_traj, random_sample=False, radius=self.args.trans_range, pose_ref=pose_ref, for_training=True)

            num_poses = poses.shape[0]
            for i in range(num_poses):
                c2w = poses[i]
                np.save(os.path.join(self.cam_path, '%05d_pose.npy'%i), c2w)
        else:
            intrinsic_matrix = np.load(os.path.join(self.cam_path, 'intrinsic.npy'))
            fx, fy, cx, cy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1], intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
            pose_list = sorted(glob.glob(os.path.join(self.cam_path, '*_pose.npy')))
            num_poses = len(pose_list)
            poses = []
            for i in range(num_poses):
                pose = np.load(pose_list[i])
                poses.append(pose)
            poses = np.stack(poses)

        if self.split != 'train':
            if self.pose_traj == 'local_fixed' or self.pose_traj == 'local_double' or self.pose_traj == 'local_circle':
                vposes = get_circle_spiral_poses_from_pose(poses[0], N_views=120, n_r=1, angle_h_start=self.args.angle-0.03, trans_start=self.args.trans_range, use_rand=False)
            elif self.pose_traj == 'local_r2l':
                vposes = get_r2l_pose(poses[0], range_center=self.args.trans_range, num_frame=120)
            else:
                if self.pose_traj == 'circle0':
                    self.pose_traj = 'circle'
                elif 'circle0_' in self.pose_traj:
                    traj_replace = 'circle_' + self.pose_traj.split('_')[-1]
                    self.pose_traj = traj_replace
                vposes = cam_traj_gen(360, traj_type=self.pose_traj, random_sample=False, radius=self.args.trans_range, pose_ref=poses[0])
            num_poses = vposes.shape[0]

        w, h = self.hw[1], self.hw[0]
        self.img_wh = [w, h]
        # self.focal_x, self.focal_y = fx*self.scale, fy*self.scale # original focal length
        # self.cx, self.cy = cx*self.scale, cy*self.scale
        self.focal_x, self.focal_y = fx, fy # original focal length
        self.cx, self.cy = cx, cy

        # ray directions for all pixels, same for all images (same H, W, focal)
        # self.directions = get_ray_directions(h, w, [self.focal_x,self.focal_y], center=[self.cx, self.cy])  # (h, w, 3)
        # self.directions = self.directions / torch.norm(self.directions, dim=-1, keepdim=True)
        self.intrinsics = torch.tensor([[self.focal_x,0,self.cx],[0,self.focal_y,self.cy],[0,0,1]]).float()

        # pose for support set
        if self.pose_traj == 'local_circle':
            poses_sprt = get_local_poses3(poses[0], range_center=self.args.trans_range)
        else:
            poses_sprt = get_local_fixed_poses2(poses[0], angle=self.args.angle, range_center=self.args.trans_range, range_yaw=0.6, range_pitch=0.2)
            # poses_sprt = get_local_fixed_poses2(poses[0], angle=0, range_center=self.args.trans_range, range_yaw=0.6, range_pitch=0.2)
        
        self.poses_support = torch.FloatTensor(poses_sprt)

        if self.split == 'train':
            print(f'Loading <{self.split}> data, {poses.shape[0]} views.')
            self.all_rays_sprt_split, self.all_rgbs_sprt_split, self.all_depth_sprt_split = None, None, None  # used for rendering of initialized model
            if self.use_support_set:
                rgbs_warp, masks_warp, depth_warp = gt_warping(img_init, depth_init, poses_sprt[0], poses_sprt[1:], h, w,
                                                                intrinsic=(self.focal_x, self.focal_y, self.cx, self.cy), 
                                                                logpath=self.root_dir, warp_depth=True, bilinear_splat=True)
                images = np.concatenate([np.expand_dims(img_init,0), rgbs_warp], 0)
                masks = np.concatenate([np.ones_like(masks_warp[:1]), masks_warp], 0)
                depths = np.concatenate([np.expand_dims(depth_init,0), depth_warp], 0)
                self.all_rays, self.all_rgbs, self.all_depth, \
                    self.all_rays_split, self.all_rgbs_split, self.all_depth_split, _ = \
                    produce_formatted_data(images, depths, masks, poses_sprt, \
                                        (self.focal_x, self.focal_y, self.cx, self.cy), 
                                        H, W, mode='train')
                self.all_rays_sprt_split, self.all_rgbs_sprt_split, self.all_depth_sprt_split = self.all_rays_split, self.all_rgbs_split, self.all_depth_split
            else:
                mask_ones = np.ones([1, H, W], dtype=np.int64)
                self.all_rays, self.all_rgbs, self.all_depth, \
                    self.all_rays_split, self.all_rgbs_split, self.all_depth_split, _ = \
                    produce_formatted_data(np.expand_dims(img_init,0), np.expand_dims(depth_init,0), mask_ones, poses_sprt[:1], \
                                        (self.focal_x, self.focal_y, self.cx, self.cy), 
                                        H, W, mode='train')
                self.all_rays_sprt_split, _ = produce_formatted_data(None, None, None, poses_sprt, \
                                        (self.focal_x, self.focal_y, self.cx, self.cy), 
                                        H, W, mode='test')
                self.all_rgbs_sprt_split = self.all_rgbs_split.repeat([poses_sprt.shape[0], 1, 1, 1])
                self.all_depth_sprt_split = self.all_depth_split.repeat([poses_sprt.shape[0], 1, 1, 1])
            # used for inpainting during training stage
            self.all_rays_gen_split, self.poses = produce_formatted_data(None, None, None, poses, \
                                        (self.focal_x, self.focal_y, self.cx, self.cy), 
                                        H, W, mode='test')  
            self.all_rgbs_gen_split, self.all_depth_gen_split = self.all_rgbs_split[:1], self.all_depth_split[:1]
            self.all_masks_gen_split = torch.ones([1, H, W])
            # used for update known views after inpainting
            self.all_rays_update, self.all_rgbs_update, self.all_depth_update = {}, {}, {}
            self.all_rays_update['%05d'%self.img_id] = self.all_rays
            self.all_rgbs_update['%05d'%self.img_id] = self.all_rgbs
            self.all_depth_update['%05d'%self.img_id] = self.all_depth
        else:  # self.split == 'test'
            img_eval_interval = 1 if self.N_vis < 0 else num_poses // self.N_vis
            idxs = list(range(0, num_poses, img_eval_interval))
            vposes = vposes[idxs]
            print(f'Loading <{self.split}> data, {vposes.shape[0]} views.')
            self.all_rays_split, self.poses = produce_formatted_data(None, None, None, vposes, \
                                        (self.focal_x, self.focal_y, self.cx, self.cy), 
                                        H, W, mode='test')

        """# self.poses = []
        # self.all_rays = []
        # self.all_rgbs = []
        # self.all_masks = []
        # self.all_depth = []
        
        # if self.split == 'train':
        #     idxs = list(range(0, num_poses))
        #     self.poses_gen = []
        #     self.all_rays_gen = []
        #     self.all_rgbs_gen = []
        #     self.all_masks_gen = []
        #     self.all_depth_gen = []
        # else:
        #     idxs = list(range(0, num_poses, img_eval_interval))

        # for i in tqdm(idxs, desc=f'Loading camera data {self.split} ({len(idxs)})'):
        #     if i == 0 and self.split == 'train':
        #         img = img_init.copy()
        #         depth = depth_init.copy()
        #     else:
        #         img = np.random.random(img_init.shape).astype(np.float32)
        #         depth = (np.random.random(depth_init.shape)*np.max(depth_init)).astype(np.float32)
        #     img = self.transform(img) # (3, h, w)
        #     img = img.view(-1, w*h).permute(1, 0)  # --> (h*w, 3) RGB
        #     if img.shape[-1]==4:
        #         img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB

        #     depth = self.transform(depth)
        #     depth = depth.view(-1, w*h).permute(1, 0)  # --> (h*w, 1)

        #     if self.do_pose_gen:
        #         pose = poses[i]
        #     else:
        #         if self.split == 'train':
        #             pose = np.load(pose_list[i])
        #         else:
        #             pose = vposes[i]
        #     c2w = torch.FloatTensor(pose)
        #     rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)

        #     if self.split == 'train':
        #         self.all_rgbs_gen += [img]
        #         self.all_depth_gen += [depth[:,0]]
        #         self.poses_gen += [c2w]
        #         self.all_rays_gen += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)
                
        #         if i == 0:
        #             self.all_rgbs += [img]
        #             self.all_depth += [depth[:,0]]
        #             self.poses += [c2w]
        #             self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)
        #     else:
        #         self.all_rgbs += [img]
        #         self.all_depth += [depth[:,0]]
        #         self.poses += [c2w]
        #         self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)

        # self.poses = torch.stack(self.poses)

        
        # if not self.is_stack:
        #     self.all_rays = torch.cat(self.all_rays, 0)  # (len(self.meta['frames])*h*w, 3)
        #     self.all_rgbs = torch.cat(self.all_rgbs, 0)  # (len(self.meta['frames])*h*w, 3)
        #     self.all_depth = torch.cat(self.all_depth, 0)  # (len(self.meta['frames])*h*w, 1)
        # else:
        #     self.all_rays = torch.stack(self.all_rays, 0)  # (len(self.meta['frames]),h*w, 3)
        #     self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1,*self.img_wh[::-1], 3)  # (len(self.meta['frames]),h,w,3)
        #     self.all_depth = torch.stack(self.all_depth, 0).reshape(-1,*self.img_wh[::-1], 1) 
        # if self.split == 'train' and len(self.poses_gen)>0:
        #     self.poses_gen = torch.stack(self.poses_gen)
        #     self.all_rays_gen = torch.stack(self.all_rays_gen, 0)
        #     self.all_rgbs_gen = torch.stack(self.all_rgbs_gen, 0).reshape(-1,*self.img_wh[::-1], 3)
        #     self.all_depth_gen = torch.stack(self.all_depth_gen, 0).reshape(-1,*self.img_wh[::-1], 1) """


    def define_transforms(self):
        self.transform = T.ToTensor()
        
    def define_proj_mat(self):
        self.proj_mat = self.intrinsics.unsqueeze(0) @ torch.inverse(self.poses)[:,:3]

    def world2ndc(self,points,lindisp=None):
        device = points.device
        return (points - self.center.to(device)) / self.radius.to(device)
        
    def __len__(self):
        return len(self.all_rgbs)

    def __getitem__(self, idx):
        sample = {'rays': self.all_rays[idx],
                    'rgbs': self.all_rgbs[idx]}

        return sample
