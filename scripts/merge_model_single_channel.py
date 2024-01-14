import torch, glob, os, sys
import shutil, random, cv2
import imageio.v2 as imageio
import numpy as np
from tqdm.auto import tqdm
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))
from models.merge_model_sc.config import Config
from models.merge_model_sc.pix2pix4depth_model import Pix2Pix4DepthModel
from dataLoader.scene_util import get_local_fixed_poses2
from dataLoader.bilateral_filtering import sparse_bilateral_filtering
from scripts.Warper import Warper

class depth_merge_model():
    def __init__(self, ckpt_name='init', device='cuda'):
        super().__init__()
        opt = Config('./models/merge_model_sc/Options.yml')
        opt.checkpoints_dir = 'weights/merge_model_sc/checkpoints'
        self.pix2pixmodel = Pix2Pix4DepthModel(opt, device=device)
        self.pix2pixmodel.save_dir = opt.checkpoints_dir
        self.pix2pixmodel.load_networks(ckpt_name)
        self.device = device
    
    def save_models(self, ckpt_name='latest'):
        self.pix2pixmodel.save_networks(ckpt_name)

    # def run_numpy(self, depth_guide, depth_original, outsize=512):
    #     self.pix2pixmodel.set_input(depth_guide, depth_original)
    #     out = self.pix2pixmodel.netG(self.pix2pixmodel.real_A)
    #     out = torch.nn.functional.interpolate(out, size=[outsize, outsize], mode='nearest').squeeze()
    #     return  out

    def run_finetune_numpy(self, depth_guide, depth_original, mask_ref, outsize=512, lr=1e-5, iter=500):
        MSEloss = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.pix2pixmodel.netG.parameters(), lr=lr, betas=(0.9, 0.99))

        depth_guide = torch.from_numpy(depth_guide).to(self.device)
        depth_original = torch.from_numpy(depth_original).to(self.device)
        depth_original = torch.nn.functional.interpolate(depth_original[None, None], size=[1024, 1024], mode='nearest')
        mask_ref = torch.from_numpy(mask_ref).to(self.device)

        self.pix2pixmodel.set_input_tensor(depth_original)
        pbar = tqdm(range(iter))
        for i in pbar:
            optimizer.zero_grad()
            out = self.pix2pixmodel.netG(self.pix2pixmodel.real_A)
            out = torch.nn.functional.interpolate(out, size=[outsize, outsize], mode='nearest').squeeze()
            loss = MSEloss(out * mask_ref, depth_guide * mask_ref)
            loss.backward()
            optimizer.step()
        out = self.pix2pixmodel.netG(self.pix2pixmodel.real_A)
        out = torch.nn.functional.interpolate(out, size=[outsize, outsize], mode='nearest').squeeze()
        return  out
    
    def run(self, depth_original, outsize=512):
        self.pix2pixmodel.set_input_tensor(depth_original)
        out = self.pix2pixmodel.netG(self.pix2pixmodel.real_A)
        out = torch.nn.functional.interpolate(out, size=[outsize, outsize], mode='nearest').squeeze()
        return  out

class data_loader(torch.utils.data.Dataset):
    def __init__(self, imgs_path_list, mask_path_list):
        super().__init__()
        self.imgs_path_list = imgs_path_list
        self.mask_path_list = mask_path_list
        self.num_mask = len(self.mask_path_list)

        ## generate warp masks
        # mask_path = 'data/data_for_training/000_warp_masks'
        # os.makedirs(mask_path, exist_ok=True)
        # angle_list = []
        # pose_ref = np.eye(4)
        # for i in range(50):
        #     angle = random.randint(5, 30)/180*np.pi
        #     trans = random.uniform(0.05, 0.3)
        #     poses = get_local_fixed_poses2(pose_ref, angle=angle, range_center=trans, range_yaw=0.6, range_pitch=random.choice([0.2, 0.3]))
        #     angle_list.append(poses[1:])
        # self.angle_list = np.concatenate(angle_list, axis=0)
        # self.num_pose = len(self.angle_list)
        # img = cv2.imread(self.imgs_path_list[0], cv2.IMREAD_UNCHANGED)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8)
        # H, W = img.shape[:2]
        # warper = Warper()
        # transformation1 = np.linalg.inv(pose_ref)
        # intrinsic_mtx = np.array([[max(H, W), 0, W // 2], [0, max(H, W), H // 2],
        #                     [0, 0, 1]]).astype(np.float32)
        # for i in range(len(imgs_path_list)):
        #     pose_id = random.choice(range(self.num_pose))
        #     pose = self.angle_list[pose_id]
        #     transformation2 = np.linalg.inv(pose)

        #     depth_path = self.imgs_path_list[i].replace('rgbs', 'depth')
        #     depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float64)
        #     push_depth = 0.5
        #     depth = cv2.blur(depth, (3, 3))
        #     depth = (depth / 10000 + push_depth).astype(np.float32)
        #     depth[depth>7.0] = 7.0

        #     _, vis_depths = sparse_bilateral_filtering(
        #         depth.copy(), img.copy(), filter_size=[7, 7, 5, 5, 5], 
        #         depth_threshold=0.03, num_iter=5, HR=False, mask=None)
        #     depth = vis_depths[-1]

        #     warped_frame2, mask2, warped_depth2, flow12 = warper.forward_warp(img, None, depth, transformation1, transformation2, intrinsic_mtx, None)
        #     imageio.imwrite(os.path.join(mask_path, '%05d.png'%i), mask2*1)

    def __len__(self):
        return len(self.imgs_path_list)
    
    def __getitem__(self, index):
        img_path = self.imgs_path_list[index]
        depth_path = img_path.replace('rgbs', 'depth')
        mask_id = random.choice(range(self.num_mask))
        mask_path = self.mask_path_list[mask_id]

        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float64)
        depth = depth / 32768. - 1.  #[-1, 1]
        H, W = depth.shape

        depth_2 = depth.copy()
        depth_2 = (depth_2-depth_2.min())/(depth_2.max()-depth_2.min())
        shift_random = random.uniform(0, 1.0)
        degree_rand = random.randint(30, 60)
        depth_scale1 = random.uniform(0.9, 1.1)*pow(depth_2, 1/degree_rand)
        depth_in = (depth_2+shift_random)*depth_scale1
        depth_in = (depth_in-depth_in.min())/(depth_in.max()-depth_in.min())*2.-1. #[-1, 1]

        mask = imageio.imread(mask_path)/255.
        depth_ref = depth*mask

        data = {'depth_ref': torch.Tensor(depth_ref),
                'depth_in': torch.Tensor(depth_in),
                'depth_out': torch.Tensor(depth),
                'mask': torch.Tensor(mask),
                }
        

        return data




if __name__ == "__main__":
    device = torch.device("cuda")
    data_path = 'data/data_for_training'
    folder_list = os.listdir(data_path)
    imgs_list = []
    for folder in folder_list:
        imgs_list += [x for x in glob.glob(os.path.join(data_path, folder, 'rgbs', "*"))
                    if (x.endswith(".jpg") or x.endswith(".png"))]
    num_total = len(imgs_list)
    masks_list = sorted([x for x in glob.glob(os.path.join(data_path, '000_warp_masks', "*"))
                    if (x.endswith(".jpg") or x.endswith(".png"))])
    dataset = data_loader(imgs_list, masks_list)
    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=20,
                                               shuffle=True,
                                               num_workers=8)
    MSEloss = torch.nn.MSELoss()
    merge_model = depth_merge_model(ckpt_name='init', device=device)
    optimizer_G = torch.optim.Adam(merge_model.pix2pixmodel.netG.parameters(), lr=1e-4, betas=(0.9, 0.999))
    
    round_epoch = 1000
    model_path = 'models/merge_model_sc'
    os.makedirs(os.path.join(model_path, 'training_test_imgs'), exist_ok=True)
    for epoch in range(round_epoch):
        current_epoch = epoch + 1
        pbar = tqdm(train_loader)
        # for data in train_loader:
        i = 0
        for data in pbar:
            i += 1
            optimizer_G.zero_grad()
            depth_ref, depth_in, depth_out = data['depth_ref'].to(device), data['depth_in'].to(device), data['depth_out'].to(device)
            depth_ref = torch.nn.functional.interpolate(depth_ref.unsqueeze(1),(1024,1024),mode='nearest',align_corners=False)
            depth_in = torch.nn.functional.interpolate(depth_in.unsqueeze(1),(1024,1024),mode='nearest',align_corners=False)
            out = merge_model.run(depth_in, outsize=512)
            loss = MSEloss(out, depth_out)
            loss.backward()
            optimizer_G.step()
            if (i) % 2 == 0:
                pbar.set_description(
                        f'Epoch {current_epoch:03d} Step {(i):05d}:'
                        + f' loss = {float(loss):.6f}'
                    )

        imageio.imwrite(os.path.join(model_path, 'training_test_imgs', '%05d_test_out_merge_out.png'%current_epoch), ((out[0]+1)/2).detach().cpu().numpy())
        imageio.imwrite(os.path.join(model_path, 'training_test_imgs', '%05d_test_out_merge_ref.png'%current_epoch), ((data['depth_ref'][0]+1)/2).numpy())
        imageio.imwrite(os.path.join(model_path, 'training_test_imgs', '%05d_test_out_merge_in.png'%current_epoch), ((data['depth_in'][0]+1)/2).numpy())
        if current_epoch%20==0:
            merge_model.save_models(ckpt_name='epoch_%05d'%current_epoch)




    

