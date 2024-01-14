import os
import glob
import numpy as np
import json
import cv2
import random, math, imageio

def read_files(basedir, rgb_file, depth_file):
    fname = os.path.join(basedir, rgb_file)
    img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
    if img.shape[-1] == 4:
        convert_fn = cv2.COLOR_BGRA2RGBA
    else:
        convert_fn = cv2.COLOR_BGR2RGB
    img = (cv2.cvtColor(img, convert_fn) / 255.).astype(np.float32) # keep 4 channels (RGBA) if available
    depth_fname = os.path.join(basedir, depth_file)
    depth = cv2.imread(depth_fname, cv2.IMREAD_UNCHANGED).astype(np.float64)
    return img, depth

def load_ground_truth_depth(basedir, train_filenames, image_size, depth_scaling_factor):
    H, W = image_size
    gt_depths = []
    gt_valid_depths = []
    for filename in train_filenames:
        filename = filename.replace("rgb", "target_depth")
        filename = filename.replace(".jpg", ".png")
        gt_depth_fname = os.path.join(basedir, filename)
        if os.path.exists(gt_depth_fname):
            gt_depth = cv2.imread(gt_depth_fname, cv2.IMREAD_UNCHANGED).astype(np.float64)
            gt_valid_depth = gt_depth > 0.5
            gt_depth = (gt_depth / depth_scaling_factor).astype(np.float32)
        else:
            gt_depth = np.zeros((H, W))
            gt_valid_depth = np.full_like(gt_depth, False)
        gt_depths.append(np.expand_dims(gt_depth, -1))
        gt_valid_depths.append(gt_valid_depth)
    gt_depths = np.stack(gt_depths, 0)
    gt_valid_depths = np.stack(gt_valid_depths, 0)
    return gt_depths, gt_valid_depths


def eliminate_depth_error(depth_stand, depth_src, pose_stand, pose_src, intri, img_std=None, img_src=None):
    """
    pose_*: 4*4, c2w
    intri: 3*3
    """
    H, W = depth_stand.shape
    w2c_src = np.linalg.inv(pose_src)
    std2src = np.transpose(np.dot(w2c_src, pose_stand))
    y = np.linspace(0, H - 1, H)
    x = np.linspace(0, W - 1, W)
    xx, yy = np.meshgrid(x, y)
    fx = intri[0, 0]
    cx = intri[0, 2]
    fy = intri[1, 1]
    cy = intri[1, 2]
    x = (xx - cx) / fx * depth_stand
    y = (yy - cy) / fy * depth_stand
    coords = np.zeros((H, W, 4))
    coords[:, :, 0] = x
    coords[:, :, 1] = -y
    coords[:, :, 2] = -depth_stand
    coords[:, :, 3] = 1

    coords_c1 = coords.transpose(2, 0, 1).reshape(4, -1)
    coords_c2 = np.matmul(np.dot(w2c_src, pose_stand), coords_c1)
    coords_c2 = coords_c2.reshape(4, H, W).transpose(1, 2, 0)
    z_src = coords_c2[:, :, 2]
    x = -coords_c2[:, :, 0] / (1e-8 + z_src) * fx + cx
    y = coords_c2[:, :, 1] / (1e-8 + z_src) * fy + cy
    # Round off the pixels in new virutal image and fill cracks with white
    x = (np.round(x)).astype(np.int16)
    y = (np.round(y)).astype(np.int16)

    myMap = np.zeros((H, W), dtype=np.uint8)
    output_image = np.ones((H, W, 3)) * 255
    points_warped = np.zeros((H, W, 3))
    list_filled = []
    for i in range(H):
        for j in range(W):
            x_o = x[i, j]
            y_o = y[i, j]
            if (x_o >= 0 and x_o < W and y_o >= 0 and y_o < H):
                if (myMap[y_o, x_o] == 0):
                    if img_std is not None:
                        output_image[y_o, x_o, :] = img_std[i, j, :] * 255
                    points_warped[y_o, x_o] = coords_c2[i, j, :3]
                    myMap[y_o, x_o] = 1
                    list_filled.append((y_o, x_o))
    
    if img_std is not None:
        output_image = output_image.astype(np.uint8)
        # imageio.imwrite('warped.png', output_image)
        # imageio.imwrite('tar.png', (img_src*255).astype(np.uint8))
        # imageio.imwrite('std.png', (img_std*255).astype(np.uint8))
    # compute points_src in its camera space
    y0 = np.linspace(0, H - 1, H)
    x0 = np.linspace(0, W - 1, W)
    xx, yy = np.meshgrid(x0, y0)
    x0 = (xx - cx) / fx * depth_src
    y0 = (yy - cy) / fy * depth_src
    points_src = np.zeros((H, W, 3))
    points_src[:, :, 0] = x0
    points_src[:, :, 1] = -y0
    points_src[:, :, 2] = -depth_src

    # calculate the scale
    scales = []
    num_max = min(len(list_filled), 1001)
    list_sample = random.sample(list_filled, num_max)
    for ii in range(len(list_sample)-1):
        y_o1, x_o1 = list_sample[ii]
        y_o2, x_o2 = list_sample[ii+1]
        p1 = points_warped[y_o1, x_o1]
        p2 = points_warped[y_o2, x_o2]
        dx1 = np.sqrt(sum((p1-p2) ** 2))
        p1 = points_src[y_o1, x_o1]
        p2 = points_src[y_o2, x_o2]
        dx2 = np.sqrt(sum((p1-p2) ** 2))
        scales.append(dx2/dx1)
    scales = np.stack(scales)
    scale = np.average(scales)

    # calculate the shift
    depth_src_scaled = depth_src*scale
    shifts = (depth_src_scaled - (-points_warped[:,:,2])) * myMap
    shift = np.sum(shifts)/np.sum(myMap)
    depth_src_shift = depth_src_scaled - shift

    uncert = (depth_src_shift - (-points_warped[:,:,2])) * myMap + 0.1 * (1-myMap)
    
    return depth_src_shift, uncert

def normalize(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.eye(4)
    m[:3] = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def R_x(theta):
    out = np.array([[1,               0,               0],
                    [0, math.cos(theta),-math.sin(theta)],
                    [0, math.sin(theta), math.cos(theta)]
                    ])
    return out
def R_y(theta):
    out = np.array([[math.cos(theta), 0, math.sin(theta)],
                    [0,               1,      0         ],
                    [-math.sin(theta),0, math.cos(theta)]
                    ])
    return out
def R_z(theta):
    out = np.array([[math.cos(theta),-math.sin(theta), 0],
                    [math.sin(theta), math.cos(theta), 0],
                    [              0,               0, 1]
                    ])
    return out
def eulerangle2rotation(theta):
    return np.dot(R_z(theta[2]), np.dot(R_y(theta[1]), R_x(theta[0])))

def cam_traj_gen(num_frames, traj_type='rectangle', random_sample=False, radius=0.5, pose_ref=None, for_training=False):
    if 'circle0_' in traj_type:
        try: 
            circle_angle = float(traj_type.split('_')[-1])
        except ValueError:
            print('traj_type circle0_angle: angle should be a figure!')
        traj_type = 'circle0_angle'
    if 'circle_' in traj_type:
        try: 
            circle_angle = float(traj_type.split('_')[-1])
        except ValueError:
            print('traj_type circle0_angle: angle should be a figure!')
        traj_type = 'circle_angle'
    if 'line_' in traj_type: # line_pitch_yaw_distance
        try:
            angle_pitch = float(traj_type.split('_')[1])
            angle_yaw = float(traj_type.split('_')[2])
            line_length = float(traj_type.split('_')[-1])
        except ValueError:
            print('traj_type line_pitch_yaw_distance: pitch/yaw/distance should be a figure!')
        traj_type = 'line_move'


    if traj_type == 'rectangle':
        if num_frames < 36:
            num_frames = 36
        corner_points = np.array([[1, 0, 1],
                                  [0, 0, 1],
                                  [0, 0, 0],
                                  [1, 0, 0],
                                  [1, 0, 1], ])
        num_path = len(corner_points) - 1
        num_each = int(num_frames/(num_path*3))
        rot_std = np.eye(3)
        trans_wights = np.array([0.005, 0.01, 0.01])
        poses = []
        for i in range(num_path):
            start_p = corner_points[i]
            end_p = corner_points[i+1]
            vec = normalize(end_p-start_p)
            dis = np.sqrt(np.sum((end_p-start_p)**2))/num_each
            theta = i*np.pi/2
            rota = R_y(theta)
            rot_ref = np.dot(rota, rot_std)
            for j in range(num_each):
                trans = start_p + vec*dis*j
                rot = rot_ref.copy()
                if random_sample:
                    ang_x = random.randint(-3, 3) * np.pi / 180
                    ang_y = random.randint(-3, 3) * np.pi / 180
                    rot_rdm = np.dot(R_y(ang_y), R_x(ang_x))
                    trans_rdm = trans_wights * (np.random.random_sample(3) * 2 - 1)
                    trans = trans + trans_rdm
                    rot = np.dot(rot_rdm, rot)
                pose = np.eye(4)
                pose[:3,:3] = rot
                pose[:3, 3] = trans
                poses.append(pose)
            # Corner transition
            num_corner = 2*num_each
            ang_j = np.pi/2/(num_corner+1)
            for j in range(num_corner):
                ang = ang_j * (j+1)
                rot = np.dot(R_y(ang), rot_ref)
                pose = np.eye(4)
                pose[:3,:3] = rot
                pose[:3, 3] = end_p
                poses.append(pose)
        poses = np.stack(poses)
    elif traj_type == 'circle0':
        rot_std = np.eye(3)
        trans_std = np.array([0.0, 0.0, 0.0])
        trans_wights = np.array([0.005, 0.01, 0.01])
        ang_i = -2 * np.pi / num_frames
        poses = []
        for i in range(num_frames):
            rot = np.dot(R_y(ang_i*i), rot_std)
            trans = trans_std.copy()
            if random_sample:
                ang_x = random.randint(-3, 3) * np.pi / 180
                ang_y = random.randint(-3, 3) * np.pi / 180
                rot_rdm = np.dot(R_y(ang_y), R_x(ang_x))
                trans_rdm = trans_wights * (np.random.random_sample(3) * 2 - 1)
                trans = trans + trans_rdm
                rot = np.dot(rot_rdm, rot)
            pose = np.eye(4)
            pose[:3,:3] = rot
            pose[:3, 3] = trans
            poses.append(pose)
        poses = np.stack(poses)
    elif traj_type == 'circle0_angle':
        rot_std = np.eye(3)
        trans_std = np.array([0.0, 0.0, 0.0])
        trans_wights = np.array([0.005, 0.01, 0.01])
        ang_i = -2 * np.pi * (circle_angle/360) / num_frames
        poses = []
        for i in range(num_frames):
            rot = np.dot(R_y(ang_i*i), rot_std)
            trans = trans_std.copy()
            if random_sample:
                ang_x = random.randint(-3, 3) * np.pi / 180
                ang_y = random.randint(-3, 3) * np.pi / 180
                rot_rdm = np.dot(R_y(ang_y), R_x(ang_x))
                trans_rdm = trans_wights * (np.random.random_sample(3) * 2 - 1)
                trans = trans + trans_rdm
                rot = np.dot(rot_rdm, rot)
            pose = np.eye(4)
            pose[:3,:3] = rot
            pose[:3, 3] = trans
            poses.append(pose)
        poses = np.stack(poses)
    elif traj_type == 'circle':
        radius = radius
        center = np.array([0., 0., 0.])
        rot_std = np.eye(3)
        if for_training:
            part_num = int(num_frames/2)
        else:
            part_num = num_frames
        ang_i = -2 * np.pi / part_num
        poses1, poses2 = [], []
        for i in range(part_num):
            rot = np.dot(R_y(ang_i*i), rot_std)
            trans = center + radius*normalize(rot[:3, 2])
            pose = np.eye(4)    
            pose[:3,:3] = rot
            pose[:3, 3] = trans
            poses1.append(pose)
            pose2 = pose.copy()
            pose2[:3, 3] = trans + 0.5*radius*normalize(rot[:3, 0])
            poses2.append(pose2)
        if for_training:
            poses1_inv = poses1[::-1]
            num = int(part_num/2)
            poses = []
            for i in range(num):
                poses.append(poses1[i])
                poses.append(poses1_inv[i])
            # poses += poses2
        else:
            poses = poses1
        poses = np.stack(poses)
    elif traj_type == 'circle_angle':
        radius = radius
        center = np.array([0., 0., 0.])
        rot_std = np.eye(3)
        trans_wights = np.array([0.005, 0.01, 0.01])
        ang_i = -2 * np.pi * (circle_angle/360) / num_frames
        poses = []
        for i in range(num_frames):
            rot = np.dot(R_y(ang_i*i), rot_std)
            trans = center + radius*normalize(rot[:3, 2])
            if random_sample:
                ang_x = random.randint(-3, 3) * np.pi / 180
                ang_y = random.randint(-3, 3) * np.pi / 180
                rot_rdm = np.dot(R_y(ang_y), R_x(ang_x))
                trans_rdm = trans_wights * (np.random.random_sample(3) * 2 - 1)
                trans = trans + trans_rdm
                rot = np.dot(rot_rdm, rot)
            pose = np.eye(4)
            pose[:3,:3] = rot
            pose[:3, 3] = trans
            poses.append(pose)
        poses = np.stack(poses)
    elif traj_type == 'line_move':
        if pose_ref is None:
            pose_ref = np.eye(4)
        delta_dis = line_length/num_frames
        up = normalize(pose_ref[:3, 1])
        h_v = normalize(pose_ref[:3, 0])
        z_v = normalize(pose_ref[:3, 2])
        center = pose_ref[:3, 3]
        direction = (z_v * np.cos(angle_pitch/180*np.pi) + up * np.sin(angle_pitch/180*np.pi)) * np.cos(angle_yaw/180*np.pi) + h_v * np.sin(angle_yaw/180*np.pi)
        poses = []
        for i in range(num_frames):
            posi = center + i*delta_dis*direction
            pose = pose_ref.copy()
            pose[:3, 3] = posi
            poses.append(pose)
        poses = np.stack(poses)


    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D
    # points = poses[:, :3, 3]
    # points2 = poses[:, :3, 3]+poses[:, :3, 2]*0.05
    # x,y,z = points[:, 0], points[:, 1], points[:, 2]
    # x2,y2,z2 = points2[:, 0], points2[:, 1], points2[:, 2]
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.scatter(x, y, z, c='b', label='trans')
    # ax.scatter(x[:1], y[:1], z[:1], c='r', label='start_point')
    # ax.scatter(x2, y2, z2, c='m', label='dir')
    # # ax.scatter([0.5], [0], [0.5], c='g', label='cen_point')
    # # ax.scatter(corner_points[:4, 0], corner_points[:4, 1], corner_points[:4, 2], c='g', label='corner_point')
    # ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    # ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    # ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    # plt.savefig('test_pose_.png')

    return poses

def get_double_circle_poses_from_center_pose(c2w, N_views, random_sample=False):
    """
    focal distance is the distance between c_cam and origin;
    """
    # standard pose
    focal = np.linalg.norm(c2w[:3, 3])
    if focal <= 0.01:
        focal = 0.2
    up = normalize(c2w[:3, 1])
    center = c2w[:3, 3]
    origin = center - focal*c2w[:3, 2]
    n1 = (N_views-1)//2
    n2 = N_views-1-n1

    render_poses = []
    render_poses.append(c2w)

    # Get start pose of first circle
    angle_h_start = 15
    alpha_list = list(np.linspace(0, 360, n1+1))[:-1]
    if random_sample:
        f_delta = 0.05 * focal * (np.random.random_sample(n1) * 2 - 1)
        f_delta = [f_delta[i] for i in range(n1)]
    else:
        f_delta = [0 for i in range(n1)]

    r = focal * np.sin(angle_h_start/180*np.pi)
    b = center - focal * (1-np.cos(angle_h_start/180*np.pi)) * normalize(c2w[:3, 2])
    for i, alpha in enumerate(alpha_list):
        angle = alpha/180*np.pi
        c = b + r * (normalize(c2w[:3, 0]) * np.cos(angle) - up * np.sin(angle))
        z = normalize(c - origin)
        if random_sample:
            c = c + f_delta[i] * z
        render_poses.append(viewmatrix(z, up, c))

    # Get start pose of second circle
    angle_h_start = 30
    alpha_list = list(np.linspace(0, 360, n2+1))[:-1]
    if random_sample:
        f_delta = 0.05 * focal * (np.random.random_sample(n2) * 2 - 1)
        f_delta = [f_delta[i] for i in range(n2)]
    else:
        f_delta = [0 for i in range(n2)]

    r = focal * np.sin(angle_h_start/180*np.pi)
    b = center - focal * (1-np.cos(angle_h_start/180*np.pi)) * normalize(c2w[:3, 2])
    for i, alpha in enumerate(alpha_list):
        angle = alpha/180*np.pi
        c = b + r * (normalize(c2w[:3, 0]) * np.cos(angle) - up * np.sin(angle))
        z = normalize(c - origin)
        if random_sample:
            c = c + f_delta[i] * z
        render_poses.append(viewmatrix(z, up, c))

    return np.stack(render_poses)
def get_rocking_traj_pose(c2w, angle_max=0.2, range_max=0.1, N_views=120, n_r=2):
    try:
        focal = range_max/np.sin(angle_max)
    except:
        focal = 10
    up = normalize(c2w[:3, 1])
    center = c2w[:3, 3]
    origin = center - focal*c2w[:3, 2]
    # Get pose
    num_per_r = int(N_views/n_r)
    angle_delta = 2*angle_max/num_per_r
    render_poses = []
    for i in range(num_per_r):
        angle = angle_max - angle_delta * i
        c = center - focal*(normalize(c2w[:3, 2])*(1-np.cos(angle))+normalize(c2w[:3, 0])*np.sin(angle))
        z = normalize(c - origin)
        render_poses.append(viewmatrix(z, up, c))

    poses2 = render_poses[::-1]
    render_poses = render_poses + poses2
    return np.stack(render_poses)


def get_circle_spiral_poses_from_pose(c2w, N_views=100, n_r=1, angle_h_start=0.2, trans_start=0.1, use_rand=False):
    """
    """
    # standard pose
    focal = 6. #np.linalg.norm(c2w[:3, 3])
    up = normalize(c2w[:3, 1])
    center = c2w[:3, 3]
    center0 = c2w[:3, 3] + 0.1*normalize(c2w[:3, 2])
    # center0 = center - focal * (1-np.cos(angle_h_start/180*np.pi)) * normalize(c2w[:3, 2])
    origin = center - focal*c2w[:3, 2]

    render_poses = []
    alpha_list = list(np.linspace(0, 360*n_r, N_views))
    if use_rand:
        posi_rand = 0.02*(np.random.random(3)*2-1)

    for i, alpha in enumerate(alpha_list):
        angle = alpha/180*np.pi
        c = center0 + trans_start * (normalize(c2w[:3, 0]) * np.cos(angle) - up * np.sin(angle))
        # z = normalize(c - origin)
        z = normalize(normalize(c2w[:3, 2]) + normalize(c - center) * np.sin(angle_h_start))
        if use_rand:
            c = c + posi_rand
        render_poses.append(viewmatrix(z, up, c))

    return np.stack(render_poses)

def get_circle_poses_from_pose(c2w, f_delta=0, N_views=120, n_r=2, angle_h_start=15, use_rand=False):
    """
    focal distance is the distance between c_cam and origin;
    Here, we let 'focal' value change in the range [focal-f_delta, focal+f_delta],
    when f_delta=0, the focal will be fixed.
    """
    # standard pose
    focal = 0.1 #np.linalg.norm(c2w[:3, 3])
    up = normalize(c2w[:3, 1])
    center = c2w[:3, 3]
    origin = center - focal*c2w[:3, 2]

    # Get start pose
    angle_h_start = angle_h_start
    angle = angle_h_start/180*np.pi
    c_s = center - focal*(normalize(c2w[:3, 2])*(1-np.cos(angle))+normalize(c2w[:3, 0])*np.sin(angle))
    z = normalize(c_s - origin)
    pose_start = viewmatrix(z, up, c_s)

    render_poses = []
    focals = [focal for i in range(N_views)]

    # n_straight = max(N_views//n_r//5, 5)
    # n_cir = N_views-n_straight

    alpha_list = list(np.linspace(0, 360*n_r, N_views))
    if use_rand:
        posi_rand = 0.02*(np.random.random(3)*2-1)

    r = focal * np.sin(angle_h_start/180*np.pi)
    for i, alpha in enumerate(alpha_list):
        angle = alpha/180*np.pi
        f = focals[i]
        b = center - f * (1-np.cos(angle_h_start/180*np.pi)) * normalize(c2w[:3, 2])
        c = b + r * (normalize(c2w[:3, 0]) * np.cos(angle) - up * np.sin(angle))
        z = normalize(c - origin)
        if use_rand:
            c = c + posi_rand
        render_poses.append(viewmatrix(z, up, c))

    return np.stack(render_poses)

def get_local_fixed_poses(c2w_basis, angle=0.3, range_center=0.2, range_yaw=0.6, range_pitch=0.3, use_rand=False, angle_rand=0.05, posi_rand=0.05):
    """
    our coordinate system is defined as: view (+z), up (+y), right (+x)
    """
    rotvecs = {'R': np.array([0,range_yaw,0]), 'L': np.array([0,-range_yaw,0]), \
                'U': np.array([range_pitch,0,0]), 'D': np.array([-range_pitch,0,0]), \
                'UR': np.array([ range_pitch/2,range_yaw/2,0]), 'UL': np.array([ range_pitch/2,-range_yaw/2,0]), \
                'DR': np.array([-range_pitch/2,range_yaw/2,0]), 'DL': np.array([-range_pitch/2,-range_yaw/2,0])} 
    posivecs = {'R': np.array([range_center,0,0]), 'L': np.array([-range_center,0,0]), \
                'U': np.array([0,-range_center/2,0]), 'D': np.array([0,range_center/2,0]), \
                'UR': np.array([range_center,-range_center/2,0]), 'UL': np.array([-range_center,-range_center/2,0]), \
                'DR': np.array([range_center,range_center/2,0]), 'DL': np.array([-range_center,range_center/2,0])} 
    poses = []
    # if use_rand:
    #     euler_rand = 0.5*angle_rand*(np.random.random(3)*2-1)
    #     posi_rand = 0.5*posi_rand*(np.random.random(3)*2-1)
    #     c2w_rand = np.eye(4, dtype=np.float32)
    #     c2w_rand[:3,:3] = eulerangle2rotation(euler_rand)
    #     c2w_rand[:3, 3] = posi_rand
    #     c2w_basis = np.dot(c2w_rand, c2w_basis)
    poses.append(c2w_basis)

    directions = ['R', 'L', 'U', 'D', 'UR', 'UL', 'DR', 'DL']
    for dir in directions:
        rot = rotvecs[dir] * angle / np.linalg.norm(rotvecs[dir])
        posi = posivecs[dir]
        if use_rand:
            euler_rand = 0.5*angle_rand*(np.random.random(3)*2-1)/180*np.pi
            posi_rand = 0.5*posi_rand*(np.random.random(3)*2-1)
            rot += euler_rand
            posi = posi + posi_rand
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = np.dot(eulerangle2rotation(rot), c2w_basis[:3, :3])
        c2w[:3, 3] = posi + c2w_basis[:3, 3]
        poses.append(c2w)
    return np.stack(poses).astype(np.float32)

def get_local_fixed_poses2(c2w_basis, angle=0.3, range_center=0.2, range_yaw=0.6, range_pitch=0.3, use_rand=False, angle_rand=0.05, posi_rand=0.05):
    """
    our coordinate system is defined as: view (+z), up (+y), right (+x)
    """
    rotvecs = {'R': np.array([0,range_yaw,0]), 'L': np.array([0,-range_yaw,0]), \
                'U': np.array([range_pitch,0,0]), 'D': np.array([-range_pitch,0,0]), \
                'UR': np.array([ range_pitch/2,range_yaw/2,0]), 'UL': np.array([ range_pitch/2,-range_yaw/2,0]), \
                'DR': np.array([-range_pitch/2,range_yaw/2,0]), 'DL': np.array([-range_pitch/2,-range_yaw/2,0])} 
    posivecs = {'R': np.array([range_center,0,0]), 'L': np.array([-range_center,0,0]), \
                'U': np.array([0,-range_center,0]), 'D': np.array([0,range_center,0]), \
                'UR': np.array([range_center,-range_center,0]), 'UL': np.array([-range_center,-range_center,0]), \
                'DR': np.array([range_center,range_center,0]), 'DL': np.array([-range_center,range_center,0])} 
    # posivecs = {'R': np.array([range_center,0,0]), 'L': np.array([-range_center,0,0]), \
    #             'U': np.array([0,-range_center/2,0]), 'D': np.array([0,range_center/2,0]), \
    #             'UR': np.array([range_center,-range_center/2,0]), 'UL': np.array([-range_center,-range_center/2,0]), \
    #             'DR': np.array([range_center,range_center/2,0]), 'DL': np.array([-range_center,range_center/2,0])} 
    poses = []
    poses.append(c2w_basis)

    # zoom-in & zoom-out
    # posi = np.array([0, 0, range_center])
    # c2w = c2w_basis.copy()
    # c2w[:3, 3] = posi
    # poses.append(c2w)
    # c2w = c2w_basis.copy()
    # c2w[:3, 3] = 0-2*posi
    # poses.append(c2w)

    directions = ['R', 'UR', 'U', 'UL', 'L', 'DL', 'D', 'DR']
    for dir in directions:
        rot = rotvecs[dir] * angle / np.linalg.norm(rotvecs[dir])
        posi = posivecs[dir]
        if use_rand:
            euler_rand = 0.5*angle_rand*(np.random.random(3)*2-1)/180*np.pi
            posi_rand = 0.5*posi_rand*(np.random.random(3)*2-1)
            rot += euler_rand
            posi = posi + posi_rand
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = np.dot(eulerangle2rotation(rot), c2w_basis[:3, :3])
        c2w[:3, 3] = posi + c2w_basis[:3, 3]
        poses.append(c2w)

    return np.stack(poses).astype(np.float32)

def get_local_poses3(c2w_basis, range_center=0.2):
    """
    our coordinate system is defined as: view (+z), up (+y), right (+x)
    """
    posivecs = {'R': np.array([range_center,0,0]), 'L': np.array([-range_center,0,0]), \
                'U': np.array([0,-range_center,0]), 'D': np.array([0,range_center,0]), \
                'UR': np.array([range_center,-range_center,0]), 'UL': np.array([-range_center,-range_center,0]), \
                'DR': np.array([range_center,range_center,0]), 'DL': np.array([-range_center,range_center,0])} 

    # standard pose
    focal = 6. #np.linalg.norm(c2w_basis[:3, 3])
    up = normalize(c2w_basis[:3, 1])
    center = c2w_basis[:3, 3]
    origin = center - focal*c2w_basis[:3, 2]

    poses = []
    poses.append(c2w_basis)

    directions = ['R', 'UR', 'U', 'UL', 'L', 'DL', 'D', 'DR']
    for dir in directions:
        c0 = posivecs[dir] + c2w_basis[:3, 3]
        z = normalize(c0 - origin)
        c = focal*z + origin
        poses.append(viewmatrix(z, up, c))

    return np.stack(poses).astype(np.float32)

def get_r2l_pose(c2w_basis, range_center=0.2, num_frame=None):
    shift_front = 0.0
    delta = 0.00
    posivecs = {'R': np.array([range_center,0,0]), 'L': np.array([-range_center,0,0]), \
                'RR': np.array([2*range_center,0,0]), 'LL': np.array([-range_center*2,0,0])}
    poses = []
    if num_frame is None:
        poses.append(c2w_basis)
        directions = ['R', 'RR', 'L', 'LL']
        for dir in directions:
            posi = posivecs[dir]
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, :3] = c2w_basis[:3, :3]
            c2w[:3, 3] = posi + c2w_basis[:3, 3]
            poses.append(c2w)
    else:
        pp = list(np.linspace(2*range_center-delta, -range_center*2+delta, int(num_frame/2)))+list(np.linspace(-2*range_center+delta, range_center*2-delta, int(num_frame/2)))
        z_shifts = list(np.linspace(shift_front, 0, int(num_frame/4)))+list(np.linspace(0, shift_front, int(num_frame/4)))+list(np.linspace(shift_front, 0, int(num_frame/4)))+list(np.linspace(0, shift_front, int(num_frame/4)))
        for ii in range(len(pp)):
            posi = np.array([pp[ii],0,shift_front])
            # posi = np.array([pp[ii],0,z_shifts[ii]])
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, :3] = c2w_basis[:3, :3]
            c2w[:3, 3] = posi + c2w_basis[:3, 3]
            poses.append(c2w)
    return np.stack(poses).astype(np.float32)


def get_sprt_poses(c2w_basis, num_poses=8, range_center=0.2):
    """
    our coordinate system is defined as: view (+z), up (-y), right (+x)
    """
    poses = []
    poses.append(c2w_basis)
    if num_poses == 0:
        return c2w_basis[np.newaxis].astype(np.float32)
    angle_each = 2 * np.pi / num_poses
    rot = np.array([0,0,0])

    for ii in range(num_poses):
        angle = ii * angle_each
        trans_x = range_center * np.cos(angle)
        trans_y = -range_center * np.sin(angle)
        posi = np.array([trans_x, trans_y, 0])
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = np.dot(eulerangle2rotation(rot), c2w_basis[:3, :3])
        c2w[:3, 3] = posi + c2w_basis[:3, 3]
        poses.append(c2w)
    return np.stack(poses).astype(np.float32)
    
def get_pretraining_poses(c2w_basis, range_center=0.2, range_yaw=20, range_pitch=15, use_rand=False, angle_rand=5, posi_rand=0.05):
    """
    our coordinate system is defined as: view (+z), up (+y), right (+x)
    """
    rotvecs = {'R': np.array([0,range_yaw,0]), 'L': np.array([0,-range_yaw,0]), \
                'U': np.array([range_pitch,0,0]), 'D': np.array([-range_pitch,0,0]), \
                'UR': np.array([ range_pitch,range_yaw,0]), 'UL': np.array([ range_pitch,-range_yaw,0]), \
                'DR': np.array([-range_pitch,range_yaw,0]), 'DL': np.array([-range_pitch,-range_yaw,0])} 
    posivecs = {'R': np.array([range_center,0,0]), 'L': np.array([-range_center,0,0]), \
                'U': np.array([0,-range_center,0]), 'D': np.array([0,range_center,0]), \
                'UR': np.array([range_center,-range_center,0]), 'UL': np.array([-range_center,-range_center,0]), \
                'DR': np.array([range_center,range_center,0]), 'DL': np.array([-range_center,range_center,0])} 
    poses = []
    if use_rand:
        euler_rand = 0.5*angle_rand*(np.random.random(3)*2-1)/180*np.pi
        posi_rand = 0.5*posi_rand*(np.random.random(3)*2-1)
        c2w_rand = np.eye(4, dtype=np.float32)
        c2w_rand[:3,:3] = eulerangle2rotation(euler_rand)
        c2w_rand[:3, 3] = posi_rand
        c2w_basis = np.dot(c2w_rand, c2w_basis)
    poses.append(c2w_basis)

    directions = ['R', 'L', 'U', 'D', 'UR', 'UL', 'DR', 'DL']
    for dir in directions:
        rot = rotvecs[dir]/180*np.pi
        posi = posivecs[dir]
        if use_rand:
            euler_rand = 0.5*angle_rand*(np.random.random(3)*2-1)/180*np.pi
            posi_rand = 0.5*posi_rand*(np.random.random(3)*2-1)
            rot += euler_rand
            posi += posi_rand
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = np.dot(eulerangle2rotation(rot), c2w_basis[:3, :3])
        c2w[:3, 3] = posi + c2w_basis[:3, 3]
        poses.append(c2w)
    return np.stack(poses).astype(np.float32)
    

def get_circle_poses_from_basis_view(c2w, N_views=120, angle=15, n_r=2):
    """
    focal distance is the distance between c_cam and origin;
    Here, we let 'focal' value change in the range [focal-f_delta, focal+f_delta],
    when f_delta=0, the focal will be fixed.
    """
    # standard pose
    focal = 0.1 #np.linalg.norm(c2w[:3, 3])
    up = normalize(c2w[:3, 1])
    center = c2w[:3, 3]
    origin = center - focal*c2w[:3, 2]

    # Get start pose
    angle_h_start = 15
    angle = -angle_h_start/180*np.pi
    c_s = center - focal*(normalize(c2w[:3, 2])*(1-np.cos(angle))+normalize(c2w[:3, 0])*np.sin(angle))
    z = normalize(c_s - origin)
    pose_start = viewmatrix(z, up, c_s)

    render_poses = []
    focals = [focal for i in range(N_views)]

    alpha_list = list(np.linspace(0, 360*n_r, N_views))

    r = focal * np.sin(angle_h_start/180*np.pi)
    for i, alpha in enumerate(alpha_list):
        angle = alpha/180*np.pi
        f = focals[i]
        b = center - f * (1-np.cos(angle_h_start/180*np.pi)) * normalize(c2w[:3, 2])
        c = b + r * (normalize(c2w[:3, 0]) * np.cos(angle) - up * np.sin(angle))
        z = normalize(c - origin)
        render_poses.append(viewmatrix(z, up, c))

    return np.stack(render_poses)
