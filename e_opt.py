import configargparse


def config_parser(cmd=None):
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, default='configs/text2nerf_scenes.txt', #e_scenes_indoor
                        help='config file path')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./log',
                        help='where to store ckpts and logs')
    
    ## training options
    # training procedure
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--regen_pose', action='store_true')
    parser.add_argument("--prompt", type=str, default='a cozy living room')
    parser.add_argument("--dibr_fill_strategy", type=str, default='filling', choices=['filling', 'average'])
    parser.add_argument("--inpaint_method", type=str, default='sdm')
    parser.add_argument("--type_depth_loss", type=str, default='mse')
    parser.add_argument("--angle", type=float, default=0.2)
    parser.add_argument("--trans_range", type=float, default=0.2)
    parser.add_argument("--push_depth", type=float, default=2.0)
    parser.add_argument("--num_sprt_poses", type=int, default=8)
    parser.add_argument("--dist_sprt_poses", type=float, default=0.2)
    parser.add_argument("--n_stage1", type=int, default=50)
    parser.add_argument("--n_stage2", type=int, default=50)
    parser.add_argument("--n_stage3", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument('--use_filter_filling_holes', action='store_true')
    parser.add_argument('--use_rendered_img_to_warp', action='store_true')
    parser.add_argument('--use_rendered_img_to_inpaint', action='store_true')
    parser.add_argument('--use_bias_elimi', action='store_true')
    parser.add_argument("--progress_refresh_rate", type=int, default=10,
                        help='how many iterations to show psnrs or iters')
    parser.add_argument("--ckpt", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--render_only", type=int, default=0)
    parser.add_argument("--render_test", type=int, default=0)
    parser.add_argument("--render_train", type=int, default=0)
    parser.add_argument("--render_path", type=int, default=0)
    parser.add_argument("--export_mesh", type=int, default=0)
    
    # loader options
    parser.add_argument("--batch_size", type=int, default=4096*2)
    parser.add_argument("--datadir", type=str, default='./data/llff/fern',
                        help='input data directory')
    parser.add_argument("--inpainted_dir", type=str, default='',
                        help='input data directory')
    parser.add_argument('--dataset_name', type=str, default='scene_gen')
    parser.add_argument('--pose_traj', type=str, default='local_fixed',
                        help="local:'local_fixed'/'local_double';  global: 'circle0'/'circle'/'rectangle'/'circle0_XXX'/'circle_XXX'/'line_pitch_yaw_distance'")
    parser.add_argument('--frame_select_type', type=str, default='auto_cos')
    parser.add_argument("--num_training", type=int, default=24)
    parser.add_argument('--use_support_set', action='store_true')

    # learning rate
    parser.add_argument("--lr_init", type=float, default=0.02,
                        help='learning rate')    
    parser.add_argument("--lr_basis", type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument("--lr_decay_iters", type=int, default=-1,
                        help = 'number of iterations the lr will decay to the target ratio; -1 will set it to n_iters')
    parser.add_argument("--lr_decay_target_ratio", type=float, default=0.1,
                        help='the target decay ratio; after decay_iters inital lr decays to lr*ratio')
    parser.add_argument("--lr_upsample_reset", type=int, default=1,
                        help='reset lr to inital after upsampling')

    # loss
    parser.add_argument("--L1_weight_inital", type=float, default=0.0,
                        help='loss weight')
    parser.add_argument("--L1_weight_rest", type=float, default=0,
                        help='loss weight')
    parser.add_argument("--Ortho_weight", type=float, default=0.0,
                        help='loss weight')
    parser.add_argument("--TV_weight_density", type=float, default=0.0,
                        help='loss weight')
    parser.add_argument("--TV_weight_app", type=float, default=0.0,
                        help='loss weight')
    
    ## model
    # volume options
    parser.add_argument("--n_lamb_sigma", type=int, action="append")
    parser.add_argument("--n_lamb_sh", type=int, action="append")
    parser.add_argument("--data_dim_color", type=int, default=27)

    parser.add_argument("--rm_weight_mask_thre", type=float, default=0.0001,
                        help='mask points in ray marching')
    parser.add_argument("--alpha_mask_thre", type=float, default=0.0001,
                        help='threshold for creating alpha mask volume')
    parser.add_argument("--distance_scale", type=float, default=25,
                        help='scaling sampling distance for computation')
    parser.add_argument("--density_shift", type=float, default=-10,
                        help='shift density in softplus; making density = 0  when feature == 0')
                        
    # network decoder
    parser.add_argument('--model_name', type=str, default='TensorVMSplit',
                        choices=['TensorVMSplit', 'TensorCP'])
    parser.add_argument("--shadingMode", type=str, default="MLP_PE_noview",
                        help='which shading mode to use')
    parser.add_argument("--pos_pe", type=int, default=6,
                        help='number of pe for pos')
    parser.add_argument("--view_pe", type=int, default=6,
                        help='number of pe for view')
    parser.add_argument("--fea_pe", type=int, default=6,
                        help='number of pe for features')
    parser.add_argument("--featureC", type=int, default=128,
                        help='hidden feature channel in MLP')
    
    # rendering options
    parser.add_argument('--lindisp', default=False, action="store_true",
                        help='use disparity depth sampling')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--accumulate_decay", type=float, default=0.998)
    parser.add_argument("--fea2denseAct", type=str, default='softplus')
    parser.add_argument('--ndc_ray', type=int, default=0)
    parser.add_argument('--nSamples', type=int, default=1e6,
                        help='sample point each ray, pass 1e6 if automatic adjust')
    parser.add_argument('--step_ratio',type=float,default=0.5)

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')

    parser.add_argument('--N_voxel_init',
                        type=int,
                        default=100**3)
    parser.add_argument('--N_voxel_final',
                        type=int,
                        default=300**3)
    parser.add_argument("--upsamp_list", type=int, action="append")
    parser.add_argument("--update_AlphaMask_list", type=int, action="append")

    parser.add_argument('--idx_view',
                        type=int,
                        default=0)
    # logging/saving options
    parser.add_argument("--N_vis", type=int, default=5,
                        help='N images to vis')
    parser.add_argument("--vis_every", type=int, default=10000,
                        help='frequency of visualize the image')

    ## TODO: remove
    parser.add_argument("--add_timestamp", type=int, default=0,
                        help='add timestamp to dir')
    parser.add_argument('--with_depth', action='store_true')
    parser.add_argument('--downsample_train', type=float, default=1.0)
    parser.add_argument('--downsample_test', type=float, default=1.0)
    parser.add_argument("--n_iters", type=int, default=30000)

    if cmd is not None:
        return parser.parse_args(cmd)
    else:
        return parser.parse_args()