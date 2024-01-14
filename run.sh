# Optimization
CUDA_VISIBLE_DEVICES=0 python text2nerf_main.py --config 'configs/text2nerf_scenes.txt' --expname 'text000' --prompt 'a beautiful garden' --datadir 'data_example/text000' --pose_traj 'local_fixed' --regen_pose

# Rendering
CUDA_VISIBLE_DEVICES=0 python text2nerf_main.py --config 'configs/text2nerf_scenes.txt' --expname 'text000' --prompt 'a beautiful garden' --datadir 'data_example/text000' --pose_traj 'local_fixed' --regen_pose --render_only 1 --render_test 1 --ckpt '[path of ckpt]'
