# [Text2NeRF](https://eckertzhang.github.io/Text2NeRF.github.io/)
Official implementation of '[Text2NeRF: Text-Driven 3D Scene Generation with Neural Radiance Fields](https://arxiv.org/abs/2305.11588)'. 
Note: This code is forked from [TensoRF](https://apchenstu.github.io/TensoRF/).

<img src='https://eckertzhang.github.io/Text2NeRF.github.io/static/images/teaser.png'>

## Installation

Install environment:
```
conda env create -f environment.yml
conda activate text2nerf
pip install -r requirements.txt
```

## Download Pre-trained Weights
1. Download pre-trained '[CLIPModel(clip-vit-base-patch32)](https://huggingface.co/openai/clip-vit-base-patch32)' into the folder 'weights'.
2. Download pre-trained '[SDM](https://huggingface.co/stabilityai/stable-diffusion-2-1-base)' into the folder 'weights'.
3. Download pre-trained '[SDM-Inpaint(stable-diffusion-2-inpainting)](https://huggingface.co/stabilityai/stable-diffusion-2-inpainting)' into the folder 'weights'.
4. Download other required weights for third parties from [Google_Drive](https://drive.google.com/file/d/1kt1VUgGYjsSEIMRKZa6Qe9qbg74Y0px5/view?usp=sharing), and put it into the folder 'weights'

## Optimization
The training script is in `text2nerf_main.py`:

#### Local Scene Generation
```
CUDA_VISIBLE_DEVICES=0 python text2nerf_main.py --config 'configs/text2nerf_scenes.txt' --expname 'text000' --prompt 'a beautiful garden' --datadir 'data_example/text000' --pose_traj 'local_fixed' --regen_pose
```
#### 360-Degree Scene Generation
```
CUDA_VISIBLE_DEVICES=0 python text2nerf_main.py --config 'configs/text2nerf_scenes.txt' --expname 'text000_360' --prompt 'a beautiful garden' --datadir 'data_example/text000' --pose_traj 'circle' --regen_pose
```


## Rendering

```
CUDA_VISIBLE_DEVICES=0 python text2nerf_main.py --config 'configs/text2nerf_scenes.txt' --expname 'text000' --prompt 'a beautiful garden' --datadir 'data_example/text000' --pose_traj 'local_fixed' --regen_pose --render_only 1 --render_test 1 --ckpt '[path of ckpt]'
```

    

## Citation
If you find our code or paper helps, please consider citing:
```
@article{zhang2023text2nerf,
  title={Text2NeRF: Text-Driven 3D Scene Generation with Neural Radiance Fields},
  author={Zhang, Jingbo and Li, Xiaoyu and Wan, Ziyu and Wang, Can and Liao, Jing},
  journal={arXiv preprint arXiv:2305.11588},
  year={2023}
}
```
