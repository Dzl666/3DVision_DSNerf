# Depth-Supervised NeRF for project of ETHZ 3DVison Lecture
## file structure: 
./logs/exp/ --experiment outputs
- bookshelf/20230515_2220 --bookshelf scene
- poster/20230515_1450 --poster scene 
- 520v_room/20230516 --room scene 
- poster_less/20230608_1545 --poster scene with less training images 

./data/experiments --experiment data
- bookshelf --bookshelf scene
- poster --poster scene 
- 520view_room --room scene 
- poster_less --poster scene with less training images 

## pipeline:
### 1, load dependencies
* If you are not running on Euler, please consult the official [tutorial](https://github.com/dunbar12138/DSNeRF#dependencies) for environment building.
* The list of dependances could be found in ./requirements.txt

#### if running on euler 
* Load module: `module load gcc/8.2.0 cuda/11.6.2 python/3.8.5 cudnn/8.0.5 cmake/3.19.8 eth_proxy`
* Activate virtual environment: `source ../env-3dvision/bin/activate`

### 2, run colmap
Note: This part should be done in your local machine 
* download COLMAP and run imgs2poses.py according to the [tutorial](https://github.com/dunbar12138/DSNeRF#generate-camera-poses-and-sparse-depth-information-using-colmap-optional) in the DS-Nerf repo
* in colmap_wrapper.py, you could tailer SiftExtraction.peak_threshold and SiftExtraction.edge_threshold according to your dataset

### 3, training and testing
* overall procedures please consolt the [tutorial](https://github.com/dunbar12138/DSNeRF#how-to-run) for DS-Nerf.
* when constructing config.txt, we have added the following new parameters:
1. seg_video --how many segments to divide the rendered video (tailor according to your GPU capacity)
2. render_itp_nodes --When render_train=True, we allow interpolations between each rendering poses to form smooth but longer videos. render_itp_node refers to the number of Slerp interpolation nodes between each rentering pose when generating the video poses.

#### If running on euler
The following command examples would be useful:
* Debug `srun --time=1:30:00 --gpus=1 --gres=gpumem:16g -n 2 --mem-per-cpu=8g --pty bash`
* Check the status of allocation: `squeue`

* Submit Job: `sbatch --time=4:00:00 --gpus=1 --gres=gpumem:16g -n 2 --mem-per-cpu=8g --output=./logs/raw_output --open-mode=append --wrap="[...cmd...]"`
* Check details of the job: `myjobs -j job_id`
* Check details of the job: `scancel job_id`

* Change access permission for others: `chmod -R u+rwx,g+rwx,o+rx ./`

> Training `sbatch --time=6:00:00 --gpus=1 --gres=gpumem:25g -n 3 --mem-per-cpu=8g --output=./logs/raw_output_poster_less_cont --open-mode=append --wrap="python run_nerf.py --config configs/exp_poster_less.txt > ./logs/training_log_poster_less_cont"`

> Only Render `sbatch --time=4:00:00 --gpus=1 --gres=gpumem:35g -n 5 --mem-per-cpu=8g --output=./logs/raw_output_bookshelf --open-mode=append --wrap="python run_nerf.py --config configs/exp_bookshelf.txt --render_only > ./logs/rendering_log_bookshelf"`


## 4, other pipelines 
Here are documents and comments related to our pipeline for converting the point cloud generated from rgbd alingment to the depth information for training DS-Nerf and using hololens coordinates instead of COLMAP estimations. **Important: these pipelines are currently unable to produce desired results due to the mismatch between the coordinates of our generated depth map/camera poses and the coordiantes expected for DS-Nerf**
* ./open3d_llff/color_map_optimization.py --aligns a set of depth map frames together with a set of RGB image frames. Please modify the input parameters at line 169, where instructiions could also be found. 
Output: a set of aligned depth maps and a list containing the paths to the RGB frames corresponding to each generated depth frames. 
* ./open3d_llff/llff_conversion.py --converts the Hololens poses, intrinsics, and the color_map_optimization.py outputs into the format suitable for the "colmap_llff" input of DS-Nerf training (see line 437 in ren_nerf.py). More instructions coule be found inside the code. 
* ./open3d_llff/poses_from_colmap.py --converts the poses_bounds.npy output of colmap into the training/testing posrs abd bounds of 'colmap_llff' form. 

## Citation and Acknowledgmentgs:
Our pipeline is build upon the [DS-Nerf](https://github.com/dunbar12138/DSNeRF) pipeline. If you use our code, please make sure to cite the original DS-Nerf paper:
```
@InProceedings{kangle2021dsnerf,
    author    = {Deng, Kangle and Liu, Andrew and Zhu, Jun-Yan and Ramanan, Deva},
    title     = {Depth-supervised {NeRF}: Fewer Views and Faster Training for Free},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022}
}
```
