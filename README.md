# Depth-Supervised NeRF for project of ETHZ 3DVison Lecture

[Video](https://youtube.com/playlist?list=PLUffCQyBEYtbOQg4-66ZrcuNmsX0OXVKv)

## File structure: 

### Code

- run_nerf.py

### Datasets and Logs

./logs/exp/ --experiment outputs
- poster_less/20230608_1545 --poster scene with less training images 

./data/experiments --experiment data
- poster_less --poster scene with less training images

This dataset is captured by Hololens2 and consists of two video recordings of two different room scenes. Each capture contains thousands of RGB video frames in 1280×720, monocular depth frames in a lower capturing frequency, the intrinsic parameters of the camera, and the corresponding camera poses and timestamp for each RGB frame. For the first capture ([AnnaTrain](https://drive.google.com/file/d/1ejI0oGDvouf8kSXmtE2YtDnUD5xQ9CJ0/view)/[GowthamTrain](https://drive.google.com/file/d/1SDoMu82SKCXeIN0Jx5hPdFrSIh5NdLd5/view)), the HoloLens has a relatively slow movement, which results in a dataset containing less motion blur. While the second capture (named [AnnaTest](https://drive.google.com/file/d/1GM86hnksWmncO_VzHofgo8cX0_KKEzvO/view)/[GowthamTest](https://drive.google.com/file/d/1ch8T6YyFJjmdYxV6ZIc7_MvTgNo4QHTE/view)) contains more motion blur. Here use the AnnaTrain as an example.
```
AnnaTrain
     ├── Depth (not used for this method)
     ├── Head (not used for this method)
     ├── SceneUnderstanding (not used for this method)
     ├── Video(rename into: images)
     └── poses_bounds.npy
```

See [google drive](https://drive.google.com/drive/folders/1QVC7wxyLZeEcIck142Z531eHLeQANbt5?usp=drive_link) for more output images, videos, logs, and saved models of our experiments.

## Coordinate System
![Alt text](<coordinate system_paper.png>)

* If use the COLMAP application to estimate the camera pose, the coordinate system of the output is in the Open3D/COLMAP system, if runing the SFM process directly by code, the coordinate system is further transfered to the LLFF system
* The NeRF is training on the NeRF coordinate system
* The direction of the HoloLens had been verified by Open3D rendering

## Pipeline:
### 1. load dependencies
* If you are not running on Euler, please consult the official [tutorial](https://github.com/dunbar12138/DSNeRF#dependencies) for environment building.
* The list of dependencies could be found at [./requirements.txt](https://github.com/Dzl666/3DVision_DSNerf/blob/master/requirements.txt)

#### if running on euler 
* Load module: `module load gcc/8.2.0 cuda/11.6.2 python/3.8.5 cudnn/8.0.5 cmake/3.19.8 eth_proxy`
* Activate virtual environment: `source ../env-3dvision/bin/activate`

### 2. run colmap
Note: This part should be done in your local machine 
* download COLMAP and run imgs2poses.py according to the [tutorial](https://github.com/dunbar12138/DSNeRF#generate-camera-poses-and-sparse-depth-information-using-colmap-optional) in the DS-Nerf repo
* in colmap_wrapper.py, you could tailer SiftExtraction.peak_threshold and SiftExtraction.edge_threshold according to your dataset

### 3. training and testing
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
* Cancel job: `scancel job_id`

* Change access permission for others: `chmod -R u+rwx,g+rwx,o+rx ./`

### Example Run

> Training and Rendering - `sbatch --time=4:00:00 --gpus=1 --gres=gpumem:24g -n 3 --mem-per-cpu=8g --output=./logs/raw_output_poster --open-mode=append --wrap="python run_nerf.py --config configs/exp_poster_less.txt > ./logs/training_log_poster"`


## 4. other pipelines 
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
