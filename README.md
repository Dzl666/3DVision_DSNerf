# Depth-Supervison NeRF for project of ETHZ 3DVison Lecture

## Euler commands
* Load module: `module load gcc/8.2.0 cuda/11.6.2 python/3.8.5 cudnn/8.0.5 cmake/3.19.8 eth_proxy`

* Activate virtual environment: `source ../env-3dvision/bin/activate`

* Submit Job: `sbatch --time=4:00:00 --gpus=1 --gres=gpumem:16g --cpus-per-task=1 --mem-per-cpu=8g --output=./logs/raw_output --open-mode=append --wrap="[...cmd...]"`

* Check details of the job: `myjobs -j job_id`

* Check details of the job: `scancel job_id`

* Change access: `chmod -R u+rwx,g+rwx,o+rx ./`

> Training `sbatch --time=8:00:00 --gpus=1 --gres=gpumem:30g --cpus-per-task=1 --mem-per-cpu=20g --output=./logs/raw_output --open-mode=append --wrap="python run_nerf.py --config configs/150vAnna_20_2.txt > ./logs/training_log"`

> Only Render `sbatch --time=1:00:00 --gpus=1 --gres=gpumem:16g --cpus-per-task=1 --mem-per-cpu=8g --output=./logs/raw_output --open-mode=append --wrap="python run_nerf.py --config configs/150v_90_15_2325.txt --render_only > ./logs/rendering_log"`



## Current Work

1. Images set --> COLMAP --> camera poses and 3d Points set
2. camera poses and 3d Points st --> depth map --> train NeRF
3. Dataset from No.90 to No.2325, gap - 15, total 150 pics (now training with a few images to speed up)
4. Generate Videos using the training camera poses, with Sphere Linear Interpolation
5. Show Video: downsample factor = 2, dataset total 80 pics

## Plan

1. Leverage the depth map and the origin camera poses (Problems: weird depth map data)
2. Train the full scene, try to improve the quality and speed up more

