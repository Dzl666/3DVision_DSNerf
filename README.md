# Depth-Supervison NeRF for project of ETHZ 3DVison Lecture

## Euler commands
* Load module: `module load gcc/8.2.0 cuda/11.6.2 python/3.8.5 cudnn/8.0.5 cmake/3.19.8 eth_proxy`

* Activate virtual environment: `source ../env-3dvision/bin/activate`

* Debug `srun --time=0:30:00 --gpus=1 --gres=gpumem:16g -n 3 --mem-per-cpu=8g --pty bash`
* Check the status of allocation: `squeue`

* Submit Job: `sbatch --time=4:00:00 --gpus=1 --gres=gpumem:16g -n 2 --mem-per-cpu=8g --output=./logs/raw_output --open-mode=append --wrap="[...cmd...]"`
* Check details of the job: `myjobs -j job_id`
* Check details of the job: `scancel job_id`

* Change access: `chmod -R u+rwx,g+rwx,o+rx ./`

> Training `sbatch --time=4:00:00 --gpus=1 --gres=gpumem:28g -n 3 --mem-per-cpu=8g --output=./logs/raw_output_room --open-mode=append --wrap="python run_nerf.py --config configs/ex_room.txt > ./logs/training_log_room"`

> Only Render `sbatch --time=1:00:00 --gpus=1 --gres=gpumem:30g -n 3 --mem-per-cpu=8g --output=./logs/raw_output_bookshelf --open-mode=append --wrap="python run_nerf.py --config configs/ex_bookshelf.txt --render_only > ./logs/rendering_log"`

> Temp `sbatch --time=8:00:00 --gpus=1 --gres=gpumem:36g -n 5 --mem-per-cpu=8g --output=./logs/raw_output_bookshelf_all_factor_1_cont --open-mode=append --wrap="python run_nerf.py --config configs/ex_bookshelf.txt > ./logs/training_log_bookshelf_all_factor_1_cont"`

## Current Work

1. Images set --> COLMAP --> camera poses and 3d Points set
2. camera poses and 3d Points st --> depth map --> train NeRF
3. Dataset from No.90 to No.2325, gap - 15, total 150 pics (now training with a few images to speed up)
4. Generate Videos using the training camera poses, with Sphere Linear Interpolation
5. Show Video: downsample factor = 2, dataset total 80 pics

## Plan

1. Leverage the depth map and the origin camera poses (Problems: weird depth map data)
2. Train the full scene, try to improve the quality and speed up more

