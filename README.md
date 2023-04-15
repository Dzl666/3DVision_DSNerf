# Depth-Supervison NeRF for project of ETHZ 3DVison Lecture

## Euler commands
* `module load gcc/8.2.0 cuda/11.6.2 python/3.8.5 cudnn/8.0.5 cmake/3.19.8 eth_proxy`
* If using virtual environment: `source ../env-3dvision/bin/activate`

Running: 

> `sbatch --time=8:00:00 --gpus=1 --gres=gpumem:24g --cpus-per-task=1 --mem-per-cpu=8g --output=./logs/raw_output --open-mode=append --wrap=""`
>
> `python run_nerf.py --config configs/80v_160_8_792.txt --render_only > ./logs/training_log`

To check details of the job, use `myjobs -j job_id`

Change access
* `chmod -R u+rwx,g+rwx,o+rx ./`

> 15 Apr `sbatch --time=4:00:00 --gpus=1 --gres=gpumem:20g --cpus-per-task=1 --mem-per-cpu=8g --output=./logs/raw_output --open-mode=append --wrap="python run_nerf.py --config configs/60v_230_5_525.txt --render_only --render_mypath > ./logs/test_mypath" `