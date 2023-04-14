# Depth-Supervison NeRF for project of ETHZ 3DVison Lecture

## Euler commands
* `module load gcc/6.3.0 gcc/8.2.0 eth_proxy`
* `module load python/3.8.5`
* `sbatch --gpus=1 --mem-per-cpu=8g --ntasks=1 --cpus-per-task=1 --gres=gpumem:24g --output=./logs/release/output --open-mode=append --wrap="python run_nerf.py --config configs/fern_2v.txt > ./logs/release/code_output"`
