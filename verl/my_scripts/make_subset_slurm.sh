#!/bin/bash
#SBATCH --job-name=make_subset
#SBATCH --partition=a6000
#SBATCH --nodelist=node06
#SBATCH --gres=gpu:0
#SBATCH --time=14-0:00:00
#SBATCH --mem=5G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/yejin/data/projects/yejin/Curiosity/CD-RL/verl/my_scripts/logs/make_subset.out

# conda 환경 활성화
ml purge
ml load cuda/12.1
eval "$(conda shell.bash hook)"
conda activate verl

basepath=/home/yejin/data/projects/yejin/Curiosity/CD-RL/verl
cd $basepath

python data/make_subset.py \
    --mode "sample" \
    --frac 0.001 \
    --repeat 5 \
    --seed 42 \
    --input data/dapo-math-17k.parquet \
    --output_dir /home/yejin/data/projects/yejin/Curiosity/CD-RL/verl/data