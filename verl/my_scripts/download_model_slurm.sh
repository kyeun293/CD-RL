#!/bin/bash
#SBATCH --job-name=download_model
#SBATCH --partition=a6000
#SBATCH --nodelist=node06
#SBATCH --gres=gpu:0
#SBATCH --time=14-0:00:00
#SBATCH --mem=20G
#SBATCH --cpus-per-task=16
#SBATCH --output=/home/yejin/data/projects/yejin/Curiosity/CD-RL/verl/my_scripts/logs/download_model.out

# conda 환경 활성화
ml purge
ml load cuda/12.1
eval "$(conda shell.bash hook)"
conda activate verl

huggingface-cli download Qwen/Qwen2.5-3B-Instruct \
    --local-dir /home/yejin/data/projects/yejin/Curiosity/CD-RL/verl/models/Qwen2.5-3B-Instruct