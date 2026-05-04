#!/bin/bash
#SBATCH --job-name=download_model2
#SBATCH --partition=a6000
#SBATCH --nodelist=node03
#SBATCH --gres=gpu:0
#SBATCH --time=14-0:00:00
#SBATCH --mem=10G
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/yejin/data/projects/yejin/Curiosity/CD-RL/my_scripts/logs/download_model.out


huggingface-cli download Qwen/Qwen2.5-3B \
    --local-dir /home/yejin/data/projects/yejin/Curiosity/CD-RL/verl/models/Qwen2.5-3B