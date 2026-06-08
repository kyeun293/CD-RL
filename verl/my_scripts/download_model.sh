#!/bin/bash
# conda 환경 활성화
source /home/soo/miniconda3/etc/profile.d/conda.sh
conda activate verl

huggingface-cli download Qwen/Qwen2.5-3B-Instruct \
    --local-dir /home/soo/yejin/CD-RL/verl/models/Qwen2.5-3B-Instruct \
    2>&1 | tee /home/soo/yejin/CD-RL/verl/my_scripts/logs/download_model.log