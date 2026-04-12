#!/bin/bash
# conda 환경 활성화
source /home/soo/miniconda3/etc/profile.d/conda.sh
conda activate verl

# 기록
LOG_OUTPUT=/home/soo/yejin/verl/my_scripts/logs

huggingface-cli download Qwen/Qwen2.5-3B \
    --local-dir /home/soo/yejin/verl/models/Qwen2.5-3B \
    &> ${LOG_OUTPUT}/download_model.log