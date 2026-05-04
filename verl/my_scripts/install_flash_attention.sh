#!/bin/bash
# conda 환경 활성화
source /home/soo/miniconda3/etc/profile.d/conda.sh
conda activate verl

LOG_OUTPUT=/home/soo/yejin/verl/my_scripts/logs
exec > >(tee "${LOG_OUTPUT}/install_flash_attention.log") 2>&1

pip uninstall flash-attn -y
MAX_JOBS=4 pip install flash-attn --no-build-isolation --no-cache-dir