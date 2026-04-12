#!/bin/bash
# conda 환경 활성화
source /home/soo/miniconda3/etc/profile.d/conda.sh
conda activate verl
# GPU 설정 (GPU 3번 사용)
export CUDA_VISIBLE_DEVICES=2

# current directory 이동
basepath=/home/soo/yejin/verl
cd $basepath

python -c "import flash_attn; print('flash_attn:', flash_attn.__version__)" \
    &> $basepath/my_scripts/logs/debug.log