#!/bin/bash
# conda 환경 활성화
source /home/soo/miniconda3/etc/profile.d/conda.sh
conda activate verl

# current directory 이동
basepath=/home/soo/yejin/CD-RL
cd $basepath

python verl/data/make_subset.py \
    --mode "sample" \
    --frac 0.001 \
    --repeat 5 \
    --seed 42 \
    --input verl/data/dapo-math-17k.parquet \
    2>&1 | tee verl/my_scripts/logs/make_subset.log