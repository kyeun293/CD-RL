#!/bin/bash
source /home/soo/miniconda3/etc/profile.d/conda.sh
conda activate verl

cd verl
export PYTHONPATH=/home/soo/yejin/verl:$PYTHONPATH

basepath=/home/soo/yejin/CD-RL/verl

SAVE_DIR=${basepath}/data
mkdir -p ${SAVE_DIR}

LOG_OUTPUT=${basepath}/my_scripts/logs
mkdir -p ${LOG_OUTPUT}

python3 << EOF 2>&1 | tee ${LOG_OUTPUT}/download_dapo_data.log
from datasets import load_dataset
ds = load_dataset('BytedTsinghua-SIA/DAPO-Math-17k', split='train')
print('데이터 수:', len(ds))
ds.to_parquet('${SAVE_DIR}/dapo-math-17k.parquet')
print('저장 완료')
EOF