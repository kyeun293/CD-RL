#!/bin/bash
source /home/soo/miniconda3/etc/profile.d/conda.sh
conda activate verl

cd verl
export PYTHONPATH=/home/soo/yejin/verl:$PYTHONPATH

basepath=/home/soo/yejin/verl

SAVE_DIR=${basepath}/data
mkdir -p ${SAVE_DIR}

LOG_OUTPUT=${basepath}/my_scripts/logs
mkdir -p ${LOG_OUTPUT}

python examples/data_preprocess/gsm8k.py \
    --local_save_dir ${SAVE_DIR}/gsm8k \
    > ${LOG_OUTPUT}/download_data.log 2>&1
