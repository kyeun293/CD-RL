#!/bin/bash
source /home/soo/miniconda3/etc/profile.d/conda.sh
conda activate verl

set -uxo pipefail

basepath=/home/soo/yejin/verl
LOG_OUTPUT=${basepath}/my_scripts/logs

export VERL_HOME=${VERL_HOME:-"${basepath}"}
export TRAIN_FILE=${TRAIN_FILE:-"${VERL_HOME}/data/dapo-math-17k.parquet"}
export TEST_FILE=${TEST_FILE:-"${VERL_HOME}/data/aime-2024.parquet"}
export OVERWRITE=${OVERWRITE:-0}

mkdir -p "${VERL_HOME}/data"

# 이 줄 이후 모든 출력을 파일로 저장
exec > "${LOG_OUTPUT}/download_data.log" 2>&1

if [ ! -f "${TRAIN_FILE}" ] || [ "${OVERWRITE}" -eq 1 ]; then
  wget -O "${TRAIN_FILE}" "https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k/resolve/main/data/dapo-math-17k.parquet?download=true"
fi

if [ ! -f "${TEST_FILE}" ] || [ "${OVERWRITE}" -eq 1 ]; then
  wget -O "${TEST_FILE}" "https://huggingface.co/datasets/BytedTsinghua-SIA/AIME-2024/resolve/main/data/aime-2024.parquet?download=true"
fi
