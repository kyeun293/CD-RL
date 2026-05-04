#!/bin/bash
#SBATCH --job-name=prepare_dapo_data
#SBATCH --partition=a6000
#SBATCH --nodelist=node03
#SBATCH --gres=gpu:0
#SBATCH --time=14-0:00:00
#SBATCH --mem=10G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/yejin/data/projects/yejin/Curiosity/CD-RL/verl/my_scripts/logs/prepare_dapo_data.out

# conda 환경 활성화
ml purge
ml load cuda/12.1
eval "$(conda shell.bash hook)"
conda activate verl

basepath=/home/yejin/data/projects/yejin/Curiosity/CD-RL/verl

export VERL_HOME=${VERL_HOME:-"${basepath}"}
export TRAIN_FILE=${TRAIN_FILE:-"${VERL_HOME}/data/dapo-math-17k.parquet"}
export TEST_FILE=${TEST_FILE:-"${VERL_HOME}/data/aime-2024.parquet"}
export OVERWRITE=${OVERWRITE:-0}

mkdir -p "${VERL_HOME}/data"

if [ ! -f "${TRAIN_FILE}" ] || [ "${OVERWRITE}" -eq 1 ]; then
  wget -O "${TRAIN_FILE}" "https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k/resolve/main/data/dapo-math-17k.parquet?download=true"
fi

if [ ! -f "${TEST_FILE}" ] || [ "${OVERWRITE}" -eq 1 ]; then
  wget -O "${TEST_FILE}" "https://huggingface.co/datasets/BytedTsinghua-SIA/AIME-2024/resolve/main/data/aime-2024.parquet?download=true"
fi
