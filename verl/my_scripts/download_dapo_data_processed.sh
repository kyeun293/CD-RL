#!/bin/bash
# Downloads open-r1/DAPO-Math-17k-Processed (subset: all) and reformats it to the
# verl-compatible schema used by recipe/dapo (matches dapo-math-17k.parquet):
#   data_source, prompt (chat-format list), ability, reward_model, extra_info
#
# The processed dataset's `source_prompt` column already carries this exact
# chat-format prompt, so we just promote it to `prompt` and drop the
# TRL-oriented flat `prompt`/`solution` columns.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SAVE_DIR=${SAVE_DIR:-"${VERL_ROOT}/../data"}
SUBSET=${SUBSET:-all}
OUT_FILE=${OUT_FILE:-"${SAVE_DIR}/dapo-math-17k-processed.parquet"}
mkdir -p "${SAVE_DIR}"

LOG_DIR="${VERL_ROOT}/my_scripts/logs"
mkdir -p "${LOG_DIR}"

PYTHON_BIN=${PYTHON_BIN:-"/home/sunwoo/miniconda3/envs/cdrl/bin/python3"}

"${PYTHON_BIN}" << EOF 2>&1 | tee "${LOG_DIR}/download_dapo_data_processed.log"
from datasets import load_dataset

ds = load_dataset('open-r1/DAPO-Math-17k-Processed', '${SUBSET}', split='train')
print('데이터 수:', len(ds))

def to_verl_format(example):
    return {
        'data_source': example['data_source'],
        'prompt': example['source_prompt'],
        'ability': example['ability'],
        'reward_model': example['reward_model'],
        'extra_info': example['extra_info'],
    }

ds = ds.map(to_verl_format, remove_columns=ds.column_names)
ds.to_parquet('${OUT_FILE}')
print('저장 완료:', '${OUT_FILE}')
EOF
