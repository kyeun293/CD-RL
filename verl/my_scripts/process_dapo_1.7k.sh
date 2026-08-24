#!/bin/bash
# dapo-1.7k.parquet was built by duplicating ~1.7k unique problems 100x each
# (179,600 rows = 1,796 unique extra_info.index values x 100 copies).
# This dedups it back down to the unique problems, keeping the verl-compatible
# schema: data_source, prompt, ability, reward_model, extra_info.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_DIR=${DATA_DIR:-"${HOME}/data"}
IN_FILE=${IN_FILE:-"${DATA_DIR}/dapo-1.7k.parquet"}
OUT_FILE=${OUT_FILE:-"${DATA_DIR}/dapo-1.7k-processed.parquet"}

LOG_DIR="${VERL_ROOT}/my_scripts/logs"
mkdir -p "${LOG_DIR}"

PYTHON_BIN=${PYTHON_BIN:-"/home/sunwoo/miniconda3/envs/cdrl/bin/python3"}

"${PYTHON_BIN}" << EOF 2>&1 | tee "${LOG_DIR}/process_dapo_1.7k.log"
import pandas as pd

df = pd.read_parquet("${IN_FILE}")
print("원본 행 수:", len(df))

df["_dedup_key"] = df["extra_info"].apply(lambda x: x["index"])
before = len(df)
df = df.drop_duplicates(subset="_dedup_key", keep="first").drop(columns="_dedup_key")
df = df.reset_index(drop=True)
print(f"중복 제거: {before} -> {len(df)}")

df.to_parquet("${OUT_FILE}")
print("저장 완료:", "${OUT_FILE}")
EOF
