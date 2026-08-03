#!/usr/bin/env bash
# Start a SECOND private Ray head on GPUs 4-7, ports offset to avoid cluster 1 (8272/6391).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export RAY_TMPDIR=/home/sunwoo/ray_tmp3
export RAY_TEMP_DIR="${RAY_TMPDIR}"
mkdir -p "${RAY_TMPDIR}" "${RAY_TMPDIR}/zmq"

RAY_GCS_PORT=6393
RAY_DASHBOARD_PORT=8274
RAY_AGENT_LISTEN_PORT=52403
RAY_AGENT_GRPC_PORT=35794
RAY_CLIENT_PORT=20003
RAY_MIN_WORKER_PORT=23000
RAY_MAX_WORKER_PORT=23999

export CUDA_VISIBLE_DEVICES=0,1,2,3
PYTHON_BIN="${PYTHON_BIN:-/home/sunwoo/miniconda3/envs/cdrl/bin/python3}"

echo "[INFO] Starting second Ray head on GPUs 0,1,2,3 (dashboard: http://127.0.0.1:${RAY_DASHBOARD_PORT})"

"${PYTHON_BIN}" -m ray.scripts.scripts start --head \
  --temp-dir="${RAY_TEMP_DIR}" \
  --port="${RAY_GCS_PORT}" \
  --dashboard-port="${RAY_DASHBOARD_PORT}" \
  --dashboard-agent-listen-port="${RAY_AGENT_LISTEN_PORT}" \
  --dashboard-agent-grpc-port="${RAY_AGENT_GRPC_PORT}" \
  --ray-client-server-port="${RAY_CLIENT_PORT}" \
  --min-worker-port="${RAY_MIN_WORKER_PORT}" \
  --max-worker-port="${RAY_MAX_WORKER_PORT}" \
  --num-gpus=4 \
  --disable-usage-stats

echo "[INFO] RAY_ADDRESS=http://127.0.0.1:${RAY_DASHBOARD_PORT}"
