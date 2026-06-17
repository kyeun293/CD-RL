#!/usr/bin/env bash
# Start a private Ray head on ports unlikely to conflict on a shared server.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export RAY_TMPDIR="${RAY_TMPDIR:-${VERL_ROOT}/ray_tmp}"
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-${RAY_TMPDIR}}"
mkdir -p "${RAY_TMPDIR}" "${RAY_TEMP_DIR}"

RAY_GCS_PORT="${RAY_GCS_PORT:-6391}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8272}"
RAY_AGENT_LISTEN_PORT="${RAY_AGENT_LISTEN_PORT:-52400}"
RAY_AGENT_GRPC_PORT="${RAY_AGENT_GRPC_PORT:-35792}"
RAY_CLIENT_PORT="${RAY_CLIENT_PORT:-20001}"
RAY_MIN_WORKER_PORT="${RAY_MIN_WORKER_PORT:-21000}"
RAY_MAX_WORKER_PORT="${RAY_MAX_WORKER_PORT:-21999}"

export RAY_ADDRESS="http://127.0.0.1:${RAY_DASHBOARD_PORT}"
export RAY_API_SERVER_ADDRESS="${RAY_ADDRESS}"
PYTHON_BIN="${PYTHON_BIN:-/home/soo/miniconda3/envs/cdrl/bin/python3}"

echo "[INFO] Stopping any Ray processes owned by this user..."
"${PYTHON_BIN}" -m ray.scripts.scripts stop --force 2>/dev/null || true
sleep 2

echo "[INFO] Starting Ray head (logs: ${RAY_TEMP_DIR}/session_*/logs/)"
"${PYTHON_BIN}" -m ray.scripts.scripts start --head \
  --temp-dir="${RAY_TEMP_DIR}" \
  --port="${RAY_GCS_PORT}" \
  --dashboard-port="${RAY_DASHBOARD_PORT}" \
  --dashboard-agent-listen-port="${RAY_AGENT_LISTEN_PORT}" \
  --dashboard-agent-grpc-port="${RAY_AGENT_GRPC_PORT}" \
  --ray-client-server-port="${RAY_CLIENT_PORT}" \
  --min-worker-port="${RAY_MIN_WORKER_PORT}" \
  --max-worker-port="${RAY_MAX_WORKER_PORT}" \
  --disable-usage-stats

echo "[INFO] RAY_ADDRESS=${RAY_ADDRESS}  (use this for: ray job submit --address ...)"
echo "[INFO] GCS port=${RAY_GCS_PORT}  (use this for: ray start --address 127.0.0.1:${RAY_GCS_PORT} to join as worker)"
echo "[INFO] Dashboard port=${RAY_DASHBOARD_PORT}  (NOT the GCS port; do not pass to ray start --address)"
