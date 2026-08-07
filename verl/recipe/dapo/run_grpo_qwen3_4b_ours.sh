#!/usr/bin/env bash
set -xeuo pipefail
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Ray logs + session files under your verl repo (override with env if needed)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# export RAY_TMPDIR="${RAY_TMPDIR:-${VERL_ROOT}/ray_tmp}"
export RAY_TMPDIR=/home/sunwoo/ray_tmp2
# Ray head stores sessions under ${RAY_TMPDIR}/ray; ray.init _temp_dir must match.
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-${RAY_TMPDIR}}"
mkdir -p "${RAY_TMPDIR}" "${RAY_TEMP_DIR}"
echo "[INFO] Ray temp/logs dir: ${RAY_TEMP_DIR}"
echo "[INFO] Worker logs (after job starts): ${RAY_TEMP_DIR}/session_*/logs/"

export CUDA_VISIBLE_DEVICES=4,5,6,7
export NCCL_SOCKET_NTHREADS=2
export NCCL_NSOCKS_PERTHREAD=8
export CUDA_DEVICE_MAX_CONNECTIONS=1
# Default matches recipe/dapo/start_ray_local.sh (8272). Do NOT use 8266/8267 (shared clusters).
export RAY_ADDRESS="${RAY_ADDRESS:-http://127.0.0.1:8273}"
export RAY_API_SERVER_ADDRESS="${RAY_API_SERVER_ADDRESS:-${RAY_ADDRESS}}"
PYTHON_BIN="${PYTHON_BIN:-/home/sunwoo/miniconda3/envs/cdrl/bin/python3}"
RAY_BIN="${RAY_BIN:-$(dirname "${PYTHON_BIN}")/ray}"
echo "[INFO] RAY_ADDRESS=${RAY_ADDRESS}"
echo "[INFO] PYTHON_BIN=${PYTHON_BIN}"
echo "[INFO] RAY_BIN=${RAY_BIN}"



# ── Debug mode ──────────────────────────────────────────────────────────────
# Set DEBUG=true to overfit on 32 fixed samples (1 batch) for fast iteration.
DEBUG=${DEBUG:-False}
# ────────────────────────────────────────────────────────────────────────────

project_name='CDRL-Qwen3-4B-Base-dapo17k'
exp_name='GRPO-vanilla-Qwen3-4B-Base'

adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=True
kl_loss_coef=0.001
clip_ratio_low=0.2
clip_ratio_high=0.2
max_prompt_length=$((2000))
max_response_length=$((3000))
enable_overlong_buffer=False
overlong_buffer_len=$((256))
overlong_penalty_factor=1.0
loss_agg_mode="token-mean"
enable_filter_groups=False
filter_groups_metric=acc
max_num_gen_batches=0

NNODES=1
NGPUS_PER_NODE=4

train_prompt_bsz=256
gen_prompt_bsz=$((train_prompt_bsz * 2))
n_resp_per_prompt=8
train_prompt_mini_bsz=256

if [[ "${DEBUG}" == "true" ]]; then
    exp_name="${exp_name}-debug32"
    train_prompt_bsz=32
    gen_prompt_bsz=32
    train_prompt_mini_bsz=8
    RESUME_MODE="disable"
    RESUME_FROM_PATH=""
    echo "[DEBUG] Overfit mode: 32 fixed samples, fresh start"
fi

# Ray (RAY_ADDRESS set above; do not override with 8266)
PWD=./
WORKING_DIR=${WORKING_DIR:-"${VERL_ROOT}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${VERL_ROOT}/recipe/dapo/runtime_env_gpu4567.yaml"}

# Paths
DATA_HOME_PATH=${DATA_HOME_PATH:-"${HOME}/data"}
CKPT_HOME_PATH=${CKPT_HOME_PATH:-"${HOME}/ckpts"}
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-4B-Base"}
PRM_MODEL_PATH=${PRM_MODEL_PATH:-"Qwen/Qwen2.5-Math-PRM-7B"}
CKPTS_DIR=${CKPTS_DIR:-"${CKPT_HOME_PATH}/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${DATA_HOME_PATH}/dapo-math-17k-processed.parquet"}
TEST_FILE=${TEST_FILE:-"${DATA_HOME_PATH}/aime-2024.parquet"}

# Resume: auto-detect latest checkpoint from latest_checkpointed_iteration.txt
# Set FRESH_START=true to ignore checkpoints and start from step 0.
FRESH_START=${FRESH_START:-false}
LATEST_ITER_FILE="${CKPTS_DIR}/latest_checkpointed_iteration.txt"
if [[ "${FRESH_START}" == "true" ]]; then
    RESUME_MODE="disable"
    RESUME_FROM_PATH=""
    echo "[INFO] FRESH_START=true; ignoring checkpoints, starting from step 0"
elif [[ -f "${LATEST_ITER_FILE}" ]]; then
    LATEST_STEP=$(cat "${LATEST_ITER_FILE}")
    RESUME_PATH="${CKPTS_DIR}/global_step_${LATEST_STEP}"
    if [[ -d "${RESUME_PATH}" ]]; then
        RESUME_MODE="resume_path"
        RESUME_FROM_PATH="${RESUME_PATH}"
        echo "[INFO] Resuming from latest checkpoint: ${RESUME_PATH}"
    else
        RESUME_MODE="disable"
        RESUME_FROM_PATH=""
        echo "[WARN] latest_checkpointed_iteration.txt points to ${RESUME_PATH} but dir not found; starting fresh"
    fi
else
    RESUME_MODE="disable"
    RESUME_FROM_PATH=""
    echo "[INFO] No checkpoint found at ${CKPTS_DIR}; starting fresh"
fi

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1  # 0 for HF rollout, -1 for vLLM rollout

# Performance Related Parameter
sp_size=1
use_dynamic_bsz=True
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 2))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 2))
offload=False
gen_tp=2

# curiosity 추가한 인자
#[Sunwoo]
#weight_overload = Ture/False # False: default
#weight_overload_method =  "1epoch" "2epoch" "mixing" : overload policy--> ref every 1/2epoch /
#percentage_of_mixing = 0.0 ~ 1.0 : mixing policy--> ref every x% of the total training steps
use_curiosity=True
overload_actor_to_ref=False
overload_actor_to_ref_freq=100
STEP_SEP="Step"
icm_intermediate_size=8192
icm_lr=1e-4
icm_lr_scheduler_type="linear"
icm_warmup_steps=10
icm_intrinsic_reward_token="last_step_token" # "last_step_token" or "all_step_tokens"
icm_calculation="normalize_prm" # "clip" or "normalize" or "whiten" or "normalize_prm" or "whiten_prm"
icm_eta=0.5
variance_gated_curiosity=True  #SW: replace fixed icm_eta/eta_token/eta_answer with per-group rollout-correctness std
dynamic_eta_coefficient=1.0  #SW: total_reward = extrinsic_reward + dynamic_eta_coefficient * dynamic_eta * intrinsic_reward
dynamic_eta_beta=0.05  #SW: saturating exponential reshape of the raw std eta; 0/null disables it

prm_mask=True  #SOO: PRM Modification, Default is True
use_tokenlevel_curiosity=False  #SOO: token level ICM
eta_token=0.04                  #SOO: token level ICM
use_answerlevel_curiosity=False  #SOO: answer level ICM
eta_answer=0.1                   #SOO: answer level ICM

# SW: Added trainer.validation_data_dir => validation 결과(response 포함) 체크포인트 저장

"${RAY_BIN}" job submit --no-wait --runtime-env="${RUNTIME_ENV}" \
    --working-dir "${WORKING_DIR}" \
    --address "${RAY_ADDRESS}" \
    -- "${PYTHON_BIN}" -m recipe.dapo.main_dapo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.train_max_samples=$([ "${DEBUG}" == "true" ] && echo 32 || echo -1) \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.train_batch_size=${train_prompt_bsz} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.use_curiosity=${use_curiosity} \
    algorithm.overload_actor_to_ref=${overload_actor_to_ref} \
    algorithm.overload_actor_to_ref_freq=${overload_actor_to_ref_freq} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.step_sep=${STEP_SEP} \
    actor_rollout_ref.prm_model_path="${PRM_MODEL_PATH}" \
    actor_rollout_ref.icm.icm_intermediate_size=${icm_intermediate_size} \
    actor_rollout_ref.icm.lr=${icm_lr} \
    actor_rollout_ref.icm.lr_scheduler_type=${icm_lr_scheduler_type} \
    actor_rollout_ref.icm.warmup_steps=${icm_warmup_steps} \
    actor_rollout_ref.icm.intrinsic_reward_token=${icm_intrinsic_reward_token} \
    actor_rollout_ref.icm.icm_calculation=${icm_calculation} \
    actor_rollout_ref.icm.eta=${icm_eta} \
    actor_rollout_ref.icm.prm_mask=${prm_mask} \
    algorithm.variance_gated_curiosity=${variance_gated_curiosity} \
    algorithm.dynamic_eta_coefficient=${dynamic_eta_coefficient} \
    algorithm.dynamic_eta_beta=${dynamic_eta_beta} \
    algorithm.use_tokenlevel_curiosity=${use_tokenlevel_curiosity} \
    actor_rollout_ref.tokenlevel_icm.eta_token=${eta_token} \
    algorithm.use_answerlevel_curiosity=${use_answerlevel_curiosity} \
    actor_rollout_ref.answerlevel_icm.eta_answer=${eta_answer} \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    +actor_rollout_ref.model.override_config.attention_dropout=0. \
    +actor_rollout_ref.model.override_config.embd_pdrop=0. \
    +actor_rollout_ref.model.override_config.resid_pdrop=0. \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    reward.reward_manager.name=dapo \
    reward.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    reward.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    reward.reward_kwargs.overlong_buffer_cfg.log=False \
    reward.reward_kwargs.max_resp_len=${max_response_length} \
    actor_rollout_ref.prm_model_path="${PRM_MODEL_PATH}" \
    trainer.logger="['console', 'wandb']" \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=True \
    trainer.test_freq=5 \
    trainer.save_freq=50 \
    trainer.total_epochs=$([ "${DEBUG}" == "true" ] && echo 10 || echo 10) \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.validation_data_dir="${CKPTS_DIR}" \
    trainer.resume_mode=${RESUME_MODE} \
    ${RESUME_FROM_PATH:+trainer.resume_from_path="${RESUME_FROM_PATH}"} \
    trainer.save_curiosity_scores=True \
    actor_rollout_ref.actor.entropy_checkpointing=True \
    actor_rollout_ref.ref.entropy_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.ref.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=True \
    +ray_kwargs.ray_init.include_dashboard=False \
    +ray_kwargs.ray_init._temp_dir=${RAY_TEMP_DIR}

