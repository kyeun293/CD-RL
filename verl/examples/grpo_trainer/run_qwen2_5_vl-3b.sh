#!/bin/bash
# conda 환경 활성화
source /home/soo/miniconda3/etc/profile.d/conda.sh
conda activate verl

# ── 환경 설정 ─────────────────────────────
GPU_ID=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=$GPU_ID
NGPUS_PER_NODE=8
NNODES=1
ENGINE=${1:-vllm}
export TORCHDYNAMO_DISABLE=1

set -x
export NCCL_DEBUG=INFO

# current directory 이동
basepath=/home/soo/yejin/CD-RL/verl
cd $basepath
export PYTHONPATH=$basepath:$PYTHONPATH

# Ray
export RAY_TMPDIR=/dev/shm/yejin_ray_tmp
mkdir -p $RAY_TMPDIR
export VERL_ZMQ_DIR=/dev/shm

# ── 경로 설정 ─────────────────────────────
MODEL_PATH="${basepath}/models/Qwen2.5-3B"
PRM_MODEL_PATH="${basepath}/models/Qwen2.5-Math-PRM-7B"
CKPTS_DIR="${basepath}/ckpts/${project_name}/${exp_name}"
TRAIN_FILE="${basepath}/data/dapo-math-17k.parquet"
TEST_FILE="${basepath}/data/aime-2024.parquet"

# ── 학습 하이퍼파라미터 ───────────────────
adv_estimator=grpo
max_prompt_length=2048
max_response_length=4096
train_batch_size=512
n_resp_per_prompt=5
ppo_mini_batch_size=128 # train_prompt_mini_bsz
ppo_micro_batch_size_per_gpu=8
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * n_resp_per_prompt))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 2))

# ── KL 설정 ──────────────────────────────
use_kl_in_reward=False
use_kl_loss=True
kl_loss_coef=0.01
# ── Rollout 설정 ─────────────────────────
gen_tp=2
gpu_memory_utilization=0.6
use_dynamic_bsz=True
# ── Trainer 설정 ─────────────────────────
total_epochs=1
save_freq=20
test_freq=5
offload=False
# ── Curiosity 설정 ─────────────────────────
use_curiosity=True
STEP_SEP="Step"
icm_intermediate_size=8192
icm_lr=1e-4
icm_lr_scheduler_type="linear"
icm_warmup_steps=10
icm_intrinsic_reward_token="all_step_tokens" # "last_step_token" or "all_step_tokens"
icm_eta=0.04

# ── 기록 ─────────────────────────
project_name='GRPO'
exp_name='GRPO-Qwen2.5-3B'
LOG_OUTPUT=/home/soo/yejin/CD-RL/verl/my_scripts/logs

# ─── 시작 전 이전 잔여 프로세스 정리 ────────────────────────────────────────
echo "[cleanup] Killing any leftover Ray/DAPO processes..."
pkill -9 -u $USER -f "main_dapo"   2>/dev/null || true
pkill -9 -u $USER -f "ray::"       2>/dev/null || true
pkill -9 -u $USER -f "ray/dashboard" 2>/dev/null || true
ray stop --force 2>/dev/null || true
sleep 2
rm -rf /dev/shm/${USER}_ray_tmp
rm -rf /tmp/ray
rm -f /tmp/rl-colocate-zmq-*.sock  # ← 추가
echo "[cleanup] Done."
 
# ─── 종료 시 자동 정리 (Ctrl+C / kill / 정상 종료 모두 처리) ─────────────────
cleanup() {
    echo "[cleanup] Caught exit signal. Cleaning up..."
    # 로그 먼저 백업
    cp -r /dev/shm/${USER}_ray_tmp/session_latest/logs/ /tmp/ray_logs_backup/ 2>/dev/null || true
    pkill -9 -u $USER -f "main_dapo"   2>/dev/null || true
    pkill -9 -u $USER -f "ray::"       2>/dev/null || true
    pkill -9 -u $USER -f "ray/dashboard" 2>/dev/null || true
    rm -rf /dev/shm/${USER}_ray_tmp
    rm -rf /tmp/ray
    echo "[cleanup] Done."
}
trap cleanup EXIT INT TERM

# ── python3 실행 ─────────────────────────
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$adv_estimator \
    data.train_files=$TRAIN_FILE \
    data.val_files=$TEST_FILE \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=False \
    data.truncation=left \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.step_sep=${STEP_SEP} \
    actor_rollout_ref.prm_model_path="${PRM_MODEL_PATH}" \
    actor_rollout_ref.icm.icm_intermediate_size=${icm_intermediate_size} \
    actor_rollout_ref.icm.lr=${icm_lr} \
    actor_rollout_ref.icm.lr_scheduler_type=${icm_lr_scheduler_type} \
    actor_rollout_ref.icm.warmup_steps=${icm_warmup_steps} \
    actor_rollout_ref.icm.intrinsic_reward_token=${icm_intrinsic_reward_token} \
    actor_rollout_ref.icm.eta=${icm_eta} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=$offload \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$offload \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=20 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$gen_tp \
    actor_rollout_ref.rollout.name=$ENGINE \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=$n_resp_per_prompt \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=20 \
    actor_rollout_ref.ref.fsdp_config.param_offload=$offload \
    algorithm.use_kl_in_reward=$use_kl_in_reward \
    algorithm.use_curiosity=$use_curiosity \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=$project_name \
    trainer.experiment_name=$exp_name \
    trainer.n_gpus_per_node=$NGPUS_PER_NODE \
    trainer.nnodes=$NNODES \
    trainer.save_freq=40 \
    trainer.test_freq=40 \
    trainer.total_epochs=$total_epochs \
    2>&1 | tee "${LOG_OUTPUT}/${exp_name}.log"
