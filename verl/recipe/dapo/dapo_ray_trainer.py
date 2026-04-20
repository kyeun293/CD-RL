# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import re
import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.reward import extract_reward
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.metric import reduce_metrics
from verl.utils.profiler import marked_timer
from verl.utils.rollout_skip import RolloutSkip

import ray
from tensordict import TensorDict

class RayDAPOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        reward_kwargs = OmegaConf.to_container(self.config.reward.reward_kwargs, resolve=True)
        self._prm_model_path = reward_kwargs.get("prm_model_path", None)
        self._step_sep = reward_kwargs.get("step_sep", None)
        print(f"[INIT-DEBUG] step_sep: {repr(self._step_sep)}", flush=True)
        self.curiosity_actor = None
        # self._prm_scorer = None
        # self.icm = None

    def init_workers(self):
        super().init_workers()

        if self._prm_model_path and self.config.algorithm.use_icm:
            from recipe.dapo.curiosity_actor import CuriosityActor
            from transformers import AutoConfig

            model_config = AutoConfig.from_pretrained(self.config.actor_rollout_ref.model.path)
            hidden_size = model_config.hidden_size
            intermediate_size = self.config.icm.icm_intermediate_size
            total_training_steps = self.config.trainer.total_epochs * len(self.train_dataloader)

            self.curiosity_actor = CuriosityActor.remote(
                self._prm_model_path,
                hidden_size,
                intermediate_size,
                self.config.icm,
                total_training_steps,
            )
            print(f"[INIT-DEBUG] CuriosityActor ready.", flush=True)

    def _find_sep_positions(self, token_ids):
        """Return token indices whose decoded string ends with the step separator."""
        positions = []
        for j in range(len(token_ids)):
            tok_str = self.tokenizer.decode([token_ids[j].item()])
            if tok_str.endswith(self._step_sep):
                positions.append(j)
        return positions

    def _attach_parsed_steps(self, batch):
        all_steps = []
        all_boundaries = []

        for i in range(len(batch)):
            response_ids = batch.batch["responses"][i]
            response_mask = batch.batch["response_mask"][i]  # 1이 실제 토큰, 0이 패딩
            valid_response_ids = response_ids[response_mask.bool()]
            valid_len = int(response_mask.sum().item())

            if self._step_sep:
                # Find step boundaries at token level so PRM (text) and ICM (ids) share
                # the exact same step indices.
                boundaries = self._find_sep_positions(valid_response_ids) # step separator가 있는 토큰의 인덱스: [60, 209, 314, 394, 453, 519, 645, 672]

                if not boundaries or len(boundaries) < 2:
                    step_texts = [self.tokenizer.decode(valid_response_ids, skip_special_tokens=True).strip()]
                    boundaries = [0, valid_len]
                    all_steps.append(step_texts)
                    all_boundaries.append(boundaries)
                    continue

                if boundaries[-1] != valid_len:
                    boundaries.append(valid_len)

                # 각 step 텍스트 추출
                step_texts = []
                for k in range(len(boundaries) - 1):
                    seg = valid_response_ids[boundaries[k]:boundaries[k+1]]
                    text = self.tokenizer.decode(seg, skip_special_tokens=True).strip()
                    step_texts.append(text)
                
                if i == 0:
                    # print(f"[PARSE-DEBUG] boundaries: {boundaries}", flush=True) # [0, 27, 51, 71, 93, 117]
                    # print(f"[PARSE-DEBUG] step_texts: {step_texts}", flush=True) # ["Step 1: ...", "Step 2: ...", ..., "Step 8: ... Answer: ..."]
                    # print(f"[PARSE-DEBUG] step len: {len(step_texts)}", flush=True) # 5
                    print(f"[PARSE-DEBUG] i={i} steps={len(step_texts)} boundaries={boundaries}", flush=True)
                    for si, s in enumerate(step_texts):
                        print(f"  step[{si}]: {repr(s[:120])}", flush=True)

            else:
                step_texts = []
                boundaries = []

            all_steps.append(step_texts)
            all_boundaries.append(boundaries)

        batch.non_tensor_batch["parsed_steps"] = np.array(all_steps, dtype=object)
        batch.non_tensor_batch["step_boundaries"] = np.array(all_boundaries, dtype=object)

        return batch

    def _score_prm(self, batch):
        problem_strs, steps_list = [], []
        
        for i in range(len(batch)):
            steps = batch.non_tensor_batch["parsed_steps"][i]
            if steps:
                prompt_ids = batch.batch["prompts"][i]
                prompt_length = prompt_ids.shape[-1]
                prompt_mask = batch.batch["attention_mask"][i][:prompt_length].bool()
                valid_prompt_ids = prompt_ids[prompt_mask]
                problem_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
                problem_strs.append(problem_str)
                steps_list.append(steps)
            else:
                problem_strs.append("")
                steps_list.append([])

        all_labels = ray.get(self.curiosity_actor.score_prm_batch.remote(problem_strs, steps_list))
        
        if len(steps_list[0]) > 0:
            print(f"[PRM-DEBUG] i=0 steps={len(steps_list[0])} labels={all_labels[0]}", flush=True) # i=0 steps=5 labels=[1, 1, 1, 1, 0]
        
        batch.non_tensor_batch["prm_labels"] = np.array(all_labels, dtype=object)
        return batch

    def compute_kl_related_metrics(self, batch: DataProto, metrics: dict, timing_raw: dict):
        batch.batch["response_mask"] = compute_response_mask(batch)

        # recompute old_log_probs
        with marked_timer("old_log_prob", timing_raw, "blue"):
            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
            entropys = old_log_prob.batch["entropys"]
            response_masks = batch.batch["response_mask"]
            actor_config = self.config.actor_rollout_ref.actor
            entropy_agg = agg_loss(
                loss_mat=entropys,
                loss_mask=response_masks,
                loss_agg_mode=actor_config.loss_agg_mode,
                loss_scale_factor=actor_config.loss_scale_factor,
            )
            old_log_prob_metrics = {
                "actor/entropy": entropy_agg.detach().item(),
                "perf/mfu/actor_infer": old_log_prob_mfu,
            }
            metrics.update(old_log_prob_metrics)
            old_log_prob.batch.pop("entropys")
            batch = batch.union(old_log_prob)

        if self.use_reference_policy:
            # compute reference log_prob
            with marked_timer("ref", timing_raw, "olive"):
                if "ref_log_prob" not in batch.batch: # 이미 compute_ref_kl_hidden_states에서 계산되어 있을 수 있음
                    ref_log_prob = self._compute_ref_log_prob(batch)
                    batch = batch.union(ref_log_prob)
                else:
                    print("ref_log_prob already in batch, skipping computation", flush=True)

        return batch

    def compute_ref_kl_hidden_states(self, batch: DataProto):

        batch.meta_info["output_hidden_states"] = True  # 추가
        ref_log_prob = self._compute_ref_log_prob(batch)
        batch = batch.union(ref_log_prob)
        batch.meta_info["output_hidden_states"] = False

        return batch
    
    def get_actor_step_embs_chunk(self, token_ids_list: list[torch.Tensor]) -> list[torch.Tensor]:
        """
            token_ids_list: 여러 reasoning step 시퀀스가 담긴 리스트
        """
        bs = len(token_ids_list)

        # dp_size의 배수로 패딩
        dp_size = self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes \
                // self.config.actor_rollout_ref.actor.ulysses_sequence_parallel_size
        padded_bs = ((bs + dp_size - 1) // dp_size) * dp_size
        pad_count = padded_bs - bs
        padded_token_ids_list = token_ids_list + [token_ids_list[0]] * pad_count

        # dummy prompt 추가해서 데이터 만들기
        max_resp_len = max(ids.shape[0] for ids in padded_token_ids_list)

        input_ids = torch.zeros(padded_bs, max_resp_len, dtype=torch.long)
        attention_mask = torch.zeros(padded_bs, max_resp_len, dtype=torch.long)
        position_ids = torch.zeros(padded_bs, max_resp_len, dtype=torch.long)
        prompt_ids = torch.zeros(padded_bs, 0, dtype=torch.long)  # empty prompt
        response_ids = torch.zeros(padded_bs, max_resp_len, dtype=torch.long)
        response_mask = torch.zeros(padded_bs, max_resp_len, dtype=torch.long)

        for i, ids in enumerate(padded_token_ids_list):
            l = ids.shape[0]
            input_ids[i, :l] = ids
            attention_mask[i, :l] = 1
            position_ids[i, :l] = torch.arange(l)
            response_ids[i, :l] = ids
            response_mask[i, :l] = 1
            
        # actor 모델로 hidden state 구하기
        step_batch = DataProto(batch=TensorDict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "responses": response_ids,
                "response_mask": response_mask,
                "position_ids": position_ids,
                "prompts": prompt_ids,
            },
            batch_size=[padded_bs],
        ))
        step_batch.meta_info["output_hidden_states"] = True
        step_batch.meta_info["compute_loss"] = False
        step_batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

        step_hidden_states = self._compute_actor_hidden_states(step_batch)

        # 각 step의 마지막 토큰의 hidden state만 추출
        step_last_hidden_states = [
            step_hidden_states[i, response_mask[i].sum() - 1]
            for i in range(bs) # padded_bs가 아닌 원래 bs까지만
        ]

        # print(f"step_hidden_states shape: {step_hidden_states.shape}", flush=True) # torch.Size([num_steps_in_batch, seq_len, hidden_dim])

        return step_last_hidden_states

    def get_actor_step_embs(self, batch, step_boundary_list):
        """
        new_batch: DataProto, responses 토큰 포함
        step_boundary_list: [[0, 61, 210, ...], [...], ...] 각 response의 step boundary 인덱스 리스트 (response 내에서의 토큰 인덱스)
        
        returns: list of list of tensors
            - 외부 리스트: 각 response
            - 내부 리스트: 각 step의 임베딩 (hidden_dim,)
        """
        responses = batch.batch["responses"]
        batch_size = responses.shape[0]

        # 1) 모든 response의 모든 step을 하나의 리스트로 flatten
        all_steps = []       # 모든 step의 input_ids
        step_map = []        # (response_idx, step_idx) 복원용 맵
        
        for idx in range(batch_size):
            response_sample = responses[idx]            
            boundaries = step_boundary_list[idx]
            
            for step_idx in range(len(boundaries) - 1):
                start = boundaries[step_idx]
                end = boundaries[step_idx + 1]
                step_tokens = response_sample[start:end]
                all_steps.append(step_tokens)
                step_map.append((idx, step_idx))


        # 2) chunk 단위로 모델에 입력하여 step별 hidden state 구하기
        step_last_hidden_states = [[] for _ in range(batch_size)]
        for chunk_start in range(0, len(all_steps), self.config.icm.step_emb_chunk_size):
            chunk_steps = all_steps[chunk_start : chunk_start + self.config.icm.step_emb_chunk_size]
            chunk_hidden_states = self.get_actor_step_embs_chunk(chunk_steps)

            # chunk_hidden_states는 chunk_steps의 step별 hidden state 리스트
            for i, hidden_state in enumerate(chunk_hidden_states):
                global_step_idx = chunk_start + i
                resp_idx, step_idx = step_map[global_step_idx]
                step_last_hidden_states[resp_idx].append(hidden_state)

        step_last_hidden_states = [
            torch.stack(hidden_states, dim=0)  # (num_steps, hidden_dim)
            for hidden_states in step_last_hidden_states
        ]

        return step_last_hidden_states
    
    def intrinsic_reward(self, next_state, next_state_hat):
        intrinsic_reward = 0.5 * (next_state - next_state_hat).norm(2, dim=-1)
        intrinsic_reward = self.whiten(intrinsic_reward)
        intrinsic_reward = intrinsic_reward.detach()
        return intrinsic_reward
    
    def add_intrinsic_reward(self, batch, intrinsic_reward, pair_map, step_boundary_list):
        """
        batch: DataProto
        intrinsic_reward: (num_steps,) 각 step에 대한 intrinsic reward 텐서
        pair_map: [(data_idx, step_idx), ...] 각 step이 어떤 데이터의 어떤 step인지 매핑하는 리스트 (마지막 step은 없음. ICM에서는 다음 step을 예측하는 것이니까)
        step_boundary_list: [[step_boundary], ...] 각 response의 step boundary 정보

        batch에 token_level_intrinsic_rewards 추가하기
        """
        # total = sum(len(batch.non_tensor_batch['prm_labels'][i]) for i in range(len(batch.non_tensor_batch['prm_labels'])))
        # print(f"pair_map len: {len(pair_map)}, intrinsic_reward len: {len(intrinsic_reward)}, prm_labels len: {total}", flush=True)
        # pair_map len: 1557, intrinsic_reward len: 1557, prm_labels len: 1557
        
        for k, (data_idx, step_idx) in enumerate(pair_map):
            boundary = step_boundary_list[data_idx]
            start = boundary[step_idx]
            end = boundary[step_idx + 1]
            prm_label = batch.non_tensor_batch['prm_labels'][data_idx][step_idx]

            if self.config.icm.intrinsic_reward_token == "all_step_tokens":
                batch.batch["token_level_rewards"][data_idx, start:end] += self.config.icm.eta * intrinsic_reward[k].item() * prm_label
            elif self.config.icm.intrinsic_reward_token == "last_step_token":
                batch.batch["token_level_rewards"][data_idx, end - 1] += self.config.icm.eta * intrinsic_reward[k].item() * prm_label

        return batch

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self.gen_steps = 0
        self.max_steps_duration = 0

        # load checkpoint before doing anything
        self._load_checkpoint()
        self.checkpoint_manager.update_weights()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.async_rollout_manager)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        self.gen_steps += 1
        last_val_metrics = None

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        timing_raw = defaultdict(float)
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        current_epoch = self.global_steps // len(self.train_dataloader)

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                new_batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                num_gen_batches += 1
                gen_batch = self._get_gen_batch(new_batch)
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, "red"):
                        print(f"[GEN-DEBUG] Generating responses", flush=True)
                        gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
                        gen_batch_output = self._attach_parsed_steps(gen_batch_output)
                        
                        if self.curiosity_actor is not None:
                            gen_batch_output = self._score_prm(gen_batch_output)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, "red"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)

                            new_batch = new_batch.union(gen_baseline_output)
                            # compute reward model score on new_batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in new_batch.batch.keys():
                                rm_scores = self._compute_reward_colocate(new_batch)
                                new_batch = new_batch.union(rm_scores)
                            reward_baseline_tensor, _ = extract_reward(new_batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            new_batch.pop(batch_keys=list(keys_to_pop))

                            new_batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output

                    new_batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
                    )
                    # repeat to align with repeated responses in rollout
                    new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    new_batch = new_batch.union(gen_batch_output)

                    # print(f"[PRM-DEBUG] prm_labels: {new_batch.non_tensor_batch['prm_labels'][:5]} (showing first 5 samples)", flush=True)
                    # [list([1, 1, 1, 1, 0]) list([1, 1, 1, 1, 1, 0]) list([1, 1, 1, 1, 0, 0]) list([1, 0, 0]) list([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])]
                
                    if self.config.algorithm.use_icm:
                        # 각 response 내 step 경계 위치 (response 기준 인덱스)
                        step_boundary_list = new_batch.non_tensor_batch["step_boundaries"].tolist()
                        # print(f"[ICM-DEBUG] step_boundary_list: {step_boundary_list[:5]} (showing first 5 samples)", flush=True)
                        # [[0, 27, 51, 71, 93, 117], [0, 23, 62, 121, 164, 193, 222], [73, 161, 259, 586, 704, 734, 963], [0, 123, 201, 248], [0, 41, 73, 101, 163, 199, 236, 260, 301, 329, 490]] (batch size 256)

                        # reference 모델로 response 부분 hidden state 구하기
                        new_batch = self.compute_ref_kl_hidden_states(new_batch)
                        ref_hidden_states_raw = new_batch.batch["ref_hidden_states"]
                        
                        # 각 step 경계 위치마다 hidden state 뽑기
                        # s_0은 'step' 토큰을 임베딩으로 이용 (프롬프트의 마지막 토큰 임베딩과 비슷할 것이므로)
                        # s_1부터는 step의 마지막 토큰을 임베딩으로 이용
                        print(f"[ICM-DEBUG] Computing reference hidden states", flush=True)
                        ref_hidden_states = []
                        for i in range(len(step_boundary_list)):
                            boundaries = step_boundary_list[i]
                            indices = np.concatenate([[boundaries[0]], np.array(boundaries[1:]) - 1])
                            ref_hidden_states.append(ref_hidden_states_raw[i, indices, :])
                        # print(f"ref_hidden_states len: {len(ref_hidden_states)}", flush=True) # batch size = 256
                        # print(f"ref_hidden_states[0] shape: {ref_hidden_states[0].shape}", flush=True) # step 수+1: torch.Size([6, 2048])
                        # print(f"ref_hidden_states[1] shape: {ref_hidden_states[1].shape}", flush=True) # step 수+1: torch.Size([7, 2048])

                        # ref_hidden_states 원본 텐서 삭제
                        del ref_hidden_states_raw
                        del new_batch.batch["ref_hidden_states"]
                        torch.cuda.empty_cache()  # 삭제한 텐서들 실제로 해제

                        # actor 모델로 각 step의 hidden state 구하기
                        print(f"[ICM-DEBUG] Computing actor step embeddings", flush=True)
                        actor_step_embs = self.get_actor_step_embs(new_batch, step_boundary_list)
                        # print(f"actor_step_embs len: {len(actor_step_embs)}", flush=True) # batch size = 256
                        # print(f"actor_step_embs[0] shape: {actor_step_embs[0].shape}", flush=True) # step 수: torch.Size([5, 2048])
                        # print(f"actor_step_embs[1] shape: {actor_step_embs[1].shape}", flush=True) # step 수: torch.Size([6, 2048])

                        # 기존 ICM 계산 코드 전체 교체
                        print(f"[ICM-DEBUG] Computing ICM intrinsic rewards", flush=True)
                        intrinsic_reward, pair_map, icm_loss_val = ray.get(
                            self.curiosity_actor.compute_icm.remote(ref_hidden_states, actor_step_embs)
                        )
                        # print(f"[ICM-DEBUG] icm_loss: {icm_loss_val}", flush=True) 8.375
                        # print(f"[ICM-DEBUG] intrinsic_reward shape:{intrinsic_reward.shape}", flush=True) # torch.Size([1327])
                        # print(f"[ICM-DEBUG] intrinsic_reward sample: {intrinsic_reward[:10]} (showing first 5 samples)", flush=True) # tensor([ 0.6016,  0.1338,  0.4023,  1.0000,  2.0000, -0.3340, -0.7344, -0.4023, -0.3340,  0.2676], dtype=torch.bfloat16)
                        # print(f"[ICM-DEBUG] prm labels: {new_batch.non_tensor_batch['prm_labels'][:5]} (showing first 5 samples)", flush=True) # [list([1, 1, 1, 1, 0]) list([1, 1, 1, 1, 1, 0]) list([1, 1, 1, 1, 0, 0]) list([1, 0, 0]) list([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])]
                        # print(f"[ICM-DEBUG] pair_map sample: {pair_map[:10]} (showing first 5 samples)", flush=True) # [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4)]
                        

                    if self.config.algorithm.use_kl_in_reward:
                        # We need these metrics for apply_kl_penalty if using kl in reward
                        new_batch = self.compute_kl_related_metrics(new_batch, metrics, timing_raw)
                        # otherwise, we will compute those after dynamic sampling

                    with marked_timer("reward", timing_raw, "yellow"):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm and "rm_scores" not in new_batch.batch.keys():
                            # we first compute reward model score
                            batch_reward = self._compute_reward_colocate(new_batch)
                            new_batch = new_batch.union(batch_reward)

                        # we combine with rule-based rm
                        reward_tensor, reward_extra_infos_dict = extract_reward(new_batch)

                        new_batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update(
                                {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                            )

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            new_batch, kl_metrics = apply_kl_penalty(
                                new_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(
                                kl_metrics
                            )  # TODO: This will be cleared if we use multiple genenration batches
                        else:
                            new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]

                    # intrinsic reward을 token_level_rewards에 더하기
                    check_tensor = new_batch.batch["token_level_rewards"][0]
                    nonzero_indices = check_tensor.nonzero(as_tuple=True)[0]
                    print(f"[ICM-DEBUG] nonzero positions: {nonzero_indices.tolist()}")
                    print(f"[ICM-DEBUG] nonzero values: {check_tensor[nonzero_indices].tolist()}")
                    if self.config.algorithm.use_icm:
                        new_batch = self.add_intrinsic_reward(new_batch, intrinsic_reward, pair_map, step_boundary_list)
                    nonzero_indices = new_batch.batch["token_level_rewards"][0].nonzero(as_tuple=True)[0]
                    print(f"[ICM-DEBUG] nonzero positions: {nonzero_indices.tolist()}")
                    print(f"[ICM-DEBUG] nonzero values: {new_batch.batch['token_level_rewards'][0][nonzero_indices].tolist()}")


                    if not self.config.algorithm.filter_groups.enable:
                        batch = new_batch
                    else:  # NOTE: When prompts after filtering is less than train batch size,
                        # we skip to the next generation batch
                        metric_name = self.config.algorithm.filter_groups.metric
                        if metric_name == "seq_final_reward":
                            # Turn to numpy for easier filtering
                            new_batch.non_tensor_batch["seq_final_reward"] = (
                                new_batch.batch["token_level_rewards"].sum(dim=-1).numpy()
                            )
                        elif metric_name == "seq_reward":
                            new_batch.non_tensor_batch["seq_reward"] = (
                                new_batch.batch["token_level_scores"].sum(dim=-1).numpy()
                            )

                        # Collect the sequence reward for each trajectory
                        prompt_uid2metric_vals = defaultdict(list)
                        for uid, metric_val in zip(
                            new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch[metric_name], strict=True
                        ):
                            prompt_uid2metric_vals[uid].append(metric_val)

                        prompt_uid2metric_std = {}
                        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                            prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)

                        kept_prompt_uids = [
                            uid
                            for uid, std in prompt_uid2metric_std.items()
                            if std > 0 or len(prompt_uid2metric_vals[uid]) == 1
                        ]
                        num_prompt_in_batch += len(kept_prompt_uids)

                        kept_traj_idxs = []
                        for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch["uid"]):
                            if traj_from_prompt_uid in kept_prompt_uids:
                                kept_traj_idxs.append(idx)

                        new_batch = new_batch[kept_traj_idxs]
                        batch = new_batch if batch is None else DataProto.concat([batch, new_batch])

                        prompt_bsz = self.config.data.train_batch_size
                        if num_prompt_in_batch < prompt_bsz:
                            print(f"{num_prompt_in_batch=} < {prompt_bsz=}")
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                print(f"{num_gen_batches=}. Keep generating...")
                                self.gen_steps += 1
                                is_last_step = self.global_steps >= self.total_training_steps
                                continue
                            else:
                                raise ValueError(
                                    f"{num_gen_batches=} >= {max_num_gen_batches=}."
                                    + " Generated too many. Please check if your data are too difficult."
                                    + " You could also try set max_num_gen_batches=0 to enable endless trials."
                                )
                        else:
                            # Align the batch
                            traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                            batch = batch[:traj_bsz]

                    # print(f"batch: {batch}", flush=True)
                    self.checkpoint_manager.sleep_replicas()

                    # === Updating ===
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    if not self.config.algorithm.use_kl_in_reward:
                        batch = self.compute_kl_related_metrics(batch, metrics, timing_raw)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, "cyan"):
                            values = self._compute_values(batch)
                            batch = batch.union(values)

                    # Compute rollout correction weights and off-policy metrics (inherited from RayPPOTrainer)
                    from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    if rollout_corr_config is not None and "rollout_log_probs" in batch.batch:
                        batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                        # IS and off-policy metrics already have rollout_corr/ prefix
                        metrics.update(is_metrics)

                    with marked_timer("adv", timing_raw, "brown"):
                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, "pink"):
                            critic_output = self._update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, "red"):
                            actor_output = self._update_actor(batch)

                        # Check if ESI/training plan is close to expiration
                        esi_close_to_expiration = should_save_ckpt_esi(
                            max_steps_duration=self.max_steps_duration,
                            redundant_time=self.config.trainer.esi_redundant_time,
                        )
                        if self.config.trainer.save_freq > 0 and (
                            is_last_step
                            or self.global_steps % self.config.trainer.save_freq == 0
                            or esi_close_to_expiration
                        ):
                            if esi_close_to_expiration:
                                print("Force saving checkpoint: ESI instance expiration approaching.")
                            with marked_timer("save_checkpoint", timing_raw, "green"):
                                self._save_checkpoint()

                        with marked_timer("update_weights", timing_raw, "red"):
                            self.checkpoint_manager.update_weights()
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if self.config.trainer.test_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                ):
                    with marked_timer("testing", timing_raw, "green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw.get("step", 0)
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = defaultdict(float)  # clear timing

                metrics["train/num_gen_batches"] = num_gen_batches
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
                self.gen_steps += 1
        # check if last step checkpint exists
        checkpoint_dir = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")
        if not os.path.exists(checkpoint_dir):
            # save last step checkpoint
            timing_raw = defaultdict(float)
            with marked_timer("save_checkpoint", timing_raw, "green"):
                self._save_checkpoint()
            metrics = {f"timing/{k}": v for k, v in timing_raw.items()}
            logger.log(data=metrics, step=self.global_steps)
