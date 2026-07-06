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
import wandb

class RayDAPOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        reward_kwargs = OmegaConf.to_container(self.config.reward.reward_kwargs, resolve=True)
        self._use_curiosity = self.config.algorithm.get("use_curiosity", False)    ##SOO
        self._step_sep = self.config.actor_rollout_ref.get("step_sep", None)       ##SOO
        self.save_curiosity_scores = self.config.trainer.get("save_curiosity_scores", False)
        self._use_tokenlevel_curiosity = self.config.algorithm.get("use_tokenlevel_curiosity", False)  #SOO: token level ICM
        self._use_answerlevel_curiosity = self.config.algorithm.get("use_answerlevel_curiosity", False)  #SOO: answer level ICM
        print(f"[INIT-DEBUG] use_curiosity: {self._use_curiosity}", flush=True)
        print(f"[INIT-DEBUG] step_sep: {repr(self._step_sep)}", flush=True)
        print(f"[INIT-DEBUG] save_curiosity_scores: {self.save_curiosity_scores}", flush=True)
        print(f"[INIT-DEBUG] use_tokenlevel_curiosity: {self._use_tokenlevel_curiosity}", flush=True)
        print(f"[INIT-DEBUG] use_answerlevel_curiosity: {self._use_answerlevel_curiosity}", flush=True)

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
            if "ref_log_prob" not in batch.batch:
                ref_log_prob = self._compute_ref_log_prob(batch)
                batch = batch.union(ref_log_prob)
            else:
                print("ref_log_prob already in batch, skipping computation", flush=True)

        return batch
    
    def add_intrinsic_reward(self, batch, curiosity_result):
        """
        batch: DataProto
        curiosity_result: DataProto, non_tensor_batch에 "intrinsic_reward", "pair_map", "step_boundaries" 포함

        batch에 token_level_intrinsic_rewards 추가하기
        """
        prm_labels = curiosity_result.non_tensor_batch["prm_labels"]
        intrinsic_reward = curiosity_result.non_tensor_batch["intrinsic_reward"]
        pair_map = curiosity_result.non_tensor_batch["pair_map"]
        step_boundary_list = curiosity_result.non_tensor_batch["step_boundaries"]
        intrinsic_reward_flat = [reward for rewards in intrinsic_reward for reward in rewards]
        pair_map_flat = [pair for pairs in pair_map for pair in pairs]

        # prm_total = sum(len(prm_labels[i]) for i in range(len(prm_labels)))
        # step_total = sum(len(step_boundary_list[i]) - 1 for i in range(len(step_boundary_list)))
        # print(f"pair_map is sorted: {all(pair_map[i][0] <= pair_map[i+1][0] for i in range(len(pair_map)-1))}", flush=True)
        # print(f"pair_map sample: {pair_map[:10]}", flush=True)
        # print(f"pair_map len: {len(pair_map_flat)}, intrinsic_reward len: {len(intrinsic_reward_flat)}, prm_labels len: {prm_total}, step_boundaries len: {step_total}", flush=True)
        # pair_map len: 1557, intrinsic_reward len: 1557, prm_labels len: 1557, step_boundaries len: 1557

        entries = []  # (data_idx, start, end, prm_label, raw)
        raw_vals = []

        for k, (data_idx, step_idx) in enumerate(pair_map_flat):
            boundary = step_boundary_list[data_idx]
            start = boundary[step_idx]
            end = boundary[step_idx + 1]
            prm_label = prm_labels[data_idx][step_idx]
            raw = intrinsic_reward_flat[k].item()
            raw_vals.append(raw)
            entries.append((data_idx, start, end, prm_label))

        scaled_vals = []
        if raw_vals:
            raw_arr = np.array(raw_vals)
            eta = self.config.actor_rollout_ref.icm.eta
            reward_token = self.config.actor_rollout_ref.icm.intrinsic_reward_token
            icm_calculation = self.config.actor_rollout_ref.icm.icm_calculation

            r_min, r_max = raw_arr.min(), raw_arr.max()

            if icm_calculation == "whiten":
                mean = raw_arr.mean()
                var = raw_arr.var()
                whitened_arr = (raw_arr - mean) / np.sqrt(var + 1e-8)

            if icm_calculation == "whiten_prm":
                prm_raw_vals_w = [raw for (_, _, _, prm_label), raw in zip(entries, raw_vals) if prm_label == 1]
                if prm_raw_vals_w:
                    prm_arr_w = np.array(prm_raw_vals_w)
                    prm_mean = prm_arr_w.mean()
                    prm_var = prm_arr_w.var()
                else:
                    prm_mean = raw_arr.mean()
                    prm_var = raw_arr.var()
                whitened_prm_arr = (raw_arr - prm_mean) / np.sqrt(prm_var + 1e-8)

            if icm_calculation == "normalize_prm":
                prm_raw_vals = [raw for (_, _, _, prm_label), raw in zip(entries, raw_vals) if prm_label == 1]
                if prm_raw_vals:
                    prm_arr = np.array(prm_raw_vals)
                    prm_r_min, prm_r_max = prm_arr.min(), prm_arr.max()
                else:
                    prm_r_min, prm_r_max = r_min, r_max

            for k, ((data_idx, start, end, prm_label), raw) in enumerate(zip(entries, raw_vals)):
                #SOO: clip, normalize, whiten, or normalize_prm of icm.
                if icm_calculation == "clip":
                    scaled = float(eta * np.clip(raw, 0.0, 1.0))
                elif icm_calculation == "normalize":
                    normalized = (raw - r_min) / (r_max - r_min + 1e-8)
                    scaled = float(eta * normalized)
                elif icm_calculation == "normalize_prm":
                    # normalize using min/max computed only over prm==1 entries
                    normalized = (raw - prm_r_min) / (prm_r_max - prm_r_min + 1e-8)
                    scaled = float(eta * normalized)
                elif icm_calculation == "whiten_prm":
                    # whiten using mean/var computed only over prm==1 entries
                    scaled = float(eta * whitened_prm_arr[k])
                else:  # whiten
                    scaled = float(eta * whitened_arr[k])
                scaled_vals.append(scaled)

                if reward_token == "all_step_tokens":
                    batch.batch["token_level_rewards"][data_idx, start:end] += scaled * prm_label
                elif reward_token == "last_step_token":
                    n_steps = sum(prm_labels[data_idx])
                    if n_steps > 0:
                        batch.batch["token_level_rewards"][data_idx, end - 1] += scaled * prm_label / n_steps

        if raw_vals:
            import wandb
            scaled_arr = np.array(scaled_vals)
            wandb.log({
                "icm/intrinsic_reward_raw/mean": raw_arr.mean(),
                "icm/intrinsic_reward_raw/max": raw_arr.max(),
                "icm/intrinsic_reward_raw/min": raw_arr.min(),
                "icm/intrinsic_reward_scaled/mean": scaled_arr.mean(),
                "icm/intrinsic_reward_scaled/max": scaled_arr.max(),
                "icm/intrinsic_reward_scaled/min": scaled_arr.min(),
            }, step=self.global_steps)

        return batch

    def add_tokenlevel_intrinsic_reward(self, batch, tokenlevel_result):  #SOO: token level ICM
        """
        Adds eta_token * tokenlevel_intrinsic_rewards to token_level_rewards at every response token.
        Follows CD-RLHF: rewards[j, start:ends[j]] += eta * intrinsic_reward[j, :ends[j]-start]
        """
        intrinsic_rewards = tokenlevel_result.batch["tokenlevel_intrinsic_rewards"]  # (B, max_resp_len)
        eta_token = self.config.actor_rollout_ref.tokenlevel_icm.eta_token
        response_mask = batch.batch["response_mask"].float()  # (B, max_resp_len)

        batch.batch["token_level_rewards"] += eta_token * intrinsic_rewards * response_mask

        valid = response_mask.bool()
        wandb.log({
            "tokenlevel_icm/intrinsic_reward/mean": intrinsic_rewards[valid].mean().item(),
            "tokenlevel_icm/intrinsic_reward/max":  intrinsic_rewards[valid].max().item(),
            "tokenlevel_icm/intrinsic_reward/min":  intrinsic_rewards[valid].min().item(),
        }, step=self.global_steps)
        return batch

    def add_answerlevel_intrinsic_reward(self, batch, answerlevel_result):  #SOO: answer level ICM
        """
        Adds eta_answer * whitened_icm_error to the last response token,
        only for sequences where the extrinsic reward is positive (correct answer).
        """
        intrinsic_rewards = answerlevel_result.batch["answerlevel_intrinsic_rewards"]  # (B,)
        eta_answer = self.config.actor_rollout_ref.answerlevel_icm.eta_answer

        # correctness mask: sequences with positive extrinsic reward
        seq_scores = batch.batch["token_level_scores"].sum(dim=-1)  # (B,)
        correct_mask = (seq_scores > 0).float()                      # (B,)

        # last valid response token position per sequence
        response_mask = batch.batch["response_mask"]                  # (B, max_resp_len)
        last_token_idx = response_mask.sum(dim=-1).long() - 1        # (B,)

        scaled = eta_answer * intrinsic_rewards * correct_mask        # (B,)
        for i in range(len(batch)):
            if scaled[i] != 0:
                batch.batch["token_level_rewards"][i, last_token_idx[i]] += scaled[i].item()

        wandb.log({
            "answerlevel_icm/intrinsic_reward/mean": intrinsic_rewards.mean().item(),
            "answerlevel_icm/intrinsic_reward/max":  intrinsic_rewards.max().item(),
            "answerlevel_icm/intrinsic_reward/min":  intrinsic_rewards.min().item(),
            "answerlevel_icm/correct_ratio": correct_mask.mean().item(),
        }, step=self.global_steps)
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

                    if "timing" in gen_batch_output.meta_info:
                        timing_raw.update(gen_batch_output.meta_info.pop("timing"))

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

                    # calculate curiosity score if needed
                    if self._use_curiosity:
                        with marked_timer("curiosity", timing_raw, "magenta"):
                            print(f"[CURIO-DEBUG] Computing curiosity scores", flush=True)
                            batch_size =len(batch.batch)
                            batch.meta_info["step_sep"] = self._step_sep
                            batch.non_tensor_batch["global_indices"] = np.arange(batch_size)
                            curiosity_result = self.actor_rollout_wg.compute_curiosity(batch)  

                        batch.batch["ref_log_prob"] = curiosity_result.batch["ref_log_prob"]
                        metrics["train/icm_loss"] = curiosity_result.non_tensor_batch["icm_loss"].mean().item()
                        all_prm_labels = curiosity_result.non_tensor_batch["prm_labels"]
                        flat_labels = [lbl for labels in all_prm_labels for lbl in labels]
                        if flat_labels:
                            metrics["train/prm_label_mean"] = float(np.mean(flat_labels))

                        # save curiosity scores for analysis
                        if self.save_curiosity_scores:
                            rows = []
                            for i in range(batch_size):
                                uid = batch.non_tensor_batch["uid"][i]
                                global_idx = batch.non_tensor_batch["global_indices"][i]
                                parsed_steps = curiosity_result.non_tensor_batch["parsed_steps"][i]
                                rewards = curiosity_result.non_tensor_batch["intrinsic_reward"][i]  # array of floats
                                prm_labels = curiosity_result.non_tensor_batch["prm_labels"][i]     # array of ints
                                pair_map = curiosity_result.non_tensor_batch["pair_map"][i]         # list of (data_idx, step_idx)
                                step_boundaries = curiosity_result.non_tensor_batch["step_boundaries"][i]  # list of ints

                                # print(f"[DEBUG] i={i} parsed_steps len={len(parsed_steps)} rewards len={len(rewards)}", flush=True)
                                # print(f"[DEBUG] i={i} parsed_steps[0]={repr(parsed_steps[0]) if len(parsed_steps) > 0 else 'EMPTY'}", flush=True)

                                for step_idx, (step_text, reward, prm_label, (data_idx, s_idx)) in enumerate(zip(parsed_steps, rewards, prm_labels, pair_map)):
                                    rows.append([
                                        uid,
                                        int(global_idx),
                                        step_idx,
                                        step_text,
                                        float(reward),
                                        int(prm_label),
                                        int(step_boundaries[step_idx]),   # step 시작 토큰 위치
                                        int(step_boundaries[step_idx + 1]) # step 끝 토큰 위치
                                    ])
                                
                            table = wandb.Table(
                                columns=["uid", "global_index", "step_idx", "step_text", "curiosity_score", "prm_label", "step_start", "step_end"],
                                data=rows
                            )
                            wandb.log({"curiosity/scores": table}, step=self.global_steps)

                        # intrinsic reward을 token_level_rewards에 더하기
                        # check_tensor = batch.batch["token_level_rewards"][0]
                        # nonzero_indices = check_tensor.nonzero(as_tuple=True)[0]
                        # print(f"[ICM-DEBUG] nonzero positions: {nonzero_indices.tolist()}")
                        # print(f"[ICM-DEBUG] nonzero values: {check_tensor[nonzero_indices].tolist()}")
                        batch = self.add_intrinsic_reward(batch, curiosity_result)
                        # nonzero_indices = batch.batch["token_level_rewards"][0].nonzero(as_tuple=True)[0]
                        # print(f"[ICM-DEBUG] nonzero positions: {nonzero_indices.tolist()}")
                        # print(f"[ICM-DEBUG] nonzero values: {batch.batch['token_level_rewards'][0][nonzero_indices].tolist()}")

                    # token-level ICM (independent from step-level ICM)  #SOO: token level ICM
                    if self._use_tokenlevel_curiosity:
                        with marked_timer("tokenlevel_curiosity", timing_raw, "magenta"):
                            print(f"[CURIO-DEBUG] Computing token-level curiosity scores", flush=True)
                            tokenlevel_result = self.actor_rollout_wg.compute_tokenlevel_curiosity(batch)
                        metrics["train/tokenlevel_icm_loss"] = tokenlevel_result.non_tensor_batch["tokenlevel_icm_loss"].mean().item()
                        batch = self.add_tokenlevel_intrinsic_reward(batch, tokenlevel_result)

                    # answer-level ICM (only applied on correct answers)  #SOO: answer level ICM
                    if self._use_answerlevel_curiosity:
                        with marked_timer("answerlevel_curiosity", timing_raw, "magenta"):
                            print(f"[CURIO-DEBUG] Computing answer-level curiosity scores", flush=True)
                            answerlevel_result = self.actor_rollout_wg.compute_answerlevel_curiosity(batch)
                        metrics["train/answerlevel_icm_loss"] = answerlevel_result.non_tensor_batch["answerlevel_icm_loss"].mean().item()
                        batch = self.add_answerlevel_intrinsic_reward(batch, answerlevel_result)

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

                        if (
                            self.config.algorithm.get("overload_actor_to_ref", False)
                            and self.config.algorithm.overload_actor_to_ref_freq > 0
                            and self.global_steps % self.config.algorithm.overload_actor_to_ref_freq == 0
                        ):
                            with marked_timer("sync_ref_from_actor", timing_raw, "cyan"):
                                self.actor_rollout_wg.sync_ref_from_actor()
                            print(f"[OVERLOAD] Synced actor → ref at step {self.global_steps}", flush=True)

                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                for key, val in timing_raw.items():
                    print(f"[TIMING] {key}: {val:.3f}s", flush=True)

                # validate
                if self.config.trainer.test_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                ):
                    with marked_timer("testing", timing_raw, "green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # n=16 validation every 10 steps for pass@16 metrics
                if self.global_steps % 10 == 0:
                    with marked_timer("testing_n16", timing_raw, "green"):
                        val_metrics_n16: dict = self._validate(n_val=16)
                    metrics.update(val_metrics_n16)

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
