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

import asyncio
import inspect
from typing import Optional

from verl import DataProto
from verl.experimental.reward_loop.reward_manager import register
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager.prm_scorer import PRMScorer


@register("dapo")
class DAPORewardManager(RewardManagerBase):
    """DAPO Reward Manager."""

    # Class-level cache so the PRM model is loaded only once across all instances.
    _prm_scorer: Optional[PRMScorer] = None

    @classmethod
    def init_class(cls, config, tokenizer):
        super().init_class(config, tokenizer)
        reward_kwargs = config.reward.get("reward_kwargs", {})
        prm_model_path = reward_kwargs.get("prm_model_path", None)
        if prm_model_path:
            print(f"[DAPORewardManager] Loading PRM from {prm_model_path} (device_map=auto)")
            cls._prm_scorer = PRMScorer(model_path=prm_model_path)
            print("[DAPORewardManager] PRM loaded.")

    def __init__(self, config, tokenizer, compute_score, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer, compute_score)
        self.compute_score = compute_score or default_compute_score
        self.is_async_reward_score = inspect.iscoroutinefunction(self.compute_score)

        # DAPO Reward Config
        reward_kwargs = config.reward.get("reward_kwargs", {})
        overlong_buffer_cfg = reward_kwargs.get("overlong_buffer_cfg", None)
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = reward_kwargs.get("max_resp_len", None)
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer

        # Step separator string for parsing responses into reasoning steps.
        # Steps are split from the decoded response text by this separator.
        # Hydra/shell may pass escape sequences as literal strings (e.g. "\\n\\n").
        # Convert them to actual characters so splitting works correctly.
        _step_sep = reward_kwargs.get("step_sep", None)
        self.step_sep: str | None = _step_sep.encode().decode("unicode_escape") if _step_sep is not None else None

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, (
                f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"
            )
            assert self.max_resp_len >= self.overlong_buffer_cfg.len, (
                "max_resp_len must be larger than overlong_buffer.len"
            )
            assert not self.overlong_buffer_cfg.enable or self.overlong_buffer_cfg.len > 0, (
                "overlong_buffer.len must be positive when overlong penalty is enabled,"
                f"but got {self.overlong_buffer_cfg.len}."
                "To disable the overlong penalty, set overlong_buffer.enable = False"
            )

    def _parse_steps(self, response_text: str) -> list[str]:
        """Split decoded response text into reasoning steps by ``step_sep``.

        Returns:
            steps: List of step strings with surrounding whitespace stripped.
        """
        steps = []
        for part in response_text.split(self.step_sep):
            stripped = part.strip()
            if stripped:
                steps.append(stripped)
        return steps

    async def _run_prm_background(
        self,
        problem_str: str,
        steps: list[str],
    ) -> None:
        """Run PRM scoring in the background (fire-and-forget).

        Does NOT feed back into training reward yet.
        labels[i] corresponds to steps[i] — alignment guaranteed by construction.
        TODO: multiply with CR scores element-wise: final[i] = labels[i] * cr[i]
        """
        try:
            assert self._prm_scorer is not None
            labels = await self.loop.run_in_executor(
                None,
                lambda: self._prm_scorer.score_steps(problem_str, steps),  # type: ignore[union-attr]
            )
            # labels: [1, 1, 0, 0, ...] — one per step, same order as steps
            print(f"[PRM-DEBUG] steps={len(steps)} labels={labels}")
        except Exception as exc:
            print(f"[PRM] scoring error: {exc}")

    async def run_single(self, data: DataProto) -> dict:
        assert len(data) == 1, "Only support single data item"
        data_item = data[0]

        prompt_ids = data_item.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]

        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        data_source = data_item.non_tensor_batch["data_source"]
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        extra_info = data_item.non_tensor_batch.get("extra_info", {})

        # Decode prompt and response to plain text (used for scoring and PRM).
        problem_str, response_str = await self.loop.run_in_executor(
            None,
            lambda: (
                self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True),
                self.tokenizer.decode(valid_response_ids, skip_special_tokens=True),
            ),
        )

        # Parse response text into steps if step_sep is configured.
        steps = self._parse_steps(response_str) if self.step_sep is not None else None

        # PRM step-level scoring runs in a separate background task so the
        # main DAPO reward flow is never blocked.  Results are printed to
        # stdout and do NOT feed back into training.
        if self._prm_scorer is not None and steps:
            asyncio.create_task(
                self._run_prm_background(
                    problem_str,
                    steps,
                )
            )
        extra_reward_kwargs = (
            {
                "reward_router_address": self.reward_router_address,
                "reward_model_tokenizer": self.reward_model_tokenizer,
            }
            if self.reward_router_address is not None
            else {}
        )
        if self.is_async_reward_score:
            result = await self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                **extra_reward_kwargs,
            )
        else:
            result = await self.loop.run_in_executor(
                None,
                lambda: self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    **extra_reward_kwargs,
                ),
            )

        reward_extra_info = {}

        score: float
        if isinstance(result, dict):
            score = result["score"]
            for key, value in result.items():
                reward_extra_info[key] = value
        else:
            score = result
            reward_extra_info["acc"] = score

        reward = score

        if self.overlong_buffer_cfg is not None and self.overlong_buffer_cfg.enable:
            overlong_buffer_len = self.overlong_buffer_cfg.len
            expected_len = self.max_resp_len - overlong_buffer_len
            exceed_len = valid_response_length - expected_len
            overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
            overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
            reward += overlong_reward
            if self.overlong_buffer_cfg.log:
                reward_extra_info["overlong_reward"] = overlong_reward
                reward_extra_info["overlong"] = overlong_reward < 0

        return {"reward_score": reward, "reward_extra_info": reward_extra_info}
