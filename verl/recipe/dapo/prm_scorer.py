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

"""Process Reward Model scorer using Qwen2.5-Math-PRM-7B.

Qwen2.5-Math-PRM-7B is a native PRM (Qwen2ForProcessRewardModel) that outputs
a 2-class logit per token position (logits shape: batch x seq_len x 2).
Step-level scores are the softmax probability of class 1 (correct) at the
token positions that decode to end with the step separator (\\n\\n).

Step boundaries are located by decoding each token individually and checking
whether the decoded string ends with "\\n\\n" — this avoids context-sensitive
BPE tokenization issues with standalone \\n\\n encoding.

Output format: [1, 1, ..., 1, 0, 0, ..., 0]
  - Steps before the first error get 1  (prob[1] > THRESHOLD).
  - The first erroneous step and all subsequent steps get 0.

Usage:
    scorer = PRMScorer(model_path="Qwen/Qwen2.5-Math-PRM-7B")
    labels = scorer.score_steps(
        problem_text="Solve x^2 - 5x + 6 = 0",
        steps=["Step 1: factor ...", "Step 2: x = 2 or x = 3"],
    )
    # e.g. [1, 0]
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer


_SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

# Threshold for softmax probability of class 1 (correct): prob > THRESHOLD → step is correct.
_SCORE_THRESHOLD = 0.8


class PRMScorer:
    """Step-level process reward scorer backed by Qwen2.5-Math-PRM-7B.

    Single forward pass over the full solution. Step scores are extracted at
    the token positions that decode to end with the step separator (\\n\\n).
    Scores are the softmax probability of class 1 (correct).

    Args:
        model_path: HuggingFace model ID or local path.
        dtype: Weight precision.
        max_length: Maximum token length (truncated on the left if exceeded).
    """

    STEP_SEP = "\n\n"

    def __init__(
        self,
        model_path: str,
        dtype: torch.dtype = torch.bfloat16,
        max_length: int = 4096,
    ) -> None:
        from transformers import AutoModel

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto",
        ).eval()

        self.device = next(self.model.parameters()).device
        self.max_length = max_length

    def _build_input_ids(self, problem_text: str, steps: list[str]) -> torch.Tensor:
        """Build input_ids via chat template.

        Each step is followed by STEP_SEP so that there are exactly N
        separators for N steps — one per step, not N-1.
        """
        solution = "".join(s + self.STEP_SEP for s in steps)
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": problem_text},
            {"role": "assistant", "content": solution},
        ]
        input_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False, return_tensors="pt"
        )
        if hasattr(input_ids, "input_ids"):
            input_ids = input_ids.input_ids
        return input_ids

    def _find_step_positions(self, input_ids: torch.Tensor) -> list[int]:
        """Find token positions that decode to end with \\n\\n.

        Decodes each token individually and checks endswith("\\n\\n").
        This is reliable even when \\n\\n is context-sensitively tokenized.
        """
        positions: list[int] = []
        for i in range(input_ids.shape[1]):
            token_str = self.tokenizer.decode([input_ids[0, i].item()])
            if token_str.endswith("\n\n"):
                positions.append(i)
        return positions

    @torch.no_grad()
    def score_steps(self, problem_text: str, steps: list[str]) -> list[int]:
        """Score each reasoning step and return a binary label list.

        Args:
            problem_text: The math problem as plain text.
            steps: List of step strings (separator already stripped).

        Returns:
            ``[1, 1, ..., 1, 0, 0, ..., 0]`` — one label per step.
        """
        if not steps:
            return []

        input_ids = self._build_input_ids(problem_text, steps)
        # Truncate on the left if over max_length.
        if input_ids.shape[1] > self.max_length:
            input_ids = input_ids[:, -self.max_length:]
        input_ids = input_ids.to(self.device)

        outputs = self.model(input_ids=input_ids, use_cache=False)
        # Qwen2ForProcessRewardModel: logits shape (1, seq_len, 2)
        # dim=-1 index 0 = bad, index 1 = good
        logits = outputs.logits  # (1, seq_len, 2)

        # Find \n\n token positions via per-token decode.
        all_positions = self._find_step_positions(input_ids)

        # Take only the last num_steps positions (assistant turn separators).
        step_positions = all_positions[-len(steps):] if len(all_positions) >= len(steps) else all_positions

        # Get softmax prob of class 1 (correct) at each step position.
        step_scores = []
        for pos in step_positions:
            probs = F.softmax(logits[0, pos], dim=-1)
            step_scores.append(probs[1].item())

        # Pad missing steps with a failing score.
        while len(step_scores) < len(steps):
            step_scores.append(0.0)

        # prob > THRESHOLD → correct (1), else wrong (0).
        raw_labels = [1 if s > _SCORE_THRESHOLD else 0 for s in step_scores]

        # Enforce [1, 1, ..., 1, 0, ..., 0]: once an error is found,
        # all subsequent steps are labelled 0.
        binary: list[int] = []
        found_error = False
        for label in raw_labels:
            if found_error or label == 0:
                binary.append(0)
                found_error = True
            else:
                binary.append(1)

        return binary

    @torch.no_grad()
    def score_steps_soft(self, problem_text: str, steps: list[str]) -> list[float]:
        """Return raw softmax probabilities (class 1) for each step — no thresholding.

        e.g. [0.9, 0.8, 0.3, 0.9, 0.7]
        """
        if not steps:
            return []

        input_ids = self._build_input_ids(problem_text, steps)
        if input_ids.shape[1] > self.max_length:
            input_ids = input_ids[:, -self.max_length:]
        input_ids = input_ids.to(self.device)

        outputs = self.model(input_ids=input_ids, use_cache=False)
        logits = outputs.logits  # (1, seq_len, 2)

        all_positions = self._find_step_positions(input_ids)
        step_positions = all_positions[-len(steps):] if len(all_positions) >= len(steps) else all_positions

        step_scores: list[float] = []
        for pos in step_positions:
            probs = F.softmax(logits[0, pos], dim=-1)
            step_scores.append(probs[1].item())

        while len(step_scores) < len(steps):
            step_scores.append(0.0)

        return step_scores
