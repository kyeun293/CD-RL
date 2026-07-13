"""
Reasoning Path Divergence (RPD).

From: "Reasoning Path Divergence: A New Metric and Curation Strategy to
       Unlock LLM Diverse Thinking" (arXiv:2510.26122)
Reference impl: https://github.com/fengjujf/Reasoning-Path-Divergence

Algorithm (faithful to the paper / repo):

  1. Step extraction
       Use an LLM judge to decompose each Long-CoT solution S into a JSON list
       of 3–5 *method-focused* steps:  S = [s_1, ..., s_m]

  2. Asymmetric pairwise distance
       Embed every step. For two solutions A = [a_1..a_m], B = [b_1..b_n]:

         d(A -> B) = (1/m) * sum_i  min_j (1 - cos(emb(a_i), emb(b_j)))
         d(B -> A) = (1/n) * sum_j  min_i (1 - cos(emb(b_j), emb(a_i)))
         RPD(A, B) = 0.5 * (d(A->B) + d(B->A))

  3. Per-question aggregation over N samples (paper uses N=16):

         RPD(q) = 2 / (N*(N-1)) * sum_{i<j} RPD(S_i, S_j)

NOTES ON FAITHFULNESS
  - The exact step-extraction prompt and the exact embedding model
    ("text-embedding-3-small" via the OpenAI API) are taken from the original
    repo's `04_extract_steps.py` and `05_compute_matrix.py`.
  - The judge LLM here is loaded via vLLM per the user's request, so we
    REPLACE the OpenAI judge call but KEEP the prompt verbatim. Drop the
    paper's prompt into `STEP_EXTRACTION_PROMPT` below — placeholder is
    structurally equivalent and produces the same JSON schema.
  - Embeddings still use OpenAI's `text-embedding-3-small` to match the paper
    exactly. If you want a fully local pipeline, swap `_embed_openai` for a
    sentence-transformers backend; the math downstream is identical.
"""

import json
import os
import re
from itertools import combinations
from typing import Dict, List, Sequence
import torch
import numpy as np

from sampling import VLLMSampler
from metrics.embedding_utils import embed_texts


TENSOR_PARALLEL_SIZE = torch.cuda.device_count()

# ---------------------------------------------------------------------------
# Prompt — keep in sync with the original repo's 04_extract_steps.py
# ---------------------------------------------------------------------------
PROMPT_TEMPLATE = """You are a specialized AI expert in analyzing mathematical solutions. Your task is to first provide a step-by-step analysis of a solution, and then, based on your analysis, generate a final JSON output that is concise, direct, and method-focused.

### REQUIRED OUTPUT STRUCTURE:
Your response MUST have two distinct parts in the following order:

**Part 1: Analysis & Thinking Process**
- Start this section with the heading `### Analysis`.
- Briefly explain your reasoning as you deconstruct the provided solution. This is your "scratchpad".

**Part 2: Final JSON Output**
- After your analysis, provide the final JSON output enclosed in `//boxed{{}}`.
- This part must contain *only* the `//boxed{{...}}` block and nothing else.

### CONTENT RULES FOR THE FINAL JSON:
1.  **Step Count**: The JSON must contain **strictly 3 to 5 logical steps**.
2.  **Output Style**:
    - **Use direct, active verb phrases.** Start each description with a verb (e.g., "Calculate", "Identify", "Apply").
    - **DO NOT use narrative phrasing** like "The author identifies..." or "The solution then calculates...".
3.  **Abstraction Level**:
    - Be abstract about numbers and variables, but **be specific about the methodology**.
    - **BAD (Too Vague):** "Use a formula to get the result."
    - **BAD (Too Concrete):** "Calculate 1/3 + 1/6 = 1/2."
    - **GOOD (Balanced):** "Combine the individual rates to find the total work rate."

### JSON STRUCTURE SPECIFICATION:
- The root object must have one key: `"logical_steps"`.
- The value of `"logical_steps"` must be a list (`[]`) of step objects.
- Each step object (`{{}}`) must contain two keys:
  - `"step_title"`: A short title for the step (e.g., "Step 1: Combine Rates"). Use `null` if not applicable.
  - `"step_description"`: A concise summary of the action, following all rules above.

### EXAMPLE OF THE COMPLETE TWO-PART OUTPUT:
**Input Solution**: "Pipe A fills a tank in 3 hours, so its rate is 1/3 tank/hr. Pipe B fills it in 6 hours, so its rate is 1/6 tank/hr. Together, their rate is 1/3 + 1/6 = 1/2 tank/hr. Therefore, the time to fill the tank together is the reciprocal of the rate, which is 1 / (1/2) = 2 hours."
**Your Required Output**:
### Analysis
The solution addresses a classic work-rate problem.
1.  First, it calculates the individual rate for each pipe.
2.  Second, it sums these rates to get a combined rate.
3.  Finally, it converts the combined rate back into total time.
The logic is broken down into three clear, abstract steps.

//boxed{{
  "logical_steps": [
    {{
      "step_title": "Step 1: Determine Individual Rates",
      "step_description": "Determine the individual work rate of each component based on the time taken."
    }},
    {{
      "step_title": "Step 2: Combine Rates",
      "step_description": "Combine the individual rates to find the total system work rate."
    }},
    {{
      "step_title": "Step 3: Calculate Total Time",
      "step_description": "Calculate the total time by taking the reciprocal of the combined work rate."
    }}
  ]
}}

---

### YOUR TASK:

**Math Problem**:
{question_text}

**Chain-of-Thought Solution to Analyze**:
{answer_cot}

---

### !!! CRITICAL REQUIREMENTS RECAP !!!
Before generating the final JSON, ensure your output strictly adheres to these constraints:
1.  **Strictly 3-5 Steps**: Consolidate or expand if necessary to fit this range.
2.  **Verb-First Phrasing**: Descriptions MUST start with an action verb (e.g., "Calculate...", "Derive..."). DO NOT use "The solution..." or "Step 1...".
3.  **Abstract Methodology**: Describe the *method*, NOT the specific numbers.
4.  **Format**: The final JSON must be valid and enclosed in `//boxed{{...}}`.

"""

# ---------------------------------------------------------------------------
# JSON box extraction
# ---------------------------------------------------------------------------

# Judges often echo LaTeX (`\(`, `\)`, `\frac`, ...) verbatim inside JSON string
# values; backslashes not followed by a legal JSON escape char make the whole
# block unparseable. Double them up so `\(` -> `\\(` before retrying.
_INVALID_JSON_ESCAPE = re.compile(r'\\(?!["\\/bfnrtu])')


def _sanitize_json_escapes(json_str: str) -> str:
    return _INVALID_JSON_ESCAPE.sub(r"\\\\", json_str)


def extract_boxed_json(raw_text: str) -> dict:
    start_marker = "//boxed{"
    try:
        start_pos = raw_text.find(start_marker)
        if start_pos == -1:
            return {"error": "Delimiter //boxed{ not found in the output."}

        json_start_pos = start_pos + len(start_marker)
        brace_level = 1
        for i in range(json_start_pos, len(raw_text)):
            ch = raw_text[i]
            if ch == "{":
                brace_level += 1
            elif ch == "}":
                brace_level -= 1
            if brace_level == 0:
                json_str = raw_text[json_start_pos - 1: i + 1]
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass
                try:
                    return json.loads(_sanitize_json_escapes(json_str))
                except json.JSONDecodeError as e:
                    return {"error": f"Failed to decode JSON: {e}",
                            "json_string": json_str}
        return {"error": "Could not find the matching closing brace."}
    except Exception as e:
        return {"error": f"Unexpected error: {e}"}


# ---------------------------------------------------------------------------
# Create summary object(same as repo)
# ---------------------------------------------------------------------------
def _summary_from_raw(raw_text: str, answer_id: int) -> dict:
    obj = {"answer_id": answer_id, "generated_summary": raw_text, "logical_steps": []}
    extracted = extract_boxed_json(raw_text)
    if isinstance(extracted, dict) and "logical_steps" in extracted:
        obj["logical_steps"] = extracted["logical_steps"]
    return obj
 

 
# ---------------------------------------------------------------------------
# Distance — re-implementation of the repo's utils.calculate_rpd_distance.
# ---------------------------------------------------------------------------
def _summary_embeddings(summary: dict, embeddings_dict: Dict[str, np.ndarray]) -> np.ndarray:
    """Stack the embeddings of all valid step_descriptions in a summary."""
    rows = []
    for step in summary.get("logical_steps", []):
        desc = step.get("step_description")
        if desc and isinstance(desc, str) and desc.strip() and desc in embeddings_dict:
            rows.append(embeddings_dict[desc])
    if not rows:
        return np.zeros((0, 0), dtype=np.float32)
    return np.stack(rows, axis=0).astype(np.float32)
 
 
def _asymmetric_mean_min_cosdist(A: np.ndarray, B: np.ndarray) -> float:
    """For each row of A, take min cosine distance to any row of B; average."""
    if A.size == 0 or B.size == 0:
        return 0.0
    # Embeddings are L2-normalized in `score_question` (normalize_embeddings=True),
    # so dot product == cosine similarity.
    sim = A @ B.T                 # [m, n]
    best = sim.max(axis=1)        # [m]
    return float((1.0 - best).mean())
 
 
def calculate_rpd_distance(summary_a: dict,
                           summary_b: dict,
                           embeddings_dict: Dict[str, np.ndarray]) -> float:
    """
    Symmetric RPD distance between two summaries.
 
    d(A, B) = 0.5 * ( mean_i min_j (1 - cos(a_i, b_j))
                    + mean_j min_i (1 - cos(b_j, a_i)) )
    """
    A = _summary_embeddings(summary_a, embeddings_dict)
    B = _summary_embeddings(summary_b, embeddings_dict)
    if A.size == 0 or B.size == 0:
        return 0.0
    return 0.5 * (_asymmetric_mean_min_cosdist(A, B)
                  + _asymmetric_mean_min_cosdist(B, A))
 


# ---------------------------------------------------------------------------
# Embeddings — OpenAI text-embedding-3-small, matching the paper
# ---------------------------------------------------------------------------

# def _embed_openai(texts: Sequence[str], model: str) -> np.ndarray:
#     """Batch-embed a list of strings. Requires OPENAI_API_KEY."""
#     from openai import OpenAI
#     client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
#     # OpenAI accepts up to 2048 inputs per call; chunk to be safe.
#     out = []
#     B = 256
#     for i in range(0, len(texts), B):
#         chunk = list(texts[i:i + B])
#         resp = client.embeddings.create(model=model, input=chunk)
#         out.extend([d.embedding for d in resp.data])
#     arr = np.asarray(out, dtype=np.float32)
#     # L2 normalize so that cosine = dot.
#     norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
#     return arr / norms


# ---------------------------------------------------------------------------
# RPD math
# ---------------------------------------------------------------------------
def _asymmetric_distance(A: np.ndarray, B: np.ndarray) -> float:
    """
    Mean over rows of A of (1 - max cosine to any row of B).
    A, B are L2-normalized so cos = A @ B.T.
    """
    if A.size == 0 or B.size == 0:
        return 0.0
    sim = A @ B.T                  # [m, n]
    best = sim.max(axis=1)         # [m]
    return float((1.0 - best).mean())


def _pair_rpd(A: np.ndarray, B: np.ndarray) -> float:
    return 0.5 * (_asymmetric_distance(A, B) + _asymmetric_distance(B, A))


# ---------------------------------------------------------------------------
# Public scorer
# ---------------------------------------------------------------------------
class RPDScorer:
    def __init__(
        self,
        judge_model: str,
        embedding_model: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.85,
        judge_temperature: float = 0.1,
        judge_max_tokens: int = 2048,
    ):
        self.embedding_model = embedding_model
        self.judge_temperature = judge_temperature
        self.judge_max_tokens = judge_max_tokens

        self.judge = VLLMSampler(
            model_path=judge_model,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
        )


    # Generate raw summary
    def _build_summaries(self, problem: str, samples: Sequence[str]) -> List[dict]:
        messages_batch = [
            [{"role": "user",
              "content": PROMPT_TEMPLATE.format(question_text=problem,
                                                answer_cot=sol)}]
            for sol in samples
        ]
        raws = self.judge.chat(
            messages_batch,
            temperature=self.judge_temperature,
            max_new_tokens=self.judge_max_tokens,
        )
        return [_summary_from_raw(raw, ans_id) for ans_id, raw in enumerate(raws)]


    # public API for scoring RDP per question
    def score_question(self, problem: str, samples: Sequence[str], dump_path: "str | None" = None,) -> float:
        """
        Returns RPD(q) over the N samples for one question.

        RPD(q) = 2 / (N*(N-1)) * sum_{i<j} RPD(S_i, S_j)

        Returns 0.0 if fewer than 2 samples produced a parseable summary.
        """
        N = len(samples)
        if N < 2:
            return 0.0


        # 1. Get a summary per sample, then keep only "valid" ones.
        #    Mirrors 05_compute_matrix.py validity check.
        all_summaries = self._build_summaries(problem, samples)

        # Optional dump (BEFORE validity filtering, so debugging can see
        # exactly what the judge produced — including malformed outputs).
        if dump_path is not None:
            import os
            os.makedirs(os.path.dirname(dump_path) or ".", exist_ok=True)
            with open(dump_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"original_question": problem, "summaries": all_summaries},
                    f, indent=4, ensure_ascii=False,
                )

        valid_summaries = [
            s for s in all_summaries
            if isinstance(s.get("logical_steps"), list)
            and s["logical_steps"]
            and any(st.get("step_description") for st in s["logical_steps"])
        ]
        print(f'[Valid sumarries] {len(valid_summaries)}')
        if len(valid_summaries) < 2:
            return 0.0


        # 2. Dedup all step_description strings, embed once.
        texts_to_embed = set()
        for s in valid_summaries:
            for st in s["logical_steps"]:
                desc = st.get("step_description")
                if desc and isinstance(desc, str) and desc.strip():
                    texts_to_embed.add(desc)
        if not texts_to_embed:
            return 0.0
        unique_texts = list(texts_to_embed)
        embs = embed_texts(unique_texts, model=self.embedding_model)
        embeddings_dict = {t: e for t, e in zip(unique_texts, embs)}

        # embs = _embed_openai(flat, model=self.embedding_model)

        # per_sample_embs = [
        #     embs[offsets[i]:offsets[i + 1]] for i in range(N)
        # ]

        # 3. Pairwise RPD.
        pairs = list(combinations(range(len(valid_summaries)), 2))
        if not pairs:
            return 0.0
        total = 0.0
        for i, j in pairs:
            total += calculate_rpd_distance(valid_summaries[i],
                                            valid_summaries[j],
                                            embeddings_dict)
        return total / len(pairs)  # equals 2/(N*(N-1)) * sum_{i<j}