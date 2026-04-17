"""
Diversity evaluation for verl-trained checkpoints.

Pipeline:
  1. Load HF checkpoint (verl saves merged HF format under `huggingface/`).
  2. Sample N (=16) responses per AIME problem with vLLM.
  3. Compute:
       - Distinct Equations  (Diversity-Aware Policy Optimization, 2505.23433)
       - Reasoning Path Divergence (RPD, 2510.26122)
  4. Dump per-question scores + aggregate to JSON, optionally log to wandb.

Usage:
  python run_eval.py \
      --model-path /path/to/verl/checkpoint/global_step_500/huggingface \
      --judge-model Qwen/Qwen2.5-7B-Instruct \
      --output-dir ./outputs/step500 \
      --n-samples 16
"""

import argparse
import json
import os
from pathlib import Path

from data.aime_loader import load_aime
from sampling import VLLMSampler
from metrics.distinct_equations import distinct_equations_score
from metrics.rpd import RPDScorer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True,
                   help="HF-format checkpoint dir (verl exports under .../huggingface/).")
    p.add_argument("--judge-model", default=None,
                   help="vLLM model used for RPD step extraction. "
                        "Defaults to --model-path if not set.")
    p.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B",
                   help="Embedding model for RPD (matches the original repo).")
    p.add_argument("--dataset", default="aime2024", choices=["aime2024", "aime2025"])
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=8000)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--skip-rpd", action="store_true",
                   help="Only compute Distinct Equations (fast path).")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------- 1. Load data --------------------
    problems = load_aime(args.dataset)  # List[{"id", "problem", "answer"}]
    print(f"[data] loaded {len(problems)} problems from {args.dataset}")
    print(f"[1st problem] {problems[0]}")

    # -------------------- 2. Sample responses --------------------
    sampler = VLLMSampler(
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
    )
    rollouts = sampler.sample(
        prompts=[p["problem"] for p in problems],
        n=args.n_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
    )  # List[List[str]]  shape: [num_problems, n]

    # Persist raw rollouts so we can re-score without re-sampling.
    rollouts_path = out_dir / "rollouts.jsonl"
    with rollouts_path.open("w") as f:
        for prob, samples in zip(problems, rollouts):
            f.write(json.dumps({
                "id": prob["id"],
                "problem": prob["problem"],
                "answer": prob["answer"],
                "samples": samples,
            }) + "\n")
    print(f"[rollout] wrote {rollouts_path}")

    # Free the policy model before loading the judge (single-GPU friendly).
    del sampler

    # -------------------- 3a. Distinct Equations --------------------
    de_per_q = []
    for prob, samples in zip(problems, rollouts):
        score = distinct_equations_score(samples)
        de_per_q.append({"id": prob["id"], "distinct_eq": score})
    de_mean = sum(d["distinct_eq"] for d in de_per_q) / max(len(de_per_q), 1)
    print(f"[metric] Distinct Equations (mean over {len(de_per_q)} q): {de_mean:.4f}")

    # -------------------- 3b. RPD --------------------
    rpd_per_q = []
    rpd_mean = None
    if not args.skip_rpd:
        rpd_scorer = RPDScorer(
            judge_model=args.judge_model or args.model_path,
            embedding_model=args.embedding_model,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )

        # judge_model = args.judge_model or args.model_path
        # same_model = (judge_model == args.model_path)
 
        # if same_model:
        #     # Reuse the already-loaded vLLM instance — no CUDA re-init needed.
        #     print(f"[rpd] reusing policy sampler as judge (same model)")
        #     rpd_scorer = RPDScorer.__new__(RPDScorer)
        #     rpd_scorer.embedding_model = args.embedding_model
        #     rpd_scorer.judge_temperature = 0.1
        #     rpd_scorer.judge_max_tokens = 2048
        #     rpd_scorer.judge = sampler
        # else:
        #     # Different judge model: must fully release policy GPU first.
        #     print(f"[rpd] freeing policy model, loading judge: {judge_model}")
        #     del sampler
        #     import gc, torch
        #     gc.collect()
        #     if torch.cuda.is_available():
        #         torch.cuda.empty_cache()
        #         torch.cuda.synchronize()
        #     rpd_scorer = RPDScorer(
        #         judge_model=judge_model,
        #         embedding_model=args.embedding_model,
        #         tensor_parallel_size=args.tensor_parallel_size,
        #         gpu_memory_utilization=args.gpu_memory_utilization,
        #     )

        summary_dir = out_dir / "summaries"
        summary_dir.mkdir(parents=True, exist_ok=True)

        for prob, samples in zip(problems, rollouts):
            # Sanitize id for filenames (AIME ids contain '/').
            safe_id = str(prob["id"]).replace("/", "_").replace(" ", "_")
            dump_path = summary_dir / f"{safe_id}.json"

            score = rpd_scorer.score_question(problem=prob["problem"], samples=samples, dump_path=dump_path)
            rpd_per_q.append({"id": prob["id"], "rpd": score})
            print(f"  [rpd] q={prob['id']} score={score:.4f}")
        rpd_mean = sum(r["rpd"] for r in rpd_per_q) / max(len(rpd_per_q), 1)
        print(f"[metric] RPD (mean over {len(rpd_per_q)} q): {rpd_mean:.4f}")

    # -------------------- 4. Dump --------------------
    summary = {
        "model_path": args.model_path,
        "dataset": args.dataset,
        "n_samples": args.n_samples,
        "distinct_equations": {
            "mean": de_mean,
            "per_question": de_per_q,
        },
        "rpd": {
            "mean": rpd_mean,
            "per_question": rpd_per_q,
        } if not args.skip_rpd else None,
    }
    summary_path = out_dir / "eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[done] wrote {summary_path}")


if __name__ == "__main__":
    main()