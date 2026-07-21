"""
Diversity + accuracy evaluation for verl-trained checkpoints.

Pipeline:
  0. Resolve --model-path / --judge-model to a loadable HF checkpoint. verl
     saves training checkpoints as sharded backend-native state dicts
     (`model_world_size_N_rank_R.pt` under `actor/`) plus an
     `actor/huggingface/` dir that only has tokenizer/config, no weights. If
     the given path isn't already a merged HF dir, it's auto-merged via
     `verl.model_merger` and cached as a `merged_hf/` sibling (see
     checkpoint_merge.py). Pass --no-auto-merge to disable and require an
     already-merged path.
  1. Load the (merged) HF checkpoint.
  2. Sample N (=32) responses per AIME problem with vLLM.
  3. Compute:
       - Avg@k, Pass@k, Potential@k        (k in --ks, e.g. 8/16/32)
       - Unique Correct Count, UCC@k       (correct-answer strategy clustering)
       - Textual diversity (1 - Self-BLEU)
       - Distinct Equations  (Diversity-Aware Policy Optimization, 2505.23433)
       - Reasoning Path Divergence (RPD, 2510.26122)
  4. Dump per-question scores + aggregate to JSON, optionally log to wandb.

Usage:
  python run_eval.py \
      --model-path /path/to/verl/checkpoint/global_step_500/actor \
      --judge-model Qwen/Qwen2.5-7B-Instruct \
      --output-dir ./outputs/step500 \
      --n-samples 32 --ks 8,16,32
"""

import argparse
import json
import os
from pathlib import Path

# This cluster's GPUs report compute capability 12.0 but the system-wide
# `nvcc` is CUDA 12.0 (doesn't know the `compute_120a` target), so vLLM's
# flashinfer JIT-compiled sampler fails to build on first use. Force the
# PyTorch-native sampler instead; only takes effect if unset by the caller.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from data.aime_loader import load_aime
from sampling import VLLMSampler
from checkpoint_merge import ensure_merged_checkpoint
from metrics.distinct_equations import distinct_equations_score
from metrics.rpd import RPDScorer
from metrics.accuracy import check_samples, accuracy_summary
from metrics.textual_diversity import textual_diversity_score
from metrics.correct_clustering import ucc_at_k, cluster_report
from metrics.embedding_utils import embed_texts


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True,
                   help="Either an already-merged HF checkpoint dir, a raw "
                        "verl actor/ checkpoint dir (auto-merged unless "
                        "--no-auto-merge), or a HF Hub id.")
    p.add_argument("--judge-model", default=None,
                   help="vLLM model used for RPD step extraction. "
                        "Defaults to --model-path if not set. Same "
                        "auto-merge rules apply.")
    p.add_argument("--no-auto-merge", action="store_true",
                   help="Don't auto-merge raw verl checkpoints; require "
                        "--model-path/--judge-model to already be merged HF dirs.")
    p.add_argument("--merge-backend", default="fsdp", choices=["fsdp", "megatron"],
                   help="Backend of the raw verl checkpoint, for auto-merge.")
    p.add_argument("--merge-target-dir", default=None,
                   help="Where to write an auto-merge of --model-path. "
                        "Default: a merged_hf/ dir next to the checkpoint's actor/ dir.")
    p.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B",
                   help="Embedding model for RPD step embeddings and UCC clustering.")
    p.add_argument("--dataset", default="aime2024", choices=["aime2024", "aime2025"])
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-samples", type=int, default=32)
    p.add_argument("--ks", default="8,16,32",
                   help="Comma-separated k values for Avg@k/Pass@k/Potential@k/UCC@k. "
                        "Each must be <= --n-samples.")
    p.add_argument("--max-new-tokens", type=int, default=8000)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--skip-rpd", action="store_true",
                   help="Skip Reasoning Path Divergence (needs a judge LLM pass).")
    p.add_argument("--skip-accuracy", action="store_true",
                   help="Skip Avg@k/Pass@k/Potential@k.")
    p.add_argument("--skip-ucc", action="store_true",
                   help="Skip Correct Answer Clustering / UCC@k (needs embeddings).")
    p.add_argument("--skip-bleu", action="store_true",
                   help="Skip Self-BLEU textual diversity.")
    p.add_argument("--ucc-sim-threshold", type=float, default=0.86,
                   help="Cosine-similarity threshold for merging two correct "
                        "solutions into the same UCC cluster.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------- 0. Resolve checkpoints (auto-merge if needed) --
    model_path = args.model_path
    judge_model_path = args.judge_model
    if not args.no_auto_merge:
        model_path = ensure_merged_checkpoint(
            args.model_path, backend=args.merge_backend, target_dir=args.merge_target_dir,
        )
        if judge_model_path:
            judge_model_path = ensure_merged_checkpoint(judge_model_path, backend=args.merge_backend)

    # -------------------- 1. Load data --------------------
    problems = load_aime(args.dataset)  # List[{"id", "problem", "answer"}]
    print(f"[data] loaded {len(problems)} problems from {args.dataset}")
    print(f"[1st problem] {problems[0]}")

    # -------------------- 2. Sample responses --------------------
    sampler = VLLMSampler(
        model_path=model_path,
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

    ks = [int(k) for k in args.ks.split(",") if k.strip()]
    invalid_ks = [k for k in ks if k > args.n_samples]
    if invalid_ks:
        raise ValueError(f"--ks values {invalid_ks} exceed --n-samples {args.n_samples}")

    # -------------------- 3a. Correctness (shared by Accuracy + UCC) -----
    correctness = None
    if not args.skip_accuracy or not args.skip_ucc:
        correctness = [check_samples(samples, prob["answer"]) for prob, samples in zip(problems, rollouts)]

    # -------------------- 3b. Avg@k / Pass@k / Potential@k --------------
    accuracy_per_k = None
    if not args.skip_accuracy:
        accuracy_per_k = accuracy_summary(correctness, ks)
        for k, m in accuracy_per_k.items():
            potential_str = "n/a" if m["potential"] is None else f"{m['potential']:.4f}"
            print(f"[metric] k={k}  Avg@k={m['avg']:.4f}  Pass@k={m['pass']:.4f}  Potential@k={potential_str}")

    # -------------------- 3c. Correct Answer Clustering / UCC@k ---------
    ucc_per_k = None
    if not args.skip_ucc:
        sample_embeddings = [
            embed_texts(samples, model=args.embedding_model) for samples in rollouts
        ]
        ucc_per_k = {}
        for k in ks:
            ucc_per_k[k] = ucc_at_k(
                rollouts, correctness, k, args.embedding_model,
                sim_threshold=args.ucc_sim_threshold,
                per_question_embeddings=sample_embeddings,
            )
            print(f"[metric] UCC@{k} (mean clusters/question): {ucc_per_k[k]:.4f}")

        # Per-question cluster assignments + a 2D PCA layout, at the largest
        # k, for `visualize_clusters.py` (not needed for the UCC number itself).
        clusters_dir = out_dir / "ucc_clusters"
        clusters_dir.mkdir(parents=True, exist_ok=True)
        report_k = max(ks)
        for prob, samples, correct, embs in zip(problems, rollouts, correctness, sample_embeddings):
            report = cluster_report(
                samples, correct, report_k, args.embedding_model,
                sim_threshold=args.ucc_sim_threshold, sample_embeddings=embs,
            )
            report["id"] = prob["id"]
            safe_id = str(prob["id"]).replace("/", "_").replace(" ", "_")
            (clusters_dir / f"{safe_id}.json").write_text(json.dumps(report, indent=2))

    # -------------------- 3d. Textual diversity (1 - Self-BLEU) ---------
    bleu_per_q = None
    bleu_mean = None
    if not args.skip_bleu:
        bleu_per_q = [
            {"id": prob["id"], "textual_diversity": textual_diversity_score(samples)}
            for prob, samples in zip(problems, rollouts)
        ]
        bleu_mean = sum(b["textual_diversity"] for b in bleu_per_q) / max(len(bleu_per_q), 1)
        print(f"[metric] Textual Diversity / 1-SelfBLEU (mean over {len(bleu_per_q)} q): {bleu_mean:.4f}")

    # -------------------- 3e. Distinct Equations --------------------
    de_per_q = []
    for prob, samples in zip(problems, rollouts):
        score = distinct_equations_score(samples)
        de_per_q.append({"id": prob["id"], "distinct_eq": score})
    de_mean = sum(d["distinct_eq"] for d in de_per_q) / max(len(de_per_q), 1)
    print(f"[metric] Distinct Equations (mean over {len(de_per_q)} q): {de_mean:.4f}")

    # -------------------- 3f. RPD --------------------
    rpd_per_q = []
    rpd_mean = None
    if not args.skip_rpd:
        rpd_scorer = RPDScorer(
            judge_model=judge_model_path or model_path,
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
    correctness_per_q = None
    if correctness is not None:
        correctness_per_q = [
            {"id": prob["id"], "n": len(c), "c": sum(c)}
            for prob, c in zip(problems, correctness)
        ]

    summary = {
        "model_path": args.model_path,
        "resolved_model_path": model_path,
        "dataset": args.dataset,
        "n_samples": args.n_samples,
        "ks": ks,
        "accuracy": {
            "per_k": {str(k): m for k, m in accuracy_per_k.items()},
            "per_question": correctness_per_q,
        } if accuracy_per_k is not None else None,
        "ucc": {str(k): v for k, v in ucc_per_k.items()} if ucc_per_k is not None else None,
        "textual_diversity": {
            "mean": bleu_mean,
            "per_question": bleu_per_q,
        } if bleu_per_q is not None else None,
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