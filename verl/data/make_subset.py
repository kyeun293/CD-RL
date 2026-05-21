"""
Create a subset of dapo-math-17k.parquet by sampling unique problems.

Modes:
  sample  (default) : sample --frac of unique problems, keep all their rows (~100 each)
  repeat            : keep ALL unique problems, take --repeat rows each

Usage:
    python make_subset.py --mode sample --frac 0.1
    python make_subset.py --mode repeat --repeat 10
"""

import argparse
import os
import pandas as pd
import numpy as np


def get_problem_text(prompt):
    if isinstance(prompt, list):
        return prompt[0]["content"]
    return str(prompt)


def mode_sample(df, frac, seed):
    """Sample frac of unique problems, keep all repetitions per problem."""
    unique_problems = df["_problem_text"].unique()
    rng = np.random.default_rng(seed)
    n_sample = max(1, int(len(unique_problems) * frac))
    sampled_set = set(rng.choice(unique_problems, size=n_sample, replace=False))
    subset = df[df["_problem_text"].isin(sampled_set)]
    print(f"  Mode: sample | problems: {n_sample:,} ({frac*100:.0f}%) | rows: {len(subset):,}")
    return subset


def mode_repeat(df, repeat, seed):
    """Keep all unique problems, take `repeat` rows each."""
    rng = np.random.default_rng(seed)
    frames = []
    for _, group in df.groupby("_problem_text", sort=False):
        n = min(repeat, len(group))
        frames.append(group.sample(n=n, random_state=int(rng.integers(1 << 31))))
    subset = pd.concat(frames)
    unique_count = df["_problem_text"].nunique()
    print(f"  Mode: repeat | problems: {unique_count:,} (100%) | {repeat}x each | rows: {len(subset):,}")
    return subset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sample", "repeat"], default="repeat")
    parser.add_argument("--frac", type=float, default=0.006, help="[sample] fraction of unique problems")
    parser.add_argument("--repeat", type=int, default=10, help="[repeat] rows per unique problem")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input", type=str, default="/home/soo/yejin/CD-RL/verl/data/dapo-math-17k.parquet")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.output is None:
        output_dir = "/home/soo/yejin/CD-RL/verl/data"
        input_basename = os.path.basename(args.input)
        if args.mode == "sample":
            filename = input_basename.replace(".parquet", f"-{int(args.frac*100)}pct.parquet")
        else:
            filename = input_basename.replace(".parquet", f"-all-{args.repeat}x.parquet")
        args.output = os.path.join(output_dir, filename)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"Loading {args.input} ...")
    df = pd.read_parquet(args.input)
    print(f"  Total rows: {len(df):,}")

    df["_problem_text"] = df["prompt"].apply(get_problem_text)
    print(f"  Unique problems: {df['_problem_text'].nunique():,}")

    if args.mode == "sample":
        subset = mode_sample(df, args.frac, args.seed)
    else:
        subset = mode_repeat(df, args.repeat, args.seed)

    subset = subset.drop(columns=["_problem_text"]).reset_index(drop=True)
    subset.to_parquet(args.output, index=False)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
