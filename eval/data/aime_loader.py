"""AIME loader. Defaults to AIME 2024 (30 problems)."""

from typing import List, Dict


def load_aime(name: str = "aime2024") -> List[Dict]:
    from datasets import load_dataset

    if name == "aime2024":
        ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
        prob_key = "Problem"
        ans_key = "Answer"
        id_key = "ID"
    elif name == "aime2025":
        # HuggingFaceH4/aime_2025 if available; else swap to your preferred mirror.
        ds = load_dataset("opencompass/AIME2025", "AIME2025-I", split="test")
        prob_key = "question"
        ans_key = "answer"
        id_key = None
    else:
        raise ValueError(f"unknown dataset: {name}")

    out = []
    for i, row in enumerate(ds):
        out.append({
            "id": row[id_key] if id_key and id_key in row else f"{name}-{i:03d}",
            "problem": row[prob_key],
            "answer": str(row[ans_key]),
        })
    return out