"""
Auto-merge verl FSDP/Megatron checkpoints into a loadable HF model dir.

verl saves training checkpoints as sharded backend-native state dicts
(`model_world_size_N_rank_R.pt` under an `actor/` dir); the `actor/huggingface/`
subdir it writes alongside only has tokenizer/config, not weights. vLLM needs
a real merged HF checkpoint (`*.safetensors`), so before loading a model path
we check whether it's already merged and, if not, merge it via
`verl.model_merger` and cache the result next to the checkpoint.
"""

from pathlib import Path
from typing import Optional

_WEIGHT_GLOBS = ("*.safetensors", "pytorch_model*.bin",
                  "model.safetensors.index.json", "pytorch_model.bin.index.json")


def _has_hf_weights(path: Path) -> bool:
    return path.is_dir() and any(list(path.glob(g)) for g in _WEIGHT_GLOBS)


def _find_actor_dir(path: Path) -> Optional[Path]:
    """`path` may be the actor dir itself, or its parent (global_step_N/)."""
    if list(path.glob("model_world_size_*_rank_*.pt")):
        return path
    actor = path / "actor"
    if actor.is_dir() and list(actor.glob("model_world_size_*_rank_*.pt")):
        return actor
    return None


def ensure_merged_checkpoint(
    model_path: str,
    backend: str = "fsdp",
    trust_remote_code: bool = True,
    target_dir: Optional[str] = None,
) -> str:
    """Return a path vLLM can load directly.

    - If `model_path` doesn't exist locally (e.g. a HF Hub id like
      "Qwen/Qwen2.5-7B"), returned unchanged.
    - If it already contains HF weight files, returned unchanged.
    - Otherwise it must be a raw verl actor checkpoint dir (or its parent);
      merge it via `verl.model_merger` into `target_dir` (default: a
      `merged_hf` sibling of the actor dir) and return that path. A prior
      merge at the same target is reused instead of re-running.
    """
    p = Path(model_path)
    if not p.exists():
        return model_path
    if _has_hf_weights(p):
        return model_path

    actor_dir = _find_actor_dir(p)
    if actor_dir is None:
        raise FileNotFoundError(
            f"'{model_path}' has neither HF weight files ({', '.join(_WEIGHT_GLOBS)}) "
            f"nor verl checkpoint shards (model_world_size_*_rank_*.pt). "
            f"Pass a merged HF dir, a raw verl actor/ dir, or a HF Hub id."
        )

    target = Path(target_dir) if target_dir else actor_dir.parent / "merged_hf"
    if _has_hf_weights(target):
        print(f"[merge] reusing cached merge at {target}")
        return str(target)

    print(f"[merge] '{model_path}' has no HF weights yet; merging {actor_dir} -> {target}")
    from verl.model_merger.base_model_merger import ModelMergerConfig

    if backend == "fsdp":
        from verl.model_merger.fsdp_model_merger import FSDPModelMerger as merger_cls
    else:
        from verl.model_merger.megatron_model_merger import MegatronModelMerger as merger_cls

    target.mkdir(parents=True, exist_ok=True)
    config = ModelMergerConfig(
        operation="merge",
        backend=backend,
        local_dir=str(actor_dir),
        hf_model_config_path=str(actor_dir / "huggingface"),
        target_dir=str(target),
        trust_remote_code=trust_remote_code,
    )
    merger = merger_cls(config)
    merger.merge_and_save()
    merger.cleanup()
    print(f"[merge] done -> {target}")
    return str(target)
