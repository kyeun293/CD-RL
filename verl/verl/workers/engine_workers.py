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
import functools
import logging
import os
from contextlib import nullcontext
from copy import deepcopy
from functools import partial
from itertools import chain
from typing import Optional

import torch
from codetiming import Timer
from omegaconf import DictConfig, open_dict
from tensordict import NonTensorData, TensorDict
from torch.distributed.device_mesh import init_device_mesh

from verl.checkpoint_engine import CheckpointEngineRegistry
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.trainer.distillation import distillation_ppo_loss, is_distillation_enabled
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import get_device_name, is_npu_available, set_expandable_segments
from verl.utils.distributed import initialize_global_process_group_ray, set_numa_affinity
from verl.utils.flops_counter import FlopsCounter
from verl.utils.memory_utils import aggressive_empty_cache
from verl.utils.metric.utils import Metric
from verl.utils.profiler import DistProfiler, DistProfilerExtension, ProfilerConfig, log_gpu_memory_usage
from verl.utils.py_functional import append_to_dict
from verl.utils.tensordict_utils import maybe_fix_3d_position_ids
from verl.utils.torch_functional import allgather_dict_into_dict
from verl.workers.config import (
    ActorConfig,
    DistillationConfig,
    HFModelConfig,
    MtpConfig,
    RolloutConfig,
    TrainingWorkerConfig,
)
from verl.workers.rollout.base import BaseRollout, get_rollout_class
from verl.workers.utils.losses import ppo_loss

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
import torch.distributed as dist
from verl import DataProto
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding
import verl.utils.tensordict_utils as tu
import numpy as np
from verl.utils.profiler import marked_timer
from collections import defaultdict

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _with_routing_replay_flag(enabled: bool):
    """Decorator to set 'enable_routing_replay' flag on the data TensorDict."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, data: TensorDict, *args, **kwargs):
            if self.enable_routing_replay:
                tu.assign_non_tensor_data(data, "enable_routing_replay", enabled)
            return func(self, data, *args, **kwargs)

        return wrapper

    return decorator


class TrainingWorker(Worker, DistProfilerExtension):
    """
    TrainingWorker provides a Tinker-like API (https://thinkingmachines.ai/tinker/) as a RayWorkerGroup
    to a single controller. Currently, we only provide more coarse grained APIs,
    and do not provide exact APIs as Tinker does. But this can be added in the future.
    """

    def __init__(self, config: TrainingWorkerConfig):
        Worker.__init__(self)

        from verl.workers.engine import BaseEngine, EngineRegistry

        # TODO(jhz): Switch to `set_expandable_segments` when the torch_npu library
        # supports `torch.npu.memory._set_allocator_settings`
        if is_npu_available:
            os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"

        initialize_global_process_group_ray(timeout_second=None)

        set_numa_affinity()

        self.config = config
        self.model_config = self.config.model_config
        self.engine_config = self.config.engine_config
        self.optimizer_config = self.config.optimizer_config
        self.checkpoint_config = self.config.checkpoint_config
        self.device_name = get_device_name()

        if self.engine_config is None:
            assert self.optimizer_config is None
            if self.config.auto_select_engine_optim_fn is None:
                raise ValueError(
                    "engine_config is not provided and auto_select_engine_optim_fn is not set. "
                    "Cannot determine engine backend."
                )
            # Support automatically select engine backend given model config
            self.engine_config, self.optimizer_config = self.config.auto_select_engine_optim_fn(
                self.model_config, self.device_name
            )

        # we use the one defined in model
        # TODO: this is not elegant and should refactor later
        self.engine_config.use_remove_padding = self.model_config.use_remove_padding
        self.engine_config.use_fused_kernels = self.model_config.use_fused_kernels

        # TODO: add DistProfilerExtension
        self.profiler_config = self.config.profiler_config
        if self.profiler_config is not None:
            self.profiler_tool_config = self.profiler_config.tool_config.get(self.profiler_config.tool, {})
        else:
            self.profiler_tool_config = None

        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=self.profiler_config, tool_config=self.profiler_tool_config)
        )

        self.model_config.model_type = self.config.model_type
        self.engine: BaseEngine = EngineRegistry.new(
            model_type=self.config.model_type,
            backend=self.engine_config.strategy,
            model_config=self.model_config,
            engine_config=self.engine_config,
            optimizer_config=self.optimizer_config,
            checkpoint_config=self.checkpoint_config,
        )

        # build dispatch info
        self._register_dispatch_collect_info(
            mesh_name="train",
            dp_rank=self.engine.get_data_parallel_rank(),
            is_collect=self.engine.is_mp_src_rank_with_outputs(),
        )

        self.flops_counter = FlopsCounter(self.model_config.hf_config)

        self.loss_fn = None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def to(self, device, model=True, optimizer=True, grad=True):
        """Manual control of load/offload"""
        assert device in ["cpu", "device"]

        if device == "device":
            device = get_device_name()

        self.engine.to(device=device, model=model, optimizer=optimizer, grad=grad)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_loss_fn(self, loss_fn):
        self.loss_fn = loss_fn

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def reset(self):
        """
        Reset the model engine to the initial state. If the engine is not initialized,
        we initialize it. Otherwise, reload ckpt and reset states
        """
        self.engine.initialize()

    def _postprocess_output(self, output, *, global_token_num, delta_time, forward_only, images_seqlens):
        """

        Args:
            output: a dictionary containing loss, model_outputs and metrics

        Returns:

        """
        # TODO: whether to log memory
        # metrics["perf/max_memory_allocated_gb"] = get_torch_device().max_memory_allocated() / (1024 ** 3)
        # metrics["perf/max_memory_reserved_gb"] = get_torch_device().max_memory_reserved() / (1024 ** 3)
        # metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024 ** 3)

        metrics: dict = output.pop("metrics")
        # perform all gather in dp group to ensure that it's correct.
        # Here each metric in metrics can be a list (micro-batch metrics) or a singleton
        # we should always sum the loss of each micro-batch as we scale by global_bsz/global_token
        loss = torch.sum(torch.tensor(output.pop("loss"), device=self.device_name))
        dp_group = self.engine.get_data_parallel_group()
        if dp_group is not None:
            torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG, group=dp_group)
        loss = loss.item()

        # For grad_norm, we do not perform all reduce because it is already been done when clipping grad
        grad_norm = metrics.pop("grad_norm", None)
        lr = metrics.pop("lr", None)

        # For other metrics, we perform all gather in dp group (only if DP > 1)
        if dp_group is not None:
            final_metrics = allgather_dict_into_dict(data=metrics, group=dp_group)
        else:
            final_metrics = metrics
        final_metrics["loss"] = loss
        if grad_norm is not None:
            final_metrics["grad_norm"] = grad_norm
        if lr is not None:
            final_metrics["lr"] = lr

        # TODO: confirm the mtp loss IS same across dp
        for k, v in final_metrics.items():
            if k.startswith("mtp_losses"):
                flatten_v = [sublist[0] for sublist in v]  # sublist should be single element
                final_metrics[k] = sum(flatten_v) / len(flatten_v)
        # compute mfu
        if global_token_num is not None:
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(
                global_token_num, delta_time, images_seqlens=images_seqlens
            )
            final_metrics["mfu"] = estimated_flops / promised_flops / torch.distributed.get_world_size()
            if forward_only:
                final_metrics["mfu"] /= 3.0
        # model outputs
        model_output = output.pop("model_output", {})
        # We only return final_metrics
        final_output = tu.get_tensordict(tensor_dict=model_output, non_tensor_dict={"metrics": final_metrics})
        return final_output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train"), blocking=False)
    def train_mini_batch(self, data: TensorDict) -> TensorDict:
        """Split a batch into N mini-batches run for multiple epochs

        Args:
            data:

        Returns:

        """
        maybe_fix_3d_position_ids(data)
        batch_size_per_dp = data.shape[0]
        disable_auto_offload = tu.pop(data, key="disable_auto_offload", default=False)
        mini_batch_size = tu.pop(data, key="mini_batch_size", default=None)
        num_mini_batch = tu.pop(data, key="num_mini_batch", default=None)
        epochs = tu.pop(data, key="epochs", default=1)
        seed = tu.pop(data, key="seed", default=42)
        dataloader_kwargs = tu.pop(data, key="dataloader_kwargs", default={})

        assert mini_batch_size is not None or num_mini_batch is not None

        if mini_batch_size is None:
            assert batch_size_per_dp % num_mini_batch == 0, f"Got {batch_size_per_dp=} and {num_mini_batch=}"
            mini_batch_size_per_gpu = batch_size_per_dp // num_mini_batch
        else:
            assert mini_batch_size % self.engine.get_data_parallel_size() == 0, (
                f"Got {mini_batch_size=} and {self.engine.get_data_parallel_size()=}"
            )
            mini_batch_size_per_gpu = mini_batch_size // self.engine.get_data_parallel_size()

        # make iterator
        dataloader = tu.make_iterator(
            data,
            mini_batch_size=mini_batch_size_per_gpu,
            epochs=epochs,
            seed=seed + self.engine.get_data_parallel_rank(),
            dataloader_kwargs=dataloader_kwargs,
        )

        with (
            self.engine.train_mode(disable_auto_offload=disable_auto_offload),
            Timer(name="train_batch", logger=None),
        ):
            # update
            output_lst = []
            total_num_iterations = data.shape[0] // mini_batch_size_per_gpu * epochs

            for batch_idx, mini_batch_td in enumerate(dataloader):
                # add global token num
                global_token_num = mini_batch_td["input_ids"].offsets().diff().tolist()  # (total_nnz,)
                # allgather from dp rank
                global_token_num_output = [None] * torch.distributed.get_world_size(
                    self.engine.get_data_parallel_group()
                )
                torch.distributed.all_gather_object(
                    global_token_num_output, global_token_num, self.engine.get_data_parallel_group()
                )
                global_token_num = [x for xs in global_token_num_output for x in xs]
                tu.assign_non_tensor(
                    mini_batch_td,
                    global_token_num=NonTensorData(global_token_num),
                    update_lr_scheduler=batch_idx == total_num_iterations - 1,
                    disable_auto_offload=True,
                )
                actor_output = self.train_batch(mini_batch_td)
                output_lst.append(actor_output)

            if self.engine.is_mp_src_rank_with_outputs():
                actor_output = [tu.get(output, "metrics") for output in output_lst]
                metrics = {}
                for output in actor_output:
                    for key, val in output.items():
                        # flattn dp and micro batch
                        if isinstance(val, list):
                            output[key] = (
                                Metric.aggregate_dp(val)
                                if isinstance(val[0], Metric)
                                else list(chain.from_iterable(val))
                            )
                    append_to_dict(metrics, output)

                output = tu.get_tensordict(tensor_dict={}, non_tensor_dict={"metrics": metrics}).cpu()
            else:
                output = None
        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train"), blocking=False)
    def train_batch(self, data: TensorDict) -> TensorDict:
        assert self.loss_fn is not None, "loss function can't be None when calling train_batch"
        assert not self.engine_config.forward_only, "Can't run `train_batch` when forward_only is in the engine config."
        # global_token_num should be a list of number of tokens of each seq in this batch
        global_token_num = tu.get(data, key="global_token_num")
        disable_auto_offload = tu.get(data, key="disable_auto_offload", default=False)
        images_seqlens = tu.get(data, key="images_seqlens", default=None)

        # inject engineering parameters if not specified
        default_keys = dict(
            use_remove_padding=self.model_config.use_remove_padding,
            use_dynamic_bsz=self.engine_config.use_dynamic_bsz,
            max_token_len_per_gpu=self.engine_config.max_token_len_per_gpu,
            micro_batch_size_per_gpu=self.engine_config.micro_batch_size_per_gpu,
            use_fused_kernels=self.engine_config.use_fused_kernels,
        )

        for key, val in default_keys.items():
            if key not in data.keys():
                tu.assign_non_tensor(data, **{key: val})

        with (
            self.engine.train_mode(disable_auto_offload=disable_auto_offload),
            Timer(name="train_batch", logger=None) as timer,
        ):
            output = self.engine.train_batch(data, loss_function=self.loss_fn)
            # containing loss, model_output and metrics
            # for training, we only care about loss and metrics
        delta_time = timer.last

        update_lr_scheduler = tu.get(data, key="update_lr_scheduler", default=False)
        # update lr scheduler
        if update_lr_scheduler:
            lr = self.engine.lr_scheduler_step()
        else:
            lr = None

        if self.engine.is_mp_src_rank_with_outputs():
            # we don't need model_output in training. Maybe we change out mind later
            output.pop("model_output")
            if lr is not None:
                output["metrics"]["lr"] = lr
            final_output = self._postprocess_output(
                output,
                global_token_num=global_token_num,
                delta_time=delta_time,
                forward_only=False,
                images_seqlens=images_seqlens,
            ).cpu()
        else:
            final_output = None

        return final_output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train"), blocking=False)
    def infer_batch(self, data: TensorDict) -> TensorDict:
        # add mfu calculator
        global_token_num = tu.get(data, key="global_token_num")
        compute_loss = tu.get(data, key="compute_loss", default=True)
        disable_auto_offload = tu.get(data, key="disable_auto_offload", default=False)
        no_lora_adapter = tu.pop(data, key="no_lora_adapter", default=False)
        images_seqlens = tu.get(data, key="images_seqlens", default=None)

        default_keys = dict(
            use_remove_padding=self.model_config.use_remove_padding,
            use_dynamic_bsz=self.engine_config.use_dynamic_bsz,
            max_token_len_per_gpu=self.engine_config.infer_max_token_len_per_gpu,
            micro_batch_size_per_gpu=self.engine_config.infer_micro_batch_size_per_gpu,
            use_fused_kernels=self.engine_config.use_fused_kernels,
        )

        for key, val in default_keys.items():
            if key not in data.keys():
                tu.assign_non_tensor(data, **{key: val})

        # for sft training, we need to compute loss in eval
        loss_function = self.loss_fn if compute_loss else None

        with (
            self.engine.eval_mode(disable_auto_offload=disable_auto_offload),
            Timer(name="eval_batch", logger=None) as timer,
        ):
            adapter_ctx = self.engine.disable_adapter() if no_lora_adapter else nullcontext()
            with adapter_ctx:
                # print(f"engine_workers.py {data.batch_size}", flush=True) # 128
                output = self.engine.infer_batch(data, loss_function=loss_function)        
        delta_time = timer.last

        if self.engine.is_mp_src_rank_with_outputs():
            final_output = self._postprocess_output(
                output,
                global_token_num=global_token_num,
                delta_time=delta_time,
                forward_only=True,
                images_seqlens=images_seqlens,
            ).cpu()
        else:
            final_output = None

        return final_output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        return self.engine.save_checkpoint(local_path, hdfs_path, global_step, max_ckpt_to_keep)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        return self.engine.load_checkpoint(local_path, hdfs_path, del_local_after_load)


class ActorRolloutRefWorker(Worker, DistProfilerExtension):
    """Hybrid worker that includes actor model, rollout and optional ref model.
    For standalone actor or rollout, use ActorWorker or BaseRollout respectively.

    NOTE: ActorRolloutRefWorker no longer support spmd mode and run native server mode.
    """

    def __init__(
        self, config: DictConfig, role: str, distillation_config: Optional[DistillationConfig] = None, **kwargs
    ):
        Worker.__init__(self)
        self.config = config
        self.distillation_config = distillation_config
        self.distillation_enabled = is_distillation_enabled(distillation_config)
        self.role = role
        self.actor: TrainingWorker = None
        self.ref: TrainingWorker = None
        self.rollout: BaseRollout = None
        assert self.role in ["actor", "rollout", "ref", "actor_rollout", "actor_rollout_ref"]
        self._is_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._is_rollout = self.role in ["rollout", "actor_rollout", "actor_rollout_ref"]
        self._is_ref = self.role in ["ref", "actor_rollout_ref"]

        if self._is_actor:
            omega_profiler_config = config.actor.get("profiler", {})
        elif self._is_rollout:
            # NOTE: In colocation mode, rollout config may not take effect (follow the actor config)
            # This is for extendability in AsyncRL cases
            omega_profiler_config = config.rollout.get("profiler", {})
        else:
            omega_profiler_config = config.ref.get("profiler", {})

        profiler_config = omega_conf_to_dataclass(omega_profiler_config, dataclass_type=ProfilerConfig)
        if omega_profiler_config.get("tool", None) in ["npu", "nsys", "torch", "torch_memory"]:
            tool_config = omega_conf_to_dataclass(
                omega_profiler_config.get("tool_config", {}).get(omega_profiler_config.get("tool"))
            )
        else:
            tool_config = None

        self.enable_routing_replay = (
            self.config.actor.strategy == "megatron" and self.config.actor.megatron.router_replay.mode != "disabled"
        )

        self.timing_raw = defaultdict(float)

        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=profiler_config, tool_config=tool_config)
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_loss_fn(self, loss_fn):
        self.actor.set_loss_fn(loss_fn=loss_fn)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def to(self, device, model=True, optimizer=True, grad=True):
        """Manual control of load/offload"""
        self.actor.to(device=device, model=model, optimizer=optimizer, grad=grad)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        model_config: HFModelConfig = omega_conf_to_dataclass(self.config.model)
        self.device_name = get_device_name()
        self.device = torch.device(self.device_name)

        # 1. build reference model
        if "ref" in self.role:
            # TODO: align ref config with actor config
            with open_dict(self.config.ref):
                self.config.ref.ppo_mini_batch_size = self.config.actor.ppo_mini_batch_size
                self.config.ref.ppo_micro_batch_size = self.config.ref.pop("log_prob_micro_batch_size", None)
                self.config.ref.ppo_micro_batch_size_per_gpu = self.config.ref.pop(
                    "log_prob_micro_batch_size_per_gpu", None
                )
                self.config.ref.use_dynamic_bsz = self.config.ref.pop("log_prob_use_dynamic_bsz", False)
                self.config.ref.ppo_max_token_len_per_gpu = self.config.ref.pop("log_prob_max_token_len_per_gpu", None)
            ref_config: ActorConfig = omega_conf_to_dataclass(self.config.ref)

            # The ref model does not need to enable MTP; force it to false.
            ref_config.model_config = deepcopy(model_config)
            ref_config.model_config.mtp = MtpConfig(enable=False)

            # construct TrainingWorkerConfig
            ref_training_config = TrainingWorkerConfig(
                model_type="language_model",
                model_config=ref_config.model_config,
                engine_config=ref_config.engine,
                optimizer_config=ref_config.optim,
                checkpoint_config=ref_config.checkpoint,
            )

            # assign engine configs
            ref_training_config.engine_config.use_dynamic_bsz = self.config.ref.use_dynamic_bsz
            ref_training_config.engine_config.infer_max_token_len_per_gpu = self.config.ref.ppo_max_token_len_per_gpu
            ref_training_config.engine_config.infer_micro_batch_size_per_gpu = (
                self.config.ref.ppo_micro_batch_size_per_gpu
            )
            ref_training_config.engine_config.use_remove_padding = model_config.use_remove_padding

            self.ref = TrainingWorker(config=ref_training_config)
            self.ref.reset()
            self.set_dispatch_collect(mesh_name="ref", **self.ref.get_dispatch_collect())

        # 2. build actor model
        if "actor" in self.role:
            actor_config: ActorConfig = omega_conf_to_dataclass(self.config.actor)
            actor_config.model_config = model_config
            distillation_config: Optional[DistillationConfig] = (
                omega_conf_to_dataclass(self.distillation_config) if self.distillation_enabled else None
            )

            actor_training_config = TrainingWorkerConfig(
                model_type="language_model",
                model_config=actor_config.model_config,
                engine_config=actor_config.engine,
                optimizer_config=actor_config.optim,
                checkpoint_config=actor_config.checkpoint,
            )

            assert self.config.actor.use_dynamic_bsz == self.config.rollout.log_prob_use_dynamic_bsz

            # assign engine configs
            actor_training_config.engine_config.use_dynamic_bsz = self.config.actor.use_dynamic_bsz
            actor_training_config.engine_config.infer_max_token_len_per_gpu = (
                self.config.rollout.log_prob_max_token_len_per_gpu
            )
            actor_training_config.engine_config.infer_micro_batch_size_per_gpu = (
                self.config.rollout.log_prob_micro_batch_size_per_gpu
            )
            actor_training_config.engine_config.max_token_len_per_gpu = self.config.actor.ppo_max_token_len_per_gpu
            actor_training_config.engine_config.micro_batch_size_per_gpu = (
                self.config.actor.ppo_micro_batch_size_per_gpu
            )
            actor_training_config.engine_config.use_remove_padding = model_config.use_remove_padding

            if self.config.actor.use_dynamic_bsz:
                assert self.config.rollout.log_prob_max_token_len_per_gpu is not None
                assert self.config.actor.ppo_max_token_len_per_gpu is not None
            else:
                assert self.config.rollout.log_prob_micro_batch_size_per_gpu is not None
                assert self.config.actor.ppo_micro_batch_size_per_gpu is not None
            if self.distillation_enabled:
                self.loss_fn = partial(
                    distillation_ppo_loss, config=actor_config, distillation_config=distillation_config
                )
            else:
                self.loss_fn = partial(ppo_loss, config=actor_config)
            self.actor = TrainingWorker(config=actor_training_config)
            self.actor.reset()
            self.actor.set_loss_fn(self.loss_fn)
            self.set_dispatch_collect(mesh_name="actor", **self.actor.get_dispatch_collect())

        # 3. build rollout engine
        if "rollout" in self.role:
            rollout_config: RolloutConfig = omega_conf_to_dataclass(self.config.rollout)

            # TODO: move rollout_device_mesh into ServerAdapter
            # 3.1 build rollout device mesh (sglang need only)
            infer_tp = rollout_config.tensor_model_parallel_size * rollout_config.data_parallel_size
            infer_pp = rollout_config.pipeline_model_parallel_size
            infer_world_size = infer_tp * infer_pp
            dp = self.world_size // infer_world_size
            assert self.world_size % infer_world_size == 0, (
                f"rollout world_size: {self.world_size} is not divisible by infer_world_size: {infer_world_size}"
            )
            rollout_device_mesh = init_device_mesh(
                get_device_name(), mesh_shape=(dp, infer_tp, infer_pp), mesh_dim_names=["dp", "infer_tp", "infer_pp"]
            )

            # 3.2 initialize rollout engine
            rollout_cls: type[BaseRollout] = get_rollout_class(rollout_config.name, rollout_config.mode)
            self.rollout = rollout_cls(
                config=rollout_config, model_config=model_config, device_mesh=rollout_device_mesh
            )

            # used for LoRA (base_sync_done is unused in merge-only mode but kept for Phase 2 adapter path)
            self.base_sync_done: bool = "dummy" not in self.config.rollout.load_format
            self.layered_summon = self.config.rollout.get("layered_summon", False)
            self.peft_merge: bool = model_config.lora.get("merge", False)

        # 4. build checkpoint engine
        if "actor" in self.role:
            checkpoint_engine_config = omega_conf_to_dataclass(self.config.rollout.checkpoint_engine)
            backend = checkpoint_engine_config.backend
            bucket_size = checkpoint_engine_config.update_weights_bucket_megabytes << 20
            engine_kwargs = checkpoint_engine_config.engine_kwargs.get(backend, {})
            self.checkpoint_engine = CheckpointEngineRegistry.new(
                backend, is_master=(torch.distributed.get_rank() == 0), bucket_size=bucket_size, **engine_kwargs
            )

        # 5. build curiosity modules (ICM & PRM)
        if "actor" in self.role and self.config.get('use_curiosity', False):            
            # curiosity_actor.py에서 실제 모델 클래스들을 임포트 (기존 CuriosityActor 내부 구조에 따라 수정 필요)
            from recipe.dapo.prm_scorer import PRMScorer
            from icm.icm_module import ICM
            from transformers import get_scheduler, AutoConfig

            # --- 토크나이저 로드 ---
            from verl.utils import hf_tokenizer
            local_path = self.config.model.path
            trust_remote_code = self.config.get("trust_remote_code", False)
            self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
            print("[INIT-DEBUG] Tokenizer loaded", flush=True)

            # --- PRM Scorer 초기화 ---
            prm_model_path = self.config.get('prm_model_path', None)
            if prm_model_path:
                self.prm_scorer = PRMScorer(prm_model_path)
                # FSDP 설정 (Actor 설정과 맞추는 것이 좋습니다)
                mixed_precision_config = MixedPrecision(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.bfloat16,
                    buffer_dtype=torch.bfloat16,
                )

                self.prm_scorer.model = FSDP(
                    self.prm_scorer.model,
                    sharding_strategy=ShardingStrategy.FULL_SHARD, # Zero-3 방식: 파라미터 전체 샤딩
                    mixed_precision=mixed_precision_config,
                    device_id=torch.cuda.current_device(),
                    limit_all_gathers=True, # 메모리 안정성을 위해 통신 제한
                )
                
                self.prm_scorer.model.eval()
                print(f"[INIT-DEBUG] PRM initialized on {self.prm_scorer.device} as dtype {self.prm_scorer.model.dtype}", flush=True)
                if dist.is_initialized():
                    world_size = dist.get_world_size()
                    rank = dist.get_rank()
                    print(f"[INIT-DEBUG] PRM world size: {world_size}, rank: {rank}", flush=True)
                else:
                    print(f"[INIT-DEBUG] PRM distributed not initialized", flush=True)
            
            # --- ICM 초기화 ---
            # Config 및 파라미터 준비
            icm_config = self.config.icm
            model_config = AutoConfig.from_pretrained(self.config.model.path)
            hidden_size = model_config.hidden_size
            intermediate_size = icm_config.icm_intermediate_size
            total_steps = self.config.total_training_steps

            # 모델 생성 및 설정
            # .cuda() 대신 .to(self.device)를 사용하여 Ray 할당 자원 준수
            self.icm = ICM(hidden_size, intermediate_size).to(dtype=torch.float32, device=self.device)
            
            self.icm_optimizer = torch.optim.Adam(
                self.icm.parameters(),
                lr=icm_config.lr,
                betas=(0.9, 0.95),
            )
            
            self.icm_lr_scheduler = get_scheduler(
                name=icm_config.lr_scheduler_type,
                optimizer=self.icm_optimizer,
                num_warmup_steps=icm_config.warmup_steps,
                num_training_steps=total_steps,
            )
            print(f"[INIT-DEBUG] ICM initialized on {self.device} as dtype {next(self.icm.parameters()).dtype}", flush=True)

        # Free cached GPU memory so colocated vLLM processes can see it via cudaMemGetInfo
        aggressive_empty_cache(force_sync=True)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="ref"))
    @DistProfiler.annotate(color="olive", role="ref_compute_log_prob")
    @_with_routing_replay_flag(enabled=False)
    def compute_ref_log_prob(self, data: TensorDict) -> TensorDict:
        output = self.ref.infer_batch(data=data)
        return output.cpu() if output is not None else None

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="blue", role="actor_compute_log_prob")
    @_with_routing_replay_flag(enabled=True)
    def compute_log_prob(self, data: TensorDict) -> TensorDict:
        output = self.actor.infer_batch(data)

        return output.cpu() if output is not None else None

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="red", role="actor_update")
    @_with_routing_replay_flag(enabled=True)
    def update_actor(self, data: TensorDict) -> TensorDict:
        output = self.actor.train_mini_batch(data=data)
        return output.cpu() if output is not None else None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        assert "actor" in self.role, "load_checkpoint only support actor role"
        self.actor.load_checkpoint(local_path, hdfs_path, del_local_after_load)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        assert "actor" in self.role, "save_checkpoint only support actor role"
        self.actor.save_checkpoint(local_path, hdfs_path, global_step, max_ckpt_to_keep)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def update_weights(self, global_steps: int = None):
        """Update weights from trainer to rollout.

        1. For sync training with colocated trainer and rollout, update rollout directly from model engine.
           - before update_weights: rollout should be in sleep mode.
           - after update_weights: rollout should be in wake_up mode.
        2. For async training with disaggregated trainer and rollout, send_weights only by checkpoint engine.

        LoRA handling: when model.lora.merge=True (peft_merge), LoRA is merged into
        base weights before sync. The engine returns full HF-keyed params with
        peft_config=None, so the rollout receives a standard weight update.
        """

        # 0. send_weights only for async training with disaggregated trainer and rollout
        if self.config.rollout.checkpoint_engine.backend != "naive":
            per_tensor_param, _ = self.actor.engine.get_per_tensor_param()
            await self.checkpoint_engine.send_weights(per_tensor_param)
            return

        set_expandable_segments(False)
        log_gpu_memory_usage("Before resume weights", logger=logger)

        # 1. resume rollout memory (weights were released during sleep)
        # When sleep_level=1 (adapter mode), sleep() only released kv_cache,
        # so skip the weight resume to avoid a no-op sglang call.
        if self.config.rollout.free_cache_engine:
            if getattr(self.rollout, "sleep_level", 2) != 1:
                await self.rollout.resume(tags=["weights"])
        log_gpu_memory_usage("After resume weights", logger=logger)

        # 2. determine if we need a base weight sync (adapter path only)
        per_tensor_param, peft_config = self.actor.engine.get_per_tensor_param(
            layered_summon=self.layered_summon, base_sync_done=self.base_sync_done
        )

        do_lora_base_sync = False
        if not self.peft_merge and peft_config is not None:
            self.rollout.sleep_level = 1
            do_lora_base_sync = not self.base_sync_done

        # 3. sync weights: base first (when needed), then adapter/merged
        if do_lora_base_sync:
            # First iteration: per_tensor_param has base params (base_sync_done was False).
            # Send base weights, then fetch and send adapter deltas.
            await self.rollout.update_weights(
                per_tensor_param, peft_config=peft_config, base_sync_done=False, global_steps=global_steps
            )
            per_tensor_param, peft_config = self.actor.engine.get_per_tensor_param(
                layered_summon=self.layered_summon, base_sync_done=True
            )
            await self.rollout.update_weights(
                per_tensor_param, peft_config=peft_config, base_sync_done=True, global_steps=global_steps
            )
        else:
            await self.rollout.update_weights(
                per_tensor_param, peft_config=peft_config, base_sync_done=self.base_sync_done, global_steps=global_steps
            )

        log_gpu_memory_usage("After update_weights", logger=logger)

        # 3. offload model to cpu
        if self.actor.engine.is_param_offload_enabled:
            self.actor.engine.to("cpu", model=True, optimizer=False, grad=False)
        aggressive_empty_cache(force_sync=True)

        # 4. resume kv_cache
        if self.config.rollout.free_cache_engine:
            await self.rollout.resume(tags=["kv_cache"])
        log_gpu_memory_usage("After resume kv_cache", logger=logger)

        self.base_sync_done = True
        set_expandable_segments(True)

    @register(dispatch_mode=Dispatch.DP_COMPUTE, blocking=False)
    def execute_checkpoint_engine(self, method: str, *args, **kwargs):
        """Execute checkpoint engine method.

        Args:
            method (str): Checkpoint engine method name.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        """
        return getattr(self.checkpoint_engine, method)(*args, **kwargs)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    # 수정: icm state 저장 및 로드
    def get_icm_state(self):
        if torch.distributed.get_rank() != 0:
            return None
        
        import copy
        # optimizer state_dict는 copy해서 작업 (원본 건드리지 않게)
        opt_state = copy.deepcopy(self.icm_optimizer.state_dict())
        for state in opt_state["state"].values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cpu()
        return {
            "icm":{k: v.cpu() for k, v in self.icm.state_dict().items()},
            "icm_optimizer": opt_state,
            "icm_lr_scheduler": self.icm_lr_scheduler.state_dict(),
        }

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_icm_state(self, state):
        if state is None:
            return
        import copy
        state = copy.deepcopy(state)

        # 명시적으로 현재 워커의 GPU device 사용
        target_device = torch.device(f"cuda:{torch.cuda.current_device()}")

        # ICM 모델 GPU로
        self.icm.load_state_dict({k: v.to(target_device) for k, v in state["icm"].items()})
        
        # optimizer state GPU로
        self.icm_optimizer.load_state_dict(state["icm_optimizer"])  # 먼저 로드한 후 텐서를 GPU로 이동
        for s in self.icm_optimizer.state.values():
            for k, v in s.items():
                if isinstance(v, torch.Tensor):
                    s[k] = v.to(target_device)
        self.icm_lr_scheduler.load_state_dict(state["icm_lr_scheduler"])



    def _find_sep_positions(self, token_ids, step_sep="Step"):
        """Return token indices whose decoded string ends with the step separator."""
        positions = []
        for j in range(len(token_ids)):
            tok_str = self.tokenizer.decode([token_ids[j].item()])
            if tok_str.endswith(step_sep):
                positions.append(j)
        return positions
    
    def _attach_parsed_steps(self, batch, step_sep=None):
        all_steps = []
        all_boundaries = []

        for i in range(len(batch)):
            response_ids = batch.batch["responses"][i]
            response_mask = batch.batch["response_mask"][i]  # 1이 실제 토큰, 0이 패딩
            valid_response_ids = response_ids[response_mask.bool()]
            valid_len = int(response_mask.sum().item())

            if step_sep:
                # Find step boundaries at token level so PRM (text) and ICM (ids) share
                # the exact same step indices.
                with marked_timer("find_sep_positions", self.timing_raw):
                    boundaries = self._find_sep_positions(valid_response_ids, step_sep=step_sep) # step separator가 있는 토큰의 인덱스: [60, 209, 314, 394, 453, 519, 645, 672]

                if not boundaries:
                    with marked_timer("decode_full_response", self.timing_raw):
                        step_texts = [self.tokenizer.decode(valid_response_ids, skip_special_tokens=True).strip()]
                        boundaries = [0, valid_len]
                        all_steps.append(step_texts)
                        all_boundaries.append(boundaries)
                        continue

                if boundaries[-1] != valid_len:
                    boundaries.append(valid_len)

                # 각 step 텍스트 추출
                with marked_timer("decode_step_texts", self.timing_raw):
                    step_texts = []
                    for k in range(len(boundaries) - 1):
                        seg = valid_response_ids[boundaries[k]:boundaries[k+1]]
                        text = self.tokenizer.decode(seg, skip_special_tokens=True).strip()
                        step_texts.append(text)
                
                # if i == 0:
                #     # print(f"[PARSE-DEBUG] boundaries: {boundaries}", flush=True) # [0, 27, 51, 71, 93, 117]
                #     # print(f"[PARSE-DEBUG] step_texts: {step_texts}", flush=True) # ["Step 1: ...", "Step 2: ...", ..., "Step 8: ... Answer: ..."]
                #     # print(f"[PARSE-DEBUG] step len: {len(step_texts)}", flush=True) # 5
                #     print(f"[PARSE-DEBUG] i={i} steps={len(step_texts)} boundaries={boundaries}", flush=True)
                #     for si, s in enumerate(step_texts):
                #         print(f"  step[{si}]: {repr(s[:120])}", flush=True)

            else:
                step_texts = []
                boundaries = []

            all_steps.append(step_texts)
            all_boundaries.append(boundaries)

        batch.non_tensor_batch["parsed_steps"] = np.array(all_steps, dtype=object)
        batch.non_tensor_batch["step_boundaries"] = np.array(all_boundaries, dtype=object)

        return batch
    
    def _score_prm(self, batch):
        all_labels = []
        
        for i in range(len(batch)):
            steps = batch.non_tensor_batch["parsed_steps"][i]
            if steps:
                prompt_ids = batch.batch["prompts"][i]
                prompt_length = prompt_ids.shape[-1]
                prompt_mask = batch.batch["attention_mask"][i][:prompt_length].bool()
                valid_prompt_ids = prompt_ids[prompt_mask]
                problem_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
                labels = self.prm_scorer.score_steps(problem_str, steps)
            else:
                labels = []
            all_labels.append(labels)
        
        # if len(all_labels[0]) > 0:
        #     print(f"[PRM-DEBUG] i=0 steps={len(all_labels[0])} labels={all_labels[0]}", flush=True) # i=0 steps=5 labels=[1, 1, 1, 1, 0]
        
        batch.non_tensor_batch["prm_labels"] = np.array(all_labels, dtype=object)
        return batch

    def _get_ref_hidden_states(self, batch):
        # extract reference hidden states at step boundaries
        batch_td = batch.to_tensordict()
        batch_td = left_right_2_no_padding(batch_td)
        metadata = {"calculate_entropy": False, "compute_loss": False, 'output_hidden_states': True}
        if self.config.ref_in_actor:
            metadata["no_lora_adapter"] = True
        tu.assign_non_tensor(batch_td, **metadata)
        
        output = self.ref.infer_batch(batch_td)

        log_probs = tu.get(output, "log_probs", default=None)
        log_probs = no_padding_2_padding(log_probs, batch_td)
        batch.batch["ref_log_prob"] = log_probs.float()

        ref_hidden_states_raw = tu.get(output, "hidden_states", default=None)
        ref_hidden_states_raw = no_padding_2_padding(ref_hidden_states_raw, batch_td, is_hidden_states=True)
        batch.meta_info["output_hidden_states"] = False
        # print(f"[REF-HS-DEBUG] ref_hidden_states_raw shape: {ref_hidden_states_raw.shape}", flush=True)

        step_boundaries = batch.non_tensor_batch["step_boundaries"]

        ref_hidden_states = []
        for i in range(len(step_boundaries)):
            boundaries = step_boundaries[i]
            indices = np.concatenate([[boundaries[0]], np.array(boundaries[1:]) - 1])
            ref_hidden_states.append(ref_hidden_states_raw[i, indices, :])

        batch.non_tensor_batch["ref_hidden_states"] = ref_hidden_states

        del output
        del ref_hidden_states_raw, ref_hidden_states
        del log_probs
        torch.cuda.empty_cache()
        
        return batch

    def _compute_actor_hidden_states(self, batch: DataProto) -> torch.Tensor:
        """
        actor로 hidden states만 추출. log_prob/entropy 계산 없음.
        """
        batch_td = batch.to_tensordict()
        batch_td = left_right_2_no_padding(batch_td)
        tu.assign_non_tensor(batch_td, calculate_entropy=False, compute_loss=False, output_hidden_states=True)
        
        output = self.actor.infer_batch(batch_td)
        
        hidden_states = tu.get(output, "hidden_states", default=None)
        assert hidden_states is not None, "output_hidden_states=True 설정 필요"
        
        return hidden_states  # (batch_size, seq_len, hidden_dim)

    def _get_actor_hidden_states_chunk(self, token_ids_list: list[torch.Tensor]) -> list[torch.Tensor]:
        """
            token_ids_list: 여러 reasoning step 시퀀스가 담긴 리스트
        """
        bs = len(token_ids_list)

        # dp_size의 배수로 패딩
        dp_size = self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes \
                // self.config.actor.ulysses_sequence_parallel_size
        padded_bs = ((bs + dp_size - 1) // dp_size) * dp_size
        pad_count = padded_bs - bs
        padded_token_ids_list = token_ids_list + [token_ids_list[0]] * pad_count

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
                "temperature": torch.ones(padded_bs, dtype=torch.float),
            },
            batch_size=[padded_bs],
        ))
        step_batch.meta_info["output_hidden_states"] = True
        step_batch.meta_info["compute_loss"] = False

        step_hidden_states = self._compute_actor_hidden_states(step_batch)

        # 각 step의 마지막 토큰의 hidden state만 추출
        step_last_hidden_states = [
            step_hidden_states[i, response_mask[i].sum() - 1]
            for i in range(bs) # padded_bs가 아닌 원래 bs까지만
        ]

        # print(f"step_hidden_states shape: {step_hidden_states.shape}", flush=True) # torch.Size([num_steps_in_batch, seq_len, hidden_dim])

        return step_last_hidden_states

    def _get_actor_hidden_states(self, batch, step_boundaries):
        responses = batch.batch["responses"]
        batch_size = responses.shape[0]

        # 모든 response의 모든 step을 하나의 리스트로 flatten
        all_steps = []       # 모든 step의 input_ids
        step_map = []        # (response_idx, step_idx) 복원용 맵
        
        for idx in range(batch_size):
            response_sample = responses[idx]            
            boundaries = step_boundaries[idx]
            
            for step_idx in range(len(boundaries) - 1):
                start = boundaries[step_idx]
                end = boundaries[step_idx + 1]
                step_tokens = response_sample[start:end]
                all_steps.append(step_tokens)
                step_map.append((idx, step_idx))

        all_hidden_states = self._get_actor_hidden_states_chunk(all_steps)
        step_last_hidden_states = [[] for _ in range(batch_size)]
        for i, hidden_state in enumerate(all_hidden_states):
            resp_idx, step_idx = step_map[i]
            step_last_hidden_states[resp_idx].append(hidden_state)
        step_last_hidden_states = [
            torch.stack(h, dim=0) for h in step_last_hidden_states
        ]  # (num_steps, hidden_dim) 리스트로 변환
        # print(f"[ACTOR-HS-DEBUG] step_last_hidden_states len: {len(step_last_hidden_states)}", flush=True) # batch size = 32개 데이터
        # print(f"[ACTOR-HS-DEBUG] step_last_hidden_states[0] shape: {step_last_hidden_states[0].shape}", flush=True) # torch.Size([5, 2048])

        batch.non_tensor_batch["actor_hidden_states"] = step_last_hidden_states

        del all_steps, all_hidden_states
        del step_last_hidden_states
        torch.cuda.empty_cache()

        return batch

    def _compute_icm(self, ref_hidden_states, actor_hidden_states):
        """
        ref_hidden_states: list of (num_steps, hidden_dim) tensors
        actor_hidden_states: list of (num_steps, hidden_dim) tensors
        returns: intrinsic_reward (num_pairs,), pair_map
        """
        s_t_batch, a_t_batch, s_t1_batch, pair_map = self.icm.prepare_icm_input(
            ref_hidden_states, actor_hidden_states
        )
        device = next(self.icm.parameters()).device
        s_t_batch = s_t_batch.to(device=device, dtype=torch.float32)
        a_t_batch = a_t_batch.to(device=device, dtype=torch.float32)
        s_t1_batch = s_t1_batch.to(device=device, dtype=torch.float32)

        next_state, next_state_hat = self.icm(s_t_batch, s_t1_batch, a_t_batch)

        icm_loss = self.icm.icm_loss(next_state, next_state_hat)

        self.icm_optimizer.zero_grad()
        icm_loss.backward()
        
        for name, param in self.icm.named_parameters():
            if param.grad is not None:
                print(f"BEFORE allreduce - {name}: param={param.device}, grad={param.grad.device}")


        device = next(self.icm.parameters()).device
        for param in self.icm.parameters():
            if param.grad is not None:
                # param.grad = param.grad.to(device)  # gradient를 GPU로 강제
                torch.distributed.all_reduce(param.grad, op=torch.distributed.ReduceOp.AVG)


        for name, param in self.icm.named_parameters():
            if param.grad is not None:
                print(f"AFTER allreduce - {name}: param={param.device}, grad={param.grad.device}")
        self.icm_optimizer.step()
        self.icm_lr_scheduler.step()

        icm_loss_val = icm_loss.item()
        del s_t_batch, a_t_batch, s_t1_batch, icm_loss

        intrinsic_reward = 0.5 * (next_state - next_state_hat).norm(2, dim=-1)
        mean = intrinsic_reward.mean()
        var = intrinsic_reward.var(unbiased=False)
        intrinsic_reward = (intrinsic_reward - mean) * torch.rsqrt(var + 1e-8)
        # 수정: sigmoid로 intrinsic reward range 0~1
        intrinsic_reward = torch.sigmoid(intrinsic_reward)
        intrinsic_reward = intrinsic_reward.detach().cpu()

        del next_state, next_state_hat
        torch.cuda.empty_cache()

        return intrinsic_reward, pair_map, icm_loss_val

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def compute_curiosity(self, batch):
        # offset 계산
        global_indices = batch.non_tensor_batch["global_indices"]
        data_idx_offset = int(global_indices[0])
        batch_size = len(batch)

        # 1. parse steps from response using step separator
        step_sep = batch.meta_info.get("step_sep", None)
        with marked_timer("parse", self.timing_raw):
            batch = self._attach_parsed_steps(batch, step_sep=step_sep)
        with marked_timer("prm_score", self.timing_raw):
            batch = self._score_prm(batch)
        # print(f"[PRM-DEBUG] prm_labels len: {len(batch.non_tensor_batch['prm_labels'])}", flush=True) # batch size = 32개 데이터
        # [list([1, 1, 1, 1, 0]) list([1, 1, 1, 1, 1, 0]) list([1, 1, 1, 1, 0, 0]) list([1, 0, 0]) list([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])]

        step_boundaries = batch.non_tensor_batch["step_boundaries"]
        # print(f"[ICM-DEBUG] step_boundaries len: {len(step_boundaries)}", flush=True) # batch size = 32개 데이터
        # print(f"[ICM-DEBUG] first step_boundaries: {step_boundaries[0]} (showing first sample)", flush=True)
        # [[0, 27, 51, 71, 93, 117], [0, 23, 62, 121, 164, 193, 222], [73, 161, 259, 586, 704, 734, 963], [0, 123, 201, 248], [0, 41, 73, 101, 163, 199, 236, 260, 301, 329, 490]] (batch size 32)

        # 2. compute ICM loss
        # print(f"[ICM-DEBUG] Computing reference hidden states", flush=True)
        with marked_timer("ref_hidden_states", self.timing_raw):
            batch = self._get_ref_hidden_states(batch)
        # print(f"ref_hidden_states len: {len(ref_hidden_states)}", flush=True) # batch size = 32개 데이터
        # print(f"ref_hidden_states[0] shape: {ref_hidden_states[0].shape}", flush=True) # step 수+1: torch.Size([6, 2048])
        # print(f"ref_hidden_states[1] shape: {ref_hidden_states[1].shape}", flush=True) # step 수+1: torch.Size([7, 2048])
        
        # print(f"[ICM-DEBUG] Computing actor hidden states", flush=True)
        
        with marked_timer("actor_hidden_states", self.timing_raw):
            batch = self._get_actor_hidden_states(batch, step_boundaries)
        # print(f"actor_hidden_states len: {len(actor_hidden_states)}", flush=True) # batch size = 32개 데이터
        # print(f"actor_hidden_states[0] shape: {actor_hidden_states[0].shape}", flush=True) # step 수: torch.Size([5, 2048])
        # print(f"actor_hidden_states[1] shape: {actor_hidden_states[1].shape}", flush=True) # step 수: torch.Size([6, 2048])

        # print(f"[ICM-DEBUG] Computing ICM intrinsic rewards", flush=True)
        with marked_timer("icm_compute", self.timing_raw):
            intrinsic_reward, pair_map, icm_loss_val = self._compute_icm(batch.non_tensor_batch["ref_hidden_states"], batch.non_tensor_batch["actor_hidden_states"])
        # print(f"[ICM-DEBUG] icm_loss: {icm_loss_val}", flush=True) # 8.375
        # print(f"[ICM-DEBUG] intrinsic_reward shape:{intrinsic_reward.shape}", flush=True) # torch.Size([1327]) 총 step 수
        # print(f"[ICM-DEBUG] intrinsic_reward sample: {intrinsic_reward[:10]} (showing first 5 samples)", flush=True) # tensor([ 0.6016,  0.1338,  0.4023,  1.0000,  2.0000, -0.3340, -0.7344, -0.4023, -0.3340,  0.2676], dtype=torch.bfloat16)
        # print(f"[ICM-DEBUG] pair_map len: {len(pair_map)}", flush=True) # 1327 총 step 수
        # print(f"[ICM-DEBUG] pair_map sample: {pair_map[:10]} (showing first 5 samples)", flush=True) # [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4)]

        # reorganize intrinsic_reward and pair map to align with batch samples
        pair_map = [(idx + data_idx_offset, step_idx) for idx, step_idx in pair_map]
        intrinsic_reward_per_sample, pair_map_per_sample = [], []
        for boundary in step_boundaries:
            n_steps = len(boundary) - 1
            intrinsic_reward_per_sample.append(intrinsic_reward[:n_steps].cpu().float().numpy())
            pair_map_per_sample.append(pair_map[:n_steps])
            intrinsic_reward = intrinsic_reward[n_steps:]
            pair_map = pair_map[n_steps:]

        result = DataProto(
            batch=TensorDict(
                {"ref_log_prob": batch.batch["ref_log_prob"]},
                batch_size=batch_size,
            ),
            non_tensor_batch={
                "intrinsic_reward": np.array(intrinsic_reward_per_sample, dtype=object),
                "prm_labels": batch.non_tensor_batch["prm_labels"],
                "pair_map": np.array(pair_map_per_sample, dtype=object),
                "step_boundaries": step_boundaries,
                "icm_loss": np.array([icm_loss_val] * batch_size, dtype=np.float32),
                "parsed_steps": batch.non_tensor_batch["parsed_steps"],
            }
        )

        del batch
        del intrinsic_reward
        del pair_map
        del icm_loss_val
        torch.cuda.empty_cache()

        for key, value in self.timing_raw.items():
            print(f"Curiosity timing - {key}: {value:.2f} seconds", flush=True)

        return result
