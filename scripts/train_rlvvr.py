"""RLVVR training: Flow-GRPO on Stable Diffusion 3.5 Medium with a LoRA adapter.

Adapted from Flow-GRPO (https://github.com/yifan123/flow_grpo, MIT license), `scripts/train_sd3.py`.
Differences from that script, all used in the paper runs:

- `DistributedKRepeatSourceBalancedSampler` draws a fixed number of unique prompts per training source in
  every batch (`config.train_source_balance`).
- Each rollout of `num_batches_per_epoch` batches draws distinct prompts.
- Rewards are gathered under a rank-consistent key schema so mixed-source batches can carry
  source-specific details.
- The PPO log-ratio is clamped to [-20, 20], and an optimizer step whose loss is non-finite on any rank is
  skipped on all ranks.
- Training stops at `config.max_global_step` optimizer steps.

Usage:
    accelerate launch --num_processes 8 scripts/train_rlvvr.py --config configs/rlvvr.py:vvr_easy
"""

import contextlib
import datetime
import hashlib
import json
import os
import random
import time
from collections import defaultdict
from concurrent import futures
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
import tqdm
from absl import app, flags
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from diffusers import StableDiffusion3Pipeline
from diffusers.utils.torch_utils import is_compiled_module
from ml_collections import config_flags
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader

import rlvvr.rewards
from rlvvr.data import (
    DistributedKRepeatSampler,
    DistributedKRepeatSourceBalancedSampler,
    MetadataPromptDataset,
    TextPromptDataset,
)
from rlvvr.diffusers_patch.sd3_pipeline_with_logprob import pipeline_with_logprob
from rlvvr.diffusers_patch.sd3_sde_with_logprob import sde_step_with_logprob
from rlvvr.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt
from rlvvr.ema import EMAModuleWrapper
from rlvvr.stat_tracking import PerPromptStatTracker

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "configs/rlvvr.py:vvr_easy", "Training configuration.")

logger = get_logger(__name__)

# Seed of the prompt sampler. It is fixed across runs; `config.seed` seeds image sampling.
SAMPLER_SEED = 42
REWARD_MISSING_VALUE = -10.0


def _wandb():
    if os.environ.get("WANDB_MODE") == "disabled":
        return None
    try:
        import wandb
    except ImportError:
        return None
    return wandb


def _enable_transformer_gradient_checkpointing(model):
    candidates = [model]
    base_model = getattr(model, "base_model", None)
    if base_model is not None:
        candidates.append(base_model)
        nested = getattr(base_model, "model", None)
        if nested is not None:
            candidates.append(nested)
    for candidate in candidates:
        if hasattr(candidate, "enable_gradient_checkpointing"):
            candidate.enable_gradient_checkpointing()
            return True
        if hasattr(candidate, "gradient_checkpointing_enable"):
            candidate.gradient_checkpointing_enable()
            return True
    return False


def _safe_info_value(value, default=0.0):
    if isinstance(value, torch.Tensor):
        return torch.nan_to_num(value.detach().float(), nan=default, posinf=default, neginf=default)
    return torch.tensor(float(value), dtype=torch.float32)


def log_metrics_local(save_dir, metrics, global_step):
    if not save_dir:
        return
    os.makedirs(save_dir, exist_ok=True)
    row = {"global_step": int(global_step)}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            value = value.item()
        elif isinstance(value, np.generic):
            value = value.item()
        row[key] = value
    with open(os.path.join(save_dir, "metrics.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")


_PROMPT_GROUP_KEY_PREFIX = "prompt_sha256_63:"


def _prompt_group_key_int(prompt):
    digest = hashlib.sha256(str(prompt).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _gather_prompt_group_keys(local_prompts, accelerator):
    """Gather prompt identity keys in rank-major order, matching the reward gather."""
    local_keys = [_prompt_group_key_int(prompt) for prompt in local_prompts]
    if not dist.is_available() or not dist.is_initialized():
        return [f"{_PROMPT_GROUP_KEY_PREFIX}{int(key)}" for key in local_keys]
    local_tensor = torch.as_tensor(local_keys, device=accelerator.device, dtype=torch.long)
    gathered = accelerator.gather(local_tensor).detach().cpu().tolist()
    return [f"{_PROMPT_GROUP_KEY_PREFIX}{int(key)}" for key in gathered]


def _reward_values_to_numpy(value, *, key, expected_len):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim != 1 or len(arr) != expected_len:
        raise ValueError(f"reward key '{key}' has shape {arr.shape}, expected ({expected_len},)")
    if not np.all(np.isfinite(arr[arr != REWARD_MISSING_VALUE])):
        raise ValueError(f"reward key '{key}' contains non-finite values")
    return arr


def _gather_reward_dict_consistent(reward_by_key, accelerator):
    """Gather reward arrays across ranks with the same keys, order, and shapes on every rank.

    Mixed-source batches can produce source-specific detail keys on only some ranks. Keys missing on a rank
    are filled with -10, which logging ignores; the optimized reward `avg` is present on every rank.
    """
    reference = reward_by_key["ori_avg"] if "ori_avg" in reward_by_key else reward_by_key["avg"]
    local_count = int(reference.shape[0])
    local_tensors = {}
    for key, value in reward_by_key.items():
        tensor = value.detach().to(accelerator.device) if torch.is_tensor(value) else torch.as_tensor(
            value, device=accelerator.device
        )
        if tensor.shape[0] != local_count:
            raise ValueError(f"reward gather: key '{key}' has {tensor.shape[0]} rows, expected {local_count}")
        local_tensors[str(key)] = tensor.float().contiguous()
    local_payload = {
        "count": local_count,
        "schema": {key: tuple(int(d) for d in t.shape[1:]) for key, t in local_tensors.items()},
    }
    if dist.is_available() and dist.is_initialized():
        payloads = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(payloads, local_payload)
    else:
        payloads = [local_payload]
    if len({int(p["count"]) for p in payloads}) != 1:
        raise RuntimeError("reward gather: inconsistent per-rank counts")
    key_shapes = {}
    for payload in payloads:
        for key, suffix in payload["schema"].items():
            if key_shapes.setdefault(key, tuple(suffix)) != tuple(suffix):
                raise RuntimeError(f"reward gather: key '{key}' has inconsistent shapes")
    ordered = [key for key in ("avg", "ori_avg") if key in key_shapes]
    ordered += sorted(key for key in key_shapes if key not in ordered)
    gathered = {}
    for key in ordered:
        tensor = local_tensors.get(key)
        if tensor is None:
            tensor = torch.full(
                (local_count, *key_shapes[key]), REWARD_MISSING_VALUE, device=accelerator.device
            )
        gathered[key] = accelerator.gather(tensor).detach().cpu().numpy()
    return gathered


def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length, device):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt(text_encoders, tokenizers, prompt, max_sequence_length)
    return prompt_embeds.to(device), pooled_prompt_embeds.to(device)


def calculate_zero_std_ratio(prompts, gathered_rewards):
    """Fraction of prompts whose rewards have zero standard deviation, and the mean per-prompt std."""
    prompt_array = np.array(prompts)
    _, inverse_indices, counts = np.unique(prompt_array, return_inverse=True, return_counts=True)
    grouped_rewards = gathered_rewards["ori_avg"][np.argsort(inverse_indices)]
    reward_groups = np.split(grouped_rewards, np.cumsum(counts)[:-1])
    prompt_std_devs = np.array([np.std(group) for group in reward_groups])
    return np.count_nonzero(prompt_std_devs == 0) / len(prompt_std_devs), prompt_std_devs.mean()


def create_generator(prompts, base_seed):
    generators = []
    for prompt in prompts:
        prompt_hash_int = int.from_bytes(hashlib.sha256(prompt.encode()).digest()[:4], "big")
        generators.append(torch.Generator().manual_seed((base_seed + prompt_hash_int) % (2**31)))
    return generators


def compute_log_prob(transformer, pipeline, sample, j, embeds, pooled_embeds, config):
    if config.train.cfg:
        noise_pred = transformer(
            hidden_states=torch.cat([sample["latents"][:, j]] * 2),
            timestep=torch.cat([sample["timesteps"][:, j]] * 2),
            encoder_hidden_states=embeds,
            pooled_projections=pooled_embeds,
            return_dict=False,
        )[0]
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + config.sample.guidance_scale * (noise_pred_text - noise_pred_uncond)
    else:
        noise_pred = transformer(
            hidden_states=sample["latents"][:, j],
            timestep=sample["timesteps"][:, j],
            encoder_hidden_states=embeds,
            pooled_projections=pooled_embeds,
            return_dict=False,
        )[0]

    # log prob of next_latents given latents under the current model
    return sde_step_with_logprob(
        pipeline.scheduler,
        noise_pred.float(),
        sample["timesteps"][:, j],
        sample["latents"][:, j].float(),
        prev_sample=sample["next_latents"][:, j].float(),
        noise_level=config.sample.noise_level,
    )


def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    return model._orig_mod if is_compiled_module(model) else model


def _capture_rng_state(accelerator):
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (numpy_state[0], numpy_state[1].tolist(), numpy_state[2], numpy_state[3], numpy_state[4]),
        "torch_cpu": torch.get_rng_state().cpu(),
        "torch_cuda": torch.cuda.get_rng_state(accelerator.device).cpu() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(rng_state, accelerator):
    random.setstate(rng_state["python"])
    numpy_state = rng_state["numpy"]
    np.random.set_state(
        (numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), numpy_state[2], numpy_state[3], numpy_state[4])
    )
    torch.set_rng_state(rng_state["torch_cpu"])
    if torch.cuda.is_available() and rng_state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state(rng_state["torch_cuda"], device=accelerator.device)


def save_ckpt(save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config,
              optimizer, epoch, stat_tracker, wandb_run_id):
    """Save the (EMA) LoRA weights plus the optimizer, EMA, stat-tracker, and RNG state needed to resume."""
    save_root = os.path.join(save_dir, "checkpoints", f"checkpoint-{global_step}")
    save_root_lora = os.path.join(save_root, "lora")
    if accelerator.is_main_process:
        os.makedirs(save_root_lora, exist_ok=True)
    accelerator.wait_for_everyone()

    local_rng_state = _capture_rng_state(accelerator)
    if dist.is_available() and dist.is_initialized():
        rng_states_by_rank = [None] * accelerator.num_processes if accelerator.is_main_process else None
        dist.gather_object(local_rng_state, rng_states_by_rank, dst=0)
    else:
        rng_states_by_rank = [local_rng_state]

    if accelerator.is_main_process:
        if config.train.ema:
            ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
        unwrap_model(transformer, accelerator).save_pretrained(save_root_lora)
        if config.train.ema:
            ema.copy_temp_to(transformer_trainable_parameters)
        training_state = {
            "global_step": global_step,
            "epoch": epoch,
            "wandb_run_id": wandb_run_id,
            "rng_world_size": accelerator.num_processes,
            "rng_states_by_rank": rng_states_by_rank,
        }
        if config.train.ema:
            training_state["ema_state_dict"] = ema.state_dict()
        if stat_tracker is not None:
            training_state["stat_tracker"] = {
                "stats": {k: v.tolist() if hasattr(v, "tolist") else list(v) for k, v in stat_tracker.stats.items()},
                "history_prompts": list(stat_tracker.history_prompts),
            }
        torch.save(training_state, os.path.join(save_root, "training_state.pt"))
        torch.save(optimizer.state_dict(), os.path.join(save_root, "optimizer.pt"))
        print(f"Saved checkpoint to {save_root} (epoch={epoch}, step={global_step})")


def _make_train_dataloader(config, accelerator):
    common = dict(
        batch_size=config.sample.train_batch_size,
        k=int(config.sample.num_image_per_prompt),
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        seed=SAMPLER_SEED,
        rollout_batches=config.sample.num_batches_per_epoch,
    )
    if config.prompt_fn == "text":
        dataset = TextPromptDataset(config.dataset, "train")
    elif config.prompt_fn == "metadata":
        dataset = MetadataPromptDataset(config.dataset, "train")
    else:
        raise ValueError(f"unknown prompt_fn {config.prompt_fn!r}")

    balance = config.get("train_source_balance")
    if balance:
        key = str(balance.key)
        weights = dict(balance.weights)
        source_indices = {source: [] for source in weights}
        unknown = set()
        for index, metadata in enumerate(dataset.metadatas):
            source = metadata.get(key)
            if source in source_indices:
                source_indices[source].append(index)
            else:
                unknown.add(str(source))
        if unknown:
            raise ValueError(f"dataset has {key} values outside train_source_balance.weights: {sorted(unknown)}")
        sampler = DistributedKRepeatSourceBalancedSampler(
            dataset=dataset,
            source_indices=source_indices,
            source_weights=weights,
            rotate_remainder=bool(balance.get("rotate_remainder", False)),
            **common,
        )
        if accelerator.is_main_process:
            print(
                f"Source-balanced sampler: pools={ {s: len(i) for s, i in source_indices.items()} } "
                f"unique_per_batch={sampler.source_counts_per_batch} rotate_remainder={sampler.rotate_remainder}",
                flush=True,
            )
    else:
        sampler = DistributedKRepeatSampler(dataset=dataset, **common)

    # The sampler state is set before each rollout batch, so the loader must not prefetch in workers.
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0, collate_fn=dataset.collate_fn)
    return sampler, loader


def main(_):
    config = FLAGS.config
    wandb = _wandb()

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    config.run_name = f"{config.run_name}_{unique_id}" if config.run_name else unique_id
    if not config.save_dir:
        config.save_dir = os.path.join(config.logdir, config.run_name)

    # number of timesteps within each trajectory to train on
    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)

    from accelerate import InitProcessGroupKwargs

    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        project_config=ProjectConfiguration(
            project_dir=os.path.join(config.logdir, config.run_name),
            automatic_checkpoint_naming=True,
            total_limit=config.num_checkpoint_limit,
        ),
        # Gradients accumulate over timesteps too, so `gradient_accumulation_steps` counts samples.
        gradient_accumulation_steps=config.train.gradient_accumulation_steps * num_train_timesteps,
        kwargs_handlers=[
            InitProcessGroupKwargs(timeout=datetime.timedelta(minutes=120)),
            DistributedDataParallelKwargs(broadcast_buffers=False, find_unused_parameters=False),
        ],
    )
    wandb_run_id = None
    resume_wandb_id = None
    if config.train.lora_path:
        state_path = os.path.join(os.path.dirname(config.train.lora_path.rstrip("/")), "training_state.pt")
        if os.path.exists(state_path):
            resume_wandb_id = torch.load(state_path, map_location="cpu").get("wandb_run_id")
    if accelerator.is_main_process and wandb is not None:
        if resume_wandb_id:
            run = wandb.init(project=config.wandb_project, id=resume_wandb_id, resume="allow")
        else:
            run = wandb.init(project=config.wandb_project, name=config.run_name, config=config.to_dict())
        wandb_run_id = run.id
    accelerator.wait_for_everyone()
    logger.info(f"\n{config}")

    def log(metrics, step):
        if accelerator.is_main_process and wandb is not None:
            wandb.log(metrics, step=step)

    set_seed(config.seed, device_specific=True)

    pipeline = StableDiffusion3Pipeline.from_pretrained(config.pretrained.model)
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.text_encoder_3.requires_grad_(False)
    pipeline.transformer.requires_grad_(not config.use_lora)
    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3]
    pipeline.safety_checker = None
    pipeline.set_progress_bar_config(
        position=1, disable=not accelerator.is_local_main_process, leave=False, desc="Timestep", dynamic_ncols=True
    )

    # Frozen weights only run inference, so cast them to half precision.
    inference_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(accelerator.mixed_precision, torch.float32)
    pipeline.vae.to(accelerator.device, dtype=torch.float32)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder_2.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder_3.to(accelerator.device, dtype=inference_dtype)
    pipeline.transformer.to(accelerator.device)

    if config.use_lora:
        transformer_lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            init_lora_weights="gaussian",
            target_modules=[
                "attn.add_k_proj",
                "attn.add_q_proj",
                "attn.add_v_proj",
                "attn.to_add_out",
                "attn.to_k",
                "attn.to_out.0",
                "attn.to_q",
                "attn.to_v",
            ],
        )
        if config.train.lora_path:
            pipeline.transformer = PeftModel.from_pretrained(pipeline.transformer, config.train.lora_path)
            # PeftModel.from_pretrained freezes the adapter; set_adapter makes it trainable again.
            pipeline.transformer.set_adapter("default")
        else:
            pipeline.transformer = get_peft_model(pipeline.transformer, transformer_lora_config)

    if config.activation_checkpointing:
        _enable_transformer_gradient_checkpointing(pipeline.transformer)

    transformer = pipeline.transformer
    transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    # This EMA setting averages over the previous 20 x 8 = 160 steps.
    ema = EMAModuleWrapper(transformer_trainable_parameters, decay=0.9, update_step_interval=8, device=accelerator.device)

    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if config.train.use_8bit_adam:
        import bitsandbytes as bnb

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(
        transformer_trainable_parameters,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    reward_fn = rlvvr.rewards.multi_score(accelerator.device, dict(config.reward_fn), list(config.mixed_sources))
    train_sampler, train_dataloader = _make_train_dataloader(config, accelerator)

    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings(
        [""], text_encoders, tokenizers, max_sequence_length=128, device=accelerator.device
    )
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.train_batch_size, 1, 1)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.train_batch_size, 1)
    train_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.train.batch_size, 1)

    if config.sample.num_image_per_prompt == 1:
        config.per_prompt_stat_tracking = False
    stat_tracker = PerPromptStatTracker(config.sample.global_std) if config.per_prompt_stat_tracking else None

    # Autocast is needed for full fine-tuning; LoRA training does not need it and it costs memory.
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast

    transformer, optimizer = accelerator.prepare(transformer, optimizer)

    # Rewards are computed in background threads while sampling continues.
    executor = futures.ThreadPoolExecutor(max_workers=int(os.environ.get("REWARD_EXECUTOR_MAX_WORKERS", "8")))

    samples_per_epoch = config.sample.train_batch_size * accelerator.num_processes * config.sample.num_batches_per_epoch
    total_train_batch_size = (
        config.train.batch_size * accelerator.num_processes * config.train.gradient_accumulation_steps
    )
    logger.info("***** Running training *****")
    logger.info(f"  Sample batch size per device = {config.sample.train_batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(f"  Gradient accumulation steps = {config.train.gradient_accumulation_steps}")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
    logger.info(f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}")

    epoch = 0
    global_step = 0
    train_iter = iter(train_dataloader)

    if config.train.lora_path:
        # lora_path points to checkpoint-*/lora; the training state is one level up.
        ckpt_root = os.path.dirname(config.train.lora_path.rstrip("/"))
        training_state_path = os.path.join(ckpt_root, "training_state.pt")
        if os.path.exists(training_state_path):
            training_state = torch.load(training_state_path, map_location="cpu")
            epoch = training_state["epoch"]
            global_step = training_state["global_step"]
            if config.train.ema and "ema_state_dict" in training_state:
                ema.load_state_dict(training_state["ema_state_dict"])
            if stat_tracker is not None and "stat_tracker" in training_state:
                st_data = training_state["stat_tracker"]
                stat_tracker.stats = {k: np.array(v) for k, v in st_data["stats"].items()}
                stat_tracker.history_prompts = set(st_data.get("history_prompts", []))
            optimizer_state_path = os.path.join(ckpt_root, "optimizer.pt")
            if os.path.exists(optimizer_state_path):
                optimizer.load_state_dict(torch.load(optimizer_state_path, map_location="cpu"))
            # Fast-forward the prompt sampler.
            for e in range(epoch):
                for i in range(config.sample.num_batches_per_epoch):
                    train_sampler.set_epoch(e * config.sample.num_batches_per_epoch + i)
                    next(train_iter)
            rng_states_by_rank = training_state.get("rng_states_by_rank")
            if rng_states_by_rank is not None and len(rng_states_by_rank) == accelerator.num_processes:
                _restore_rng_state(rng_states_by_rank[accelerator.process_index], accelerator)
            if accelerator.is_main_process:
                print(f"Resumed from checkpoint: epoch={epoch}, global_step={global_step}")

    while True:
        if epoch % config.save_freq == 0 and epoch > 0:
            save_ckpt(config.save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters,
                      config, optimizer, epoch, stat_tracker, wandb_run_id)
        accelerator.wait_for_everyone()
        if config.max_global_step is not None and global_step >= int(config.max_global_step):
            if accelerator.is_main_process:
                print(f"Reached max_global_step={config.max_global_step} at epoch={epoch}; stopping.", flush=True)
                if wandb is not None:
                    wandb.finish()
            return

        #################### SAMPLING ####################
        pipeline.transformer.eval()
        samples = []
        local_sample_prompts = []
        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)
            prompts, prompt_metadata = next(train_iter)
            local_sample_prompts.extend(prompts)

            prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
                prompts, text_encoders, tokenizers, max_sequence_length=128, device=accelerator.device
            )
            generator = create_generator(prompts, base_seed=epoch * 10000 + i) if config.sample.same_latent else None
            with autocast():
                with torch.no_grad():
                    images, latents, log_probs = pipeline_with_logprob(
                        pipeline,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        negative_prompt_embeds=sample_neg_prompt_embeds,
                        negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds,
                        num_inference_steps=config.sample.num_steps,
                        guidance_scale=config.sample.guidance_scale,
                        output_type="pt",
                        height=config.resolution,
                        width=config.resolution,
                        noise_level=config.sample.noise_level,
                        generator=generator,
                    )

            latents = torch.stack(latents, dim=1)  # (batch_size, num_steps + 1, 16, 96, 96)
            log_probs = torch.stack(log_probs, dim=1)  # (batch_size, num_steps)
            timesteps = pipeline.scheduler.timesteps.repeat(config.sample.train_batch_size, 1)

            # compute rewards asynchronously
            rewards = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=True)
            time.sleep(0)  # yield so reward computation starts

            samples.append(
                {
                    "prompt_embeds": prompt_embeds,
                    "pooled_prompt_embeds": pooled_prompt_embeds,
                    "timesteps": timesteps,
                    "latents": latents[:, :-1],  # latent before timestep t
                    "next_latents": latents[:, 1:],  # latent after timestep t
                    "log_probs": log_probs,
                    "rewards": rewards,
                }
            )

        for sample in tqdm(samples, desc="Waiting for rewards", disable=not accelerator.is_local_main_process, position=0):
            rewards, _ = sample["rewards"].result()
            expected_len = int(sample["prompt_embeds"].shape[0])
            sample["rewards"] = {
                key: torch.as_tensor(
                    _reward_values_to_numpy(value, key=key, expected_len=expected_len), device=accelerator.device
                ).float()
                for key, value in rewards.items()
            }

        # Collate samples into a dict of (num_batches_per_epoch * batch_size, ...) tensors. A detail key that a
        # batch lacks (e.g. a VVR detail in an all-OCR batch) is filled with -10.
        collated = {}
        for k in samples[0].keys():
            if not isinstance(samples[0][k], dict):
                collated[k] = torch.cat([s[k] for s in samples], dim=0)
                continue
            sub_keys = sorted({sub_key for s in samples for sub_key in s[k].keys()})
            collated_sub = {}
            for sub_key in sub_keys:
                parts = []
                for s in samples:
                    if sub_key in s[k]:
                        parts.append(s[k][sub_key])
                    else:
                        ref = next(iter(s[k].values()))
                        parts.append(torch.full(ref.shape, REWARD_MISSING_VALUE, device=ref.device, dtype=ref.dtype))
                collated_sub[sub_key] = torch.cat(parts, dim=0)
            collated[k] = collated_sub
        samples = collated

        samples["rewards"]["ori_avg"] = samples["rewards"]["avg"]
        # Repeat the reward along the timestep dimension so timestep-dependent advantages can be added later.
        samples["rewards"]["avg"] = samples["rewards"]["avg"].unsqueeze(1).repeat(1, num_train_timesteps)
        gathered_rewards = _gather_reward_dict_consistent(samples["rewards"], accelerator)
        gathered_prompts = _gather_prompt_group_keys(local_sample_prompts, accelerator)
        if len(gathered_prompts) != gathered_rewards["avg"].shape[0]:
            raise RuntimeError("gathered prompt count does not match gathered rewards")

        if accelerator.is_main_process:
            reward_log = {}
            for key, value in gathered_rewards.items():
                valid = value[value != REWARD_MISSING_VALUE]
                reward_log[f"reward_{key}"] = float(np.mean(valid)) if valid.size else float("nan")
            log({"epoch": epoch, **reward_log}, global_step)
            log_metrics_local(config.save_dir, {"type": "train_reward", "epoch": epoch, **reward_log}, global_step)

        # per-prompt mean/std tracking
        if stat_tracker is not None:
            advantages = stat_tracker.update(gathered_prompts, gathered_rewards["avg"])
            group_size, trained_prompt_num = stat_tracker.get_stats()
            zero_std_ratio, reward_std_mean = calculate_zero_std_ratio(gathered_prompts, gathered_rewards)
            log(
                {
                    "group_size": group_size,
                    "trained_prompt_num": trained_prompt_num,
                    "zero_std_ratio": zero_std_ratio,
                    "reward_std_mean": reward_std_mean,
                },
                global_step,
            )
            stat_tracker.clear()
        else:
            avg = gathered_rewards["avg"]
            advantages = (avg - avg.mean()) / (avg.std() + 1e-4)

        # keep only the advantages of this process's samples
        advantages = torch.as_tensor(advantages)
        samples["advantages"] = (
            advantages.reshape(accelerator.num_processes, -1, advantages.shape[-1])[accelerator.process_index]
            .to(accelerator.device)
        )
        del samples["rewards"]

        # Drop samples whose advantage is zero at every timestep, keeping a multiple of num_batches_per_epoch.
        mask = samples["advantages"].abs().sum(dim=1) != 0
        num_batches = config.sample.num_batches_per_epoch
        true_count = mask.sum()
        if true_count == 0:
            # Keep every rank in the same collective sequence; these samples carry zero policy signal.
            mask[torch.arange(len(mask), device=mask.device)[:num_batches]] = True
            true_count = mask.sum()
        if true_count % num_batches != 0:
            false_indices = torch.where(~mask)[0]
            num_to_change = num_batches - (true_count % num_batches)
            if len(false_indices) >= num_to_change:
                mask[false_indices[torch.randperm(len(false_indices))[:num_to_change]]] = True
        log({"actual_batch_size": mask.sum().item() // num_batches}, global_step)
        samples = {k: v[mask] for k, v in samples.items()}

        total_batch_size, num_timesteps = samples["timesteps"].shape
        assert num_timesteps == config.sample.num_steps

        #################### TRAINING ####################
        for inner_epoch in range(config.train.num_inner_epochs):
            perm = torch.randperm(total_batch_size, device=accelerator.device)
            samples = {k: v[perm] for k, v in samples.items()}
            samples_batched = {
                k: v.reshape(-1, total_batch_size // num_batches, *v.shape[1:]) for k, v in samples.items()
            }
            samples_batched = [dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())]

            pipeline.transformer.train()
            info = defaultdict(list)
            for i, sample in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not accelerator.is_local_main_process,
            ):
                if config.train.cfg:
                    # concatenate negative prompts to avoid a second forward pass
                    embeds = torch.cat([train_neg_prompt_embeds[: len(sample["prompt_embeds"])], sample["prompt_embeds"]])
                    pooled_embeds = torch.cat(
                        [train_neg_pooled_prompt_embeds[: len(sample["pooled_prompt_embeds"])], sample["pooled_prompt_embeds"]]
                    )
                else:
                    embeds = sample["prompt_embeds"]
                    pooled_embeds = sample["pooled_prompt_embeds"]

                for j in tqdm(
                    range(num_train_timesteps),
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not accelerator.is_local_main_process,
                ):
                    with accelerator.accumulate(transformer):
                        with autocast():
                            _, log_prob, prev_sample_mean, std_dev_t = compute_log_prob(
                                transformer, pipeline, sample, j, embeds, pooled_embeds, config
                            )
                            if config.train.beta > 0:
                                with torch.no_grad():
                                    with accelerator.unwrap_model(transformer).disable_adapter():
                                        _, _, prev_sample_mean_ref, _ = compute_log_prob(
                                            transformer, pipeline, sample, j, embeds, pooled_embeds, config
                                        )

                        # GRPO loss
                        advantages = torch.clamp(
                            sample["advantages"][:, j], -config.train.adv_clip_max, config.train.adv_clip_max
                        )
                        log_ratio = log_prob - sample["log_probs"][:, j]
                        log_ratio = torch.nan_to_num(log_ratio.float(), nan=0.0, posinf=20.0, neginf=-20.0).clamp(
                            -20.0, 20.0
                        )
                        ratio = torch.exp(log_ratio)
                        unclipped_loss = -advantages * ratio
                        clipped_loss = -advantages * torch.clamp(
                            ratio, 1.0 - config.train.clip_range, 1.0 + config.train.clip_range
                        )
                        policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
                        if config.train.beta > 0:
                            kl_loss = ((prev_sample_mean - prev_sample_mean_ref) ** 2).mean(
                                dim=(1, 2, 3), keepdim=True
                            ) / (2 * std_dev_t**2)
                            kl_loss = torch.mean(kl_loss)
                            loss = policy_loss + config.train.beta * kl_loss
                        else:
                            loss = policy_loss

                        info["approx_kl"].append(0.5 * torch.mean(log_ratio**2))
                        info["clipfrac"].append(torch.mean((torch.abs(ratio - 1.0) > config.train.clip_range).float()))
                        info["policy_loss"].append(_safe_info_value(policy_loss))
                        if config.train.beta > 0:
                            info["kl_loss"].append(_safe_info_value(kl_loss))

                        # Skip the update on every rank if any rank has a non-finite value.
                        finite_checks = [loss, policy_loss, log_prob, prev_sample_mean, std_dev_t]
                        if config.train.beta > 0:
                            finite_checks.extend([kl_loss, prev_sample_mean_ref])
                        local_nonfinite = torch.zeros((), device=accelerator.device, dtype=torch.float32)
                        for value in finite_checks:
                            local_nonfinite = torch.maximum(
                                local_nonfinite, (~torch.isfinite(value.detach()).all()).float()
                            )
                        global_nonfinite = local_nonfinite.clone()
                        if dist.is_available() and dist.is_initialized():
                            dist.all_reduce(global_nonfinite, op=dist.ReduceOp.MAX)
                        skip_nonfinite = bool(global_nonfinite.item() > 0)
                        info["loss"].append(_safe_info_value(loss))
                        info["skipped_nonfinite_step"].append(
                            torch.tensor(float(skip_nonfinite), device=accelerator.device)
                        )

                        if skip_nonfinite:
                            optimizer.zero_grad(set_to_none=True)
                            if accelerator.is_main_process:
                                logger.warning(f"Epoch {epoch} batch {i} timestep {j}: non-finite loss; skipped update.")
                        else:
                            accelerator.backward(loss)
                            if accelerator.sync_gradients:
                                accelerator.clip_grad_norm_(transformer.parameters(), config.train.max_grad_norm)
                            optimizer.step()
                            optimizer.zero_grad()

                    # an optimizer step happened behind the scenes
                    if accelerator.sync_gradients:
                        info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        info = accelerator.reduce(info, reduction="mean")
                        info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        log(info, global_step)
                        if accelerator.is_main_process:
                            log_metrics_local(
                                config.save_dir,
                                {"type": "train", **{k: float(v) for k, v in info.items()}},
                                global_step,
                            )
                        global_step += 1
                        info = defaultdict(list)
                if config.train.ema:
                    ema.step(transformer_trainable_parameters, global_step)

        epoch += 1


if __name__ == "__main__":
    app.run(main)
