"""Configs for the nine SD3.5-M training runs in the paper.

Run one with `--config configs/rlvvr.py:<name>`, where <name> is one of `RUNS`. Paths in
`config.dataset` are the directories written by `scripts/prepare_data.py`.

The VVR runs (`vvr_easy`, `vvr_matched`, and the four mixtures) used 256 GPUs with gradient accumulation 1.
The three baselines (`geneval2`, `ocr`, `five_reward`) used 64 GPUs with gradient accumulation 4. Both give
eight unique prompts per GPU-batch of 24 images and 3,000 optimizer steps.
"""

import ml_collections


def base():
    config = ml_collections.ConfigDict()
    config.run_name = ""
    config.logdir = "logs"
    config.save_dir = ""
    config.seed = 0
    config.mixed_precision = "fp16"
    config.allow_tf32 = True
    config.use_lora = True
    config.activation_checkpointing = False
    config.num_checkpoint_limit = 5
    config.save_freq = 15  # epochs
    config.max_global_step = 3000
    config.per_prompt_stat_tracking = True
    config.wandb_project = "rlvvr"

    config.pretrained = pretrained = ml_collections.ConfigDict()
    pretrained.model = "stabilityai/stable-diffusion-3.5-medium"

    config.dataset = ""
    config.prompt_fn = "metadata"
    config.reward_fn = ml_collections.ConfigDict()
    config.mixed_sources = []
    config.train_source_balance = None

    config.sample = sample = ml_collections.ConfigDict()
    sample.num_steps = 25
    sample.guidance_scale = 4.5
    sample.noise_level = 0.7
    sample.train_batch_size = 3
    sample.num_image_per_prompt = 24
    sample.num_batches_per_epoch = 8
    sample.global_std = True
    sample.same_latent = False

    config.resolution = 512

    config.train = train = ml_collections.ConfigDict()
    train.batch_size = 3
    train.gradient_accumulation_steps = 1
    train.learning_rate = 3e-4
    train.adam_beta1 = 0.9
    train.adam_beta2 = 0.999
    train.adam_weight_decay = 1e-4
    train.adam_epsilon = 1e-8
    train.use_8bit_adam = False
    train.max_grad_norm = 1.0
    train.num_inner_epochs = 1
    train.cfg = True
    train.adv_clip_max = 5
    train.clip_range = 1e-4
    train.timestep_fraction = 0.99
    train.beta = 0.04
    train.ema = True
    train.lora_path = ""  # resume: path to checkpoints/checkpoint-*/lora
    return config


def _vvr_run(dataset, reward_fn, source_weights=None, rotate_remainder=False, mixed_sources=()):
    config = base()
    config.dataset = f"data/{dataset}"
    config.reward_fn = ml_collections.ConfigDict(reward_fn)
    config.mixed_sources = list(mixed_sources)
    if source_weights:
        config.train_source_balance = ml_collections.ConfigDict(
            {"key": "training_source", "weights": source_weights, "rotate_remainder": rotate_remainder}
        )
    return config


def _baseline_run(dataset, reward_fn, prompt_fn="metadata", mixed_sources=()):
    config = base()
    config.dataset = f"data/{dataset}"
    config.prompt_fn = prompt_fn
    config.reward_fn = ml_collections.ConfigDict(reward_fn)
    config.mixed_sources = list(mixed_sources)
    config.seed = 42
    config.save_freq = 30
    config.train.gradient_accumulation_steps = 4
    return config


_FIVE = ["geneval", "geneval2", "ocr", "pickscore", "unifiedreward"]


def vvr_easy():
    return _vvr_run("vvr_easy", {"vvr": 1.0})


def vvr_matched():
    return _vvr_run("vvr_matched", {"vvr": 1.0})


def geneval2_vvr_easy():
    return _vvr_run(
        "geneval2_vvr_easy",
        {"mixed": 1.0},
        source_weights={"geneval2": 1.0, "vvr_easy": 1.0},
        mixed_sources=["geneval2"],
    )


def geneval2_vvr_matched():
    return _vvr_run(
        "geneval2_vvr_matched",
        {"mixed": 1.0},
        source_weights={"geneval2": 1.0, "vvr_matched": 1.0},
        mixed_sources=["geneval2"],
    )


def ocr_vvr_easy():
    return _vvr_run(
        "ocr_vvr_easy",
        {"mixed": 1.0},
        source_weights={"ocr": 1.0, "vvr_easy": 1.0},
        mixed_sources=["ocr"],
    )


def five_reward_vvr_easy():
    return _vvr_run(
        "five_reward_vvr_easy",
        {"mixed": 1.0},
        source_weights={**{name: 1.0 for name in _FIVE}, "vvr_easy": 1.0},
        rotate_remainder=True,
        mixed_sources=_FIVE,
    )


def geneval2():
    return _baseline_run("geneval2", {"geneval2": 1.0})


def ocr():
    return _baseline_run("ocr", {"ocr": 1.0}, prompt_fn="text")


def five_reward():
    config = _baseline_run("five_reward", {"mixed": 1.0}, mixed_sources=_FIVE)
    config.activation_checkpointing = True
    return config


RUNS = [
    "vvr_easy",
    "vvr_matched",
    "geneval2_vvr_easy",
    "geneval2_vvr_matched",
    "ocr_vvr_easy",
    "five_reward_vvr_easy",
    "geneval2",
    "ocr",
    "five_reward",
]


def get_config(name):
    if name not in RUNS:
        raise ValueError(f"unknown run {name!r}; choose one of {RUNS}")
    config = globals()[name]()
    config.run_name = name
    return config
