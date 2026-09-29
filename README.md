# VVRBench

Code for *Verifiable Visual Rewards Transfer from Synthetic Scenes to Natural Prompts*.

VVR (verifiable visual rewards) scores a generated image against a formal task specification with deterministic program verifiers: no learned reward model and no reference image. This repository contains

- `vvr_bench/`: the VVRBench evaluator and the verifier library, and
- RLVVR training code for text-to-image post-training with VVR rewards (being added).

Data: [huggingface.co/datasets/stellalisy/VVRBench](https://huggingface.co/datasets/stellalisy/VVRBench)

| Configuration | Split | Tasks | Use |
|---|---|---|---|
| `default` (VVRBench) | test | 10,000 | Main benchmark, complexity 3 to 48 |
| `fast` (VVRBench-Fast) | test | 820 | 20 tasks at each attainable integer complexity from 3 to 44 |
| `challenge` (VVRBench-Challenge) | test | 720 | 20 tasks at each integer complexity from 45 to 80 |
| `vvr_easy` | train | 100,000 | RLVVR training, complexity at most 20 |
| `vvr_matched` | train | 100,000 | RLVVR training, matched to the VVRBench distribution |

## Install

```bash
git clone https://github.com/stellalisy/VVRBench.git
cd VVRBench
pip install -e .
```

## Evaluate a model on VVRBench

1. Generate one image per task and save it as `{id}.png`, for example `vvr_bench_00000.png`. Alternatively, write a JSONL manifest with `id` and `image_path` fields.

   ```python
   from datasets import load_dataset

   tasks = load_dataset("stellalisy/VVRBench", split="test")  # or "fast", "challenge" as the config name
   for task in tasks:
       image = generate(task["prompt"])  # your model
       image.save(f"images/{task['id']}.png")
   ```

2. Score the images. The evaluator downloads the pinned benchmark data and checks its hash and the verifier's hash.

   ```bash
   vvr-bench-evaluate --predictions-dir images --output-dir results --require-complete
   vvr-bench-evaluate --config fast --predictions-dir images_fast --output-dir results_fast --require-complete
   vvr-bench-evaluate --config challenge --predictions-dir images_challenge --output-dir results_challenge --require-complete
   ```

The evaluator writes `summary.json` and `per_example.jsonl`. The main metric is accuracy: the fraction of tasks whose image satisfies every constraint.

| Field | Meaning |
|---|---|
| `strict_accuracy` | Accuracy over the tasks that have an image |
| `strict_accuracy_missing_as_failure` | Accuracy with missing images counted as failures, as in the paper |
| `strict_by_difficulty` | Accuracy by complexity range: `1` to `5` are the VVRBench ranges C1 to C5 on VVRBench and VVRBench-Fast, and five equal-width ranges from 45 to 80 on VVRBench-Challenge (the paper's VVRBench-Fast and Challenge tables use their own integer ranges) |
| `strict_by_criterion` | Pass rate of each constraint type |
| `dense_vvr`, `by_difficulty`, `by_family`, `by_criterion` | Partial-credit scores; do not report these as accuracy |

Pass `--require-complete` to stop when an image is missing.

The evaluator scores each image at the resolution you save it. In the paper, open-weight and post-trained models were scored on VVRBench after resizing to 512×512; API models on VVRBench-Fast and all models on VVRBench-Challenge were scored at 1024×1024.

## The verifier

`vvr_bench/verifier.py` extracts objects from the image with fixed pixel operations and checks each constraint with a program verifier. `score_image_spec(image, spec)` returns a score and a dictionary of per-constraint results; `strict` in that dictionary is the pass-or-fail decision used for accuracy. [`VERIFIER.md`](VERIFIER.md) lists the 46 constraint types and the validation record.

## License

MIT; see [`LICENSE`](LICENSE).
