"""Evaluate generated images on VVR-Bench."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image

from . import __version__, verifier as verifier_module
from .verifier import score_image_spec
from .taxonomy import ATOMIC_TO_FAMILY, canonical_families, family_count_tier


REPO_ID = "stellalisy/VVRBench"
DATA_FILES = {
    "default": "data/test-00000-of-00001.parquet",
    "challenge": "data/challenge-00000-of-00001.parquet",
    "fast": "data/fast-00000-of-00001.parquet",
}
DATASET_REVISIONS = {
    "default": "7a79a160060e15a9af52c3737e0ad1a83186804c",
    "challenge": "7a79a160060e15a9af52c3737e0ad1a83186804c",
    "fast": "51c69da829f13fe705b8393606e22d84b87b8177",
}
DATASET_VERSIONS = {"default": "1.3", "challenge": "2.0", "fast": "1.0"}
DATA_FILE_SHA256 = {
    "default": "c246cabb6e20e6de5b32c7b4054eac144144cdd18869dcb29791ced09476c89f",
    "challenge": "3b6c6baf5318842e5a113116fe4ff28a0442b70c35b38669d3dbb28d57989c45",
    "fast": "2cf0fac7671d74fd01f5ba5aca435e79755f39f323b206260c031c42a59ab035",
}
VERIFIER_SHA256 = "2c3661be030c078e69c53284949524b4473150c67fd23bb294ffaafd7fa64f77"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar(details: dict[str, Any], key: str) -> float:
    value = details.get(key, [0.0])
    if isinstance(value, list):
        value = value[0] if value else 0.0
    return float(value)


def criterion_score(criterion: str, details: dict[str, Any]) -> float:
    key = f"visual_logic_predicate_{criterion}"
    if key in details:
        return scalar(details, key)
    fallback = {
        "exact_count": "visual_logic_object_count_score",
        "color_attribute": "visual_logic_object_count_score",
        "shape_attribute": "visual_logic_shape_score",
        "color_shape_binding": "visual_logic_shape_score",
        "absolute_region": "visual_logic_layout_score",
        "grid_occupancy": "visual_logic_layout_score",
    }
    return scalar(details, fallback[criterion])


def criterion_strict(criterion: str, details: dict[str, Any]) -> float | None:
    """The verifier's pass-or-fail decision for one constraint type, when it reports one."""
    key = f"visual_logic_predicate_{criterion}_strict"
    return scalar(details, key) if key in details else None


def image_index(directory: Path | None, manifest: Path | None) -> dict[str, Path]:
    result: dict[str, Path] = {}
    if directory:
        for extension in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
            for path in directory.glob(extension):
                result[path.stem] = path
    if manifest:
        base = manifest.parent
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            path = Path(row["image_path"])
            result[str(row["id"])] = path if path.is_absolute() else base / path
    return result


def evaluate_one(task: tuple[dict[str, Any], str]) -> dict[str, Any]:
    row, path_text = task
    image = np.asarray(Image.open(path_text).convert("RGB"))
    spec = json.loads(row["visual_spec_json"])
    dense, details = score_image_spec(image, spec)
    criteria = list(row["active_criteria"])
    predicate_scores = {criterion: criterion_score(criterion, details) for criterion in criteria}
    family_scores: dict[str, list[float]] = defaultdict(list)
    for criterion in criteria:
        family_scores[ATOMIC_TO_FAMILY[criterion]].append(predicate_scores[criterion])
    active_families = canonical_families(criteria)
    return {
        "id": row["id"],
        "prompt": row["prompt"],
        "tier": family_count_tier(len(active_families)),
        "family_count": len(active_families),
        "difficulty_bin": int(row["difficulty_bin"]),
        "active_families": active_families,
        "active_criteria": criteria,
        "construction_groups": list(row["active_families"]),
        "construction_tier": row["tier"],
        "dense_vvr": float(dense),
        "strict": scalar(details, "visual_logic_strict"),
        "valid_spec": scalar(details, "visual_logic_valid_spec"),
        "extra_components": scalar(details, "visual_logic_extra_components"),
        "predicate_scores": predicate_scores,
        "predicate_strict": {
            criterion: value
            for criterion in criteria
            if (value := criterion_strict(criterion, details)) is not None
        },
        "family_scores": {family: float(np.mean(values)) for family, values in family_scores.items()},
    }


def means(groups: dict[str, list[float]]) -> dict[str, float]:
    return {key: float(np.mean(values)) for key, values in sorted(groups.items()) if values}


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="Local benchmark parquet; downloaded from Hugging Face when omitted")
    parser.add_argument("--config", choices=sorted(DATA_FILES), default="default")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--predictions-dir", type=Path)
    source.add_argument("--predictions-manifest", type=Path, help="JSONL rows with id and image_path")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument(
        "--allow-unpinned-data",
        action="store_true",
        help="Permit a custom dataset whose hash differs from the frozen release",
    )
    args = parser.parse_args()

    data_path = args.data or Path(
        hf_hub_download(
            REPO_ID,
            DATA_FILES[args.config],
            repo_type="dataset",
            revision=DATASET_REVISIONS[args.config],
        )
    )
    data_sha256 = sha256(data_path)
    if not args.allow_unpinned_data and data_sha256 != DATA_FILE_SHA256[args.config]:
        raise SystemExit(
            f"dataset hash mismatch for {args.config}: {data_sha256}; "
            f"expected {DATA_FILE_SHA256[args.config]}"
        )
    verifier_sha256 = sha256(Path(verifier_module.__file__))
    if verifier_sha256 != VERIFIER_SHA256:
        raise SystemExit(
            f"verifier hash mismatch: {verifier_sha256}; expected {VERIFIER_SHA256}"
        )
    rows = pq.read_table(data_path).to_pylist()
    images = image_index(args.predictions_dir, args.predictions_manifest)
    missing = [row["id"] for row in rows if row["id"] not in images]
    if missing and args.require_complete:
        raise SystemExit(f"missing {len(missing)} predictions; first: {missing[:5]}")
    available = [row for row in rows if row["id"] in images]
    tasks = [(row, str(images[row["id"]])) for row in available]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        scored = list(executor.map(evaluate_one, tasks, chunksize=4))

    by_tier: dict[str, list[float]] = defaultdict(list)
    by_difficulty: dict[str, list[float]] = defaultdict(list)
    by_family: dict[str, list[float]] = defaultdict(list)
    by_criterion: dict[str, list[float]] = defaultdict(list)
    strict_by_difficulty: dict[str, list[float]] = defaultdict(list)
    strict_by_criterion: dict[str, list[float]] = defaultdict(list)
    for row in scored:
        by_tier[row["tier"]].append(row["dense_vvr"])
        by_difficulty[str(row["difficulty_bin"])].append(row["dense_vvr"])
        for family, value in row["family_scores"].items():
            by_family[family].append(value)
        for criterion, value in row["predicate_scores"].items():
            by_criterion[criterion].append(value)
        strict_by_difficulty[str(row["difficulty_bin"])].append(row["strict"])
        for criterion, value in row["predicate_strict"].items():
            strict_by_criterion[criterion].append(value)

    summary = {
        "benchmark": "VVR-Bench-Fast" if args.config == "fast" else "VVR-Bench",
        "version": DATASET_VERSIONS[args.config],
        "evaluator_version": __version__,
        "config": args.config,
        "data_sha256": data_sha256,
        "verifier_sha256": verifier_sha256,
        "taxonomy": "grounding-cardinality-spatial-size-topology-v1",
        "expected": len(rows),
        "evaluated": len(scored),
        "missing": len(missing),
        "dense_vvr": float(np.mean([row["dense_vvr"] for row in scored])) if scored else None,
        "strict_accuracy": float(np.mean([row["strict"] for row in scored])) if scored else None,
        "strict_accuracy_missing_as_failure": float(sum(row["strict"] for row in scored) / len(rows)) if rows else None,
        "strict_by_difficulty": means(strict_by_difficulty),
        "strict_by_criterion": means(strict_by_criterion),
        "family_macro": float(np.mean(list(means(by_family).values()))) if by_family else None,
        "by_tier": means(by_tier),
        "by_difficulty": means(by_difficulty),
        "by_family": means(by_family),
        "by_criterion": means(by_criterion),
        "missing_ids": missing,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "per_example.jsonl", scored)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("evaluated", "missing", "strict_accuracy", "strict_accuracy_missing_as_failure", "dense_vvr")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
