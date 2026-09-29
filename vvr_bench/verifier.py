"""Frozen VVR image and video verifier library."""

from __future__ import annotations

import math
import re
from itertools import product
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


Color = Tuple[int, int, int]

COLOR_RGB: Dict[str, Color] = {
    "red": (220, 40, 40),
    "blue": (40, 90, 220),
    "green": (40, 170, 80),
    "yellow": (230, 210, 40),
    "purple": (140, 70, 200),
    "orange": (230, 130, 40),
}

VVR_BENCH_VERIFIER_V1 = "vvr_bench_v1"
VVR_BENCH_VERIFIER_V2 = "vvr_bench_v2"
VVR_BENCH_VERIFIER_V3 = "vvr_bench_v3"
VVR_BENCH_VERIFIER_V4 = "vvr_bench_v4"
# Older tags remain supported for exact reproduction of historical artifacts.
VVR_BENCH_VERIFIER_VERSION = VVR_BENCH_VERIFIER_V4
VVR_BENCH_VERIFIER_VERSIONS = (
    VVR_BENCH_VERIFIER_V1,
    VVR_BENCH_VERIFIER_V2,
    VVR_BENCH_VERIFIER_V3,
    VVR_BENCH_VERIFIER_V4,
)
VVR_BENCH_COLOR_RGB: Dict[str, Color] = {
    **COLOR_RGB,
    # Generated images commonly render saturated blue near RGB (0, 145, 240).
    # Keep that region on the blue side of the blue/cyan partition.
    "blue": (30, 125, 235),
    "cyan": (30, 190, 210),
    "pink": (235, 80, 150),
}

# Perceptual hue anchors are deliberately broader than the canonical RGB
# rendering palette. Diffusion models commonly render green as lime, blue as
# cyan-blue, and nominal orange/red objects with substantial shading.
VVR_BENCH_HUE_DEGREES: Dict[str, float] = {
    "red": 0.0,
    "orange": 30.0,
    "yellow": 58.0,
    # Image generators often render nominal green as lime. Centering this
    # category at 105 degrees keeps those objects on the green side of the
    # yellow/green boundary.
    "green": 105.0,
    "cyan": 180.0,
    "blue": 220.0,
    "purple": 275.0,
    "pink": 330.0,
}

VVR_BENCH_HUE_INTERVALS: Dict[str, Tuple[float, float]] = {
    "red": (345.0, 15.0),
    "orange": (15.0, 43.0),
    "yellow": (43.0, 75.0),
    "green": (75.0, 160.0),
    "cyan": (160.0, 200.0),
    "blue": (200.0, 250.0),
    "purple": (250.0, 310.0),
    "pink": (310.0, 345.0),
}

# V4 boundaries are calibrated against human judgments on diffusion outputs.
# The v1-v3 interval map above remains frozen for reproducibility.
VVR_BENCH_HUE_INTERVALS_V4: Dict[str, Tuple[float, float]] = {
    "red": (355.0, 15.0),
    "orange": (15.0, 43.0),
    "yellow": (43.0, 75.0),
    "green": (75.0, 170.0),
    "cyan": (170.0, 195.0),
    "blue": (195.0, 245.0),
    "purple": (245.0, 290.0),
    "pink": (290.0, 355.0),
}

BACKGROUND_RGB: Dict[str, Color] = {
    "white": (245, 245, 245),
    "black": (20, 20, 20),
    "light gray": (200, 200, 200),
    "dark gray": (60, 60, 60),
    "beige": (215, 190, 145),
    "pale pink": (245, 190, 205),
    "pale cyan": (180, 230, 230),
}

SUPPORTED_VERIFIERS = [
    "background_color",
    "object_count",
    "color_attribute",
    "shape_attribute",
    "color_shape_binding",
    "absolute_region",
    "grid_occupancy",
    "relative_position",
    "containment",
    "between_relation",
    "distance_relation",
    "size_attribute",
    "forbidden_color",
    "extra_object_penalty",
    "track_persistence",
    "motion_direction",
    "stationary_object",
    "velocity_range",
    "acceleration_trend",
    "size_change",
    "count_consistency",
    "trajectory_shape",
    "temporal_order",
    "relative_motion",
    "spatial_temporal_relation",
]

SUPPORTED_SHAPES = ["circle", "square"]
VVR_BENCH_SUPPORTED_SHAPES = ["circle", "square", "triangle"]
SUPPORTED_SIZES = ["small", "medium", "large"]
SUPPORTED_REGIONS = [
    "top_left",
    "top",
    "top_right",
    "left",
    "center",
    "right",
    "bottom_left",
    "bottom",
    "bottom_right",
]
SUPPORTED_RELATIONS = [
    "left_of",
    "right_of",
    "above",
    "below",
    "inside",
    "contains",
    "between",
    "near",
    "far",
    "not_touching",
]
VVR_BENCH_RELATIONS = [
    "same_count",
    "more_than_count",
    "fewer_than_count",
    "twice_as_many",
    "times_as_many",
    "larger_than",
    "smaller_than",
    "same_size",
    "all_larger_than",
    "all_smaller_than",
    "all_same_size",
    "not_all_same_size",
    "largest",
    "smallest",
    "closer_than",
    "farther_than",
    "touching",
    "separate",
    "same_row",
    "same_column",
    "all_same_row",
    "all_same_column",
    "not_all_same_row",
    "not_all_same_column",
    "leftmost",
    "rightmost",
    "topmost",
    "bottommost",
    "all_left_of",
    "all_right_of",
    "all_above",
    "all_below",
    "each_inside",
    "each_contains",
]

VVR_BENCH_FAMILY_REGISTRY: Dict[str, Dict[str, Any]] = {
    "quantity": {
        "criteria": [
            "exact_count",
            "same_count",
            "more_than_count",
            "fewer_than_count",
            "twice_as_many",
            "times_as_many",
        ],
    },
    "binding": {
        "criteria": ["color_attribute", "shape_attribute", "color_shape_binding"],
    },
    "size": {
        "criteria": [
            "larger_than",
            "smaller_than",
            "same_size",
            "all_larger_than",
            "all_smaller_than",
            "all_same_size",
            "not_all_same_size",
            "largest",
            "smallest",
        ],
    },
    "location": {
        "criteria": ["absolute_region"],
    },
    "direction_order": {
        "criteria": [
            "left_of",
            "right_of",
            "above",
            "below",
            "leftmost",
            "rightmost",
            "topmost",
            "bottommost",
            "all_left_of",
            "all_right_of",
            "all_above",
            "all_below",
        ],
    },
    "proximity": {
        "criteria": ["closer_than", "farther_than"],
    },
    "topology": {
        "criteria": [
            "inside",
            "contains",
            "each_inside",
            "each_contains",
            "touching",
            "not_touching",
        ],
    },
    "between": {
        "criteria": ["between"],
    },
    "structured_layout": {
        "criteria": [
            "same_row",
            "same_column",
            "all_same_row",
            "all_same_column",
            "not_all_same_row",
            "not_all_same_column",
            "grid_occupancy",
        ],
    },
}
SUPPORTED_TEMPORAL = [
    "persistent",
    "moves_direction",
    "stationary",
    "velocity_range",
    "acceleration_trend",
    "size_change",
    "count_consistency",
    "trajectory_shape",
    "temporal_order",
    "relative_distance_change",
    "relative_position_change",
]


@dataclass
class Component:
    color: str
    area: int
    bbox: Tuple[int, int, int, int]
    centroid: Tuple[float, float]
    aspect_ratio: float
    fill_ratio: float
    shape_scores: Dict[str, float]
    boundary: Optional[np.ndarray] = None
    solidity: float = 1.0
    hull_vertex_count: int = 0
    hole_count: int = 0
    centroid_offset: float = 0.0
    multiplicity: int = 1
    robust_extent: float = 0.0
    color_confidence: float = 1.0
    confidence_peak_multiplicity: int = 1


def _is_vvr_bench_spec(spec: Dict[str, Any]) -> bool:
    return spec.get("verifier_version") in VVR_BENCH_VERIFIER_VERSIONS


def _is_vvr_bench_v2(spec: Dict[str, Any]) -> bool:
    return spec.get("verifier_version") in (
        VVR_BENCH_VERIFIER_V2,
        VVR_BENCH_VERIFIER_V3,
        VVR_BENCH_VERIFIER_V4,
    )


def _is_vvr_bench_v3(spec: Dict[str, Any]) -> bool:
    return spec.get("verifier_version") in (VVR_BENCH_VERIFIER_V3, VVR_BENCH_VERIFIER_V4)


def _is_vvr_bench_v4(spec: Dict[str, Any]) -> bool:
    return spec.get("verifier_version") == VVR_BENCH_VERIFIER_V4


def _color_palette(spec: Dict[str, Any]) -> Dict[str, Color]:
    return VVR_BENCH_COLOR_RGB if _is_vvr_bench_spec(spec) else COLOR_RGB


def _supported_shapes(spec: Dict[str, Any]) -> Sequence[str]:
    return VVR_BENCH_SUPPORTED_SHAPES if _is_vvr_bench_spec(spec) else SUPPORTED_SHAPES


def _containment_edge(relation: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    """Return the directed inner-to-outer edge encoded by a relation."""

    relation_type = relation.get("type")
    subject = relation.get("subject")
    object_id = relation.get("object")
    if subject is None or object_id is None:
        return None
    if relation_type in ("inside", "each_inside"):
        return str(subject), str(object_id)
    if relation_type in ("contains", "each_contains"):
        return str(object_id), str(subject)
    return None


def _containment_cycle(relations: Sequence[Dict[str, Any]]) -> Optional[List[str]]:
    """Return one impossible containment cycle, or None for a DAG."""

    adjacency: Dict[str, set[str]] = {}
    for relation in relations:
        edge = _containment_edge(relation)
        if edge is None:
            continue
        inner, outer = edge
        adjacency.setdefault(inner, set()).add(outer)
        adjacency.setdefault(outer, set())

    state: Dict[str, int] = {node: 0 for node in adjacency}
    path: List[str] = []
    cycle: Optional[List[str]] = None

    def visit(node: str) -> bool:
        nonlocal cycle
        state[node] = 1
        path.append(node)
        for outer in sorted(adjacency[node]):
            if state[outer] == 0 and visit(outer):
                return True
            if state[outer] == 1:
                start = path.index(outer)
                cycle = [*path[start:], outer]
                return True
        path.pop()
        state[node] = 2
        return False

    for node in sorted(adjacency):
        if state[node] == 0 and visit(node):
            return cycle
    return None


def validate_spec(spec: Dict[str, Any]) -> List[str]:
    """Return validation errors for a visual verifier spec."""

    errors: List[str] = []
    verifier_version = spec.get("verifier_version")
    if verifier_version not in (None, *VVR_BENCH_VERIFIER_VERSIONS):
        errors.append(f"unsupported verifier_version: {verifier_version}")
    is_benchmark = _is_vvr_bench_spec(spec)
    palette = _color_palette(spec)
    supported_shapes = _supported_shapes(spec)
    modality = spec.get("modality", "image")
    if modality not in ("image", "video"):
        errors.append(f"unsupported modality: {modality}")
    background = spec.get("background", {})
    if background and background.get("color") not in BACKGROUND_RGB:
        errors.append(f"unsupported background color: {background.get('color')}")
    objects = spec.get("objects", [])
    if not objects:
        errors.append("spec must contain at least one object set")
    ids = set()
    for idx, obj in enumerate(objects):
        prefix = f"objects[{idx}]"
        obj_id = obj.get("id")
        if not obj_id:
            errors.append(f"{prefix} missing id")
        elif obj_id in ids:
            errors.append(f"duplicate object id: {obj_id}")
        else:
            ids.add(obj_id)
        if obj.get("color") not in palette:
            errors.append(f"{prefix} unsupported color: {obj.get('color')}")
        if obj.get("shape", "circle") not in supported_shapes:
            errors.append(f"{prefix} unsupported shape: {obj.get('shape')}")
        if obj.get("size") is not None and obj.get("size") not in SUPPORTED_SIZES:
            errors.append(f"{prefix} unsupported size: {obj.get('size')}")
        if is_benchmark and obj.get("size") is not None:
            errors.append(f"{prefix} absolute size is intentionally excluded from VVR-Bench")
        count_mode = obj.get("count_mode", "exact")
        if count_mode not in ("exact", "relative"):
            errors.append(f"{prefix} unsupported count_mode: {count_mode}")
        if count_mode == "exact" and int(obj.get("count", 0)) <= 0:
            errors.append(f"{prefix} count must be positive")
        if count_mode == "relative" and int(obj.get("max_count", 0)) <= 0:
            errors.append(f"{prefix} relative count requires a positive max_count")
        layout = obj.get("layout")
        if layout:
            _validate_layout(layout, prefix, errors)
    object_by_id = {obj.get("id"): obj for obj in objects if obj.get("id")}
    for idx, relation in enumerate(spec.get("relations", [])):
        prefix = f"relations[{idx}]"
        rel_type = relation.get("type")
        supported_relations = SUPPORTED_RELATIONS + (VVR_BENCH_RELATIONS if is_benchmark else [])
        if rel_type not in supported_relations:
            errors.append(f"{prefix} unsupported relation: {rel_type}")
        if rel_type in (
            "all_same_row",
            "all_same_column",
            "not_all_same_row",
            "not_all_same_column",
            "not_all_same_size",
        ):
            subject_id = relation.get("subject")
            if subject_id not in ids:
                errors.append(f"{prefix} subject not found: {subject_id}")
            elif int(object_by_id.get(subject_id, {}).get("count", 0)) < 2:
                errors.append(f"{prefix} requires a subject group with at least two objects")
        elif rel_type == "all_same_size" and relation.get("object") is None:
            subject_id = relation.get("subject")
            if subject_id not in ids:
                errors.append(f"{prefix} subject not found: {subject_id}")
            elif int(object_by_id.get(subject_id, {}).get("count", 0)) < 2:
                errors.append(f"{prefix} requires a subject group with at least two objects")
        elif rel_type in ("each_inside", "each_contains"):
            subject_id = relation.get("subject")
            object_id = relation.get("object")
            if subject_id not in ids:
                errors.append(f"{prefix} subject not found: {subject_id}")
            if object_id not in ids:
                errors.append(f"{prefix} object not found: {object_id}")
            subject_spec = object_by_id.get(subject_id, {})
            object_spec = object_by_id.get(object_id, {})
            for object_name, item in (("subject", subject_spec), ("object", object_spec)):
                if item.get("count_mode", "exact") != "exact":
                    errors.append(f"{prefix} {object_name} must have an exact count")
            subject_count = int(subject_spec.get("count", 0))
            object_count = int(object_spec.get("count", 0))
            if subject_count > object_count:
                errors.append(
                    f"{prefix} distinct containment requires subject count "
                    f"({subject_count}) <= object count ({object_count})"
                )
        elif rel_type in ("between", "closer_than", "farther_than"):
            if relation.get("subject") not in ids:
                errors.append(f"{prefix} subject not found: {relation.get('subject')}")
            if relation.get("anchor_a") not in ids:
                errors.append(f"{prefix} anchor_a not found: {relation.get('anchor_a')}")
            if relation.get("anchor_b") not in ids:
                errors.append(f"{prefix} anchor_b not found: {relation.get('anchor_b')}")
        elif rel_type in ("largest", "smallest", "leftmost", "rightmost", "topmost", "bottommost"):
            if relation.get("subject") not in ids:
                errors.append(f"{prefix} subject not found: {relation.get('subject')}")
            references = relation.get("references")
            if references is not None:
                if not isinstance(references, list) or not references:
                    errors.append(f"{prefix} references must be a non-empty list")
                else:
                    for reference in references:
                        if reference not in ids:
                            errors.append(f"{prefix} reference not found: {reference}")
        else:
            if relation.get("subject") not in ids:
                errors.append(f"{prefix} subject not found: {relation.get('subject')}")
            if relation.get("object") not in ids:
                errors.append(f"{prefix} object not found: {relation.get('object')}")
        raw_references = relation.get("references")
        reference_list = raw_references if isinstance(raw_references, list) else []
        referenced = [
            relation.get("subject"),
            relation.get("object"),
            relation.get("anchor_a"),
            relation.get("anchor_b"),
            *reference_list,
        ]
        referenced = [item for item in referenced if item is not None]
        if len(referenced) != len(set(referenced)):
            errors.append(f"{prefix} cannot compare an object with itself")
        if is_benchmark:
            singleton_relations = {
                "larger_than",
                "smaller_than",
                "same_size",
                "largest",
                "smallest",
                "closer_than",
                "farther_than",
                "touching",
                "separate",
                "not_touching",
                "same_row",
                "same_column",
                "leftmost",
                "rightmost",
                "topmost",
                "bottommost",
            }
            if rel_type in singleton_relations:
                singleton_ids = list(referenced)
                if rel_type in ("largest", "smallest", "leftmost", "rightmost", "topmost", "bottommost") and raw_references is None:
                    singleton_ids.extend(object_id for object_id in ids if object_id != relation.get("subject"))
                for object_id in singleton_ids:
                    if int(object_by_id.get(object_id, {}).get("count", 0)) != 1:
                        errors.append(f"{prefix} requires singleton object groups: {object_id}")
            if rel_type in ("closer_than", "farther_than"):
                if float(relation.get("max_distance_ratio", 0.70)) <= 0.0:
                    errors.append(f"{prefix} max_distance_ratio must be positive")
                if float(relation.get("min_margin_px", 8.0)) < 0.0:
                    errors.append(f"{prefix} min_margin_px cannot be negative")
    containment_cycle = _containment_cycle(spec.get("relations", []))
    if containment_cycle is not None:
        errors.append(
            "containment relations form an impossible cycle: "
            + " inside ".join(containment_cycle)
        )
    variable_size_groups = {
        str(relation.get("subject"))
        for relation in spec.get("relations", [])
        if relation.get("type") == "not_all_same_size"
    }
    for idx, relation in enumerate(spec.get("relations", [])):
        if relation.get("type") != "all_same_size":
            continue
        same_size_groups = {str(relation.get("subject"))}
        if relation.get("object") is not None:
            same_size_groups.add(str(relation.get("object")))
        if variable_size_groups.intersection(same_size_groups):
            errors.append(
                f"relations[{idx}] all_same_size contradicts not_all_same_size for a referenced group"
            )
    for idx, temporal in enumerate(spec.get("temporal", [])):
        prefix = f"temporal[{idx}]"
        temp_type = temporal.get("type")
        if temp_type not in SUPPORTED_TEMPORAL:
            errors.append(f"{prefix} unsupported temporal verifier: {temp_type}")
        if temporal.get("object") not in ids:
            errors.append(f"{prefix} object not found: {temporal.get('object')}")
        if temp_type in ("relative_distance_change", "relative_position_change"):
            if temporal.get("reference_object") not in ids:
                errors.append(f"{prefix} reference_object not found: {temporal.get('reference_object')}")
    for idx, forbidden in enumerate(spec.get("forbidden", [])):
        prefix = f"forbidden[{idx}]"
        if forbidden.get("type") == "text":
            # Text absence is a declared verifier, but this module currently
            # treats it as unscored unless an OCR-backed implementation is added.
            continue
        color = forbidden.get("color")
        if color is not None and color not in palette:
            errors.append(f"{prefix} unsupported forbidden color: {color}")
    return errors


def score_sample_spec(sample: np.ndarray, spec: Dict[str, Any], reward_mode: str = "default") -> Tuple[float, Dict[str, Any]]:
    """Score an image or video against a formal visual spec."""

    modality = spec.get("modality", "image")
    if modality == "video":
        return score_video_spec(sample, spec)
    return score_image_spec(sample, spec, reward_mode=reward_mode)


def _validate_layout(layout: Dict[str, Any], prefix: str, errors: List[str]) -> None:
    layout_type = layout.get("type")
    if layout_type == "regions":
        regions = layout.get("regions", [])
        if not regions:
            errors.append(f"{prefix}.layout regions cannot be empty")
        for region in regions:
            if region not in SUPPORTED_REGIONS:
                errors.append(f"{prefix}.layout unsupported region: {region}")
    elif layout_type == "grid_cells":
        rows = int(layout.get("rows", 0))
        cols = int(layout.get("cols", 0))
        cells = layout.get("cells", [])
        if rows <= 0 or cols <= 0:
            errors.append(f"{prefix}.layout rows/cols must be positive")
        for cell in cells:
            if len(cell) != 2:
                errors.append(f"{prefix}.layout invalid cell: {cell}")
                continue
            row, col = int(cell[0]), int(cell[1])
            if row < 0 or row >= rows or col < 0 or col >= cols:
                errors.append(f"{prefix}.layout cell out of bounds: {cell}")
    elif layout_type in ("horizontal_row", "vertical_column", "diagonal_down"):
        return
    else:
        errors.append(f"{prefix}.layout unsupported type: {layout_type}")


def score_image_spec(image: np.ndarray, spec: Dict[str, Any], reward_mode: str = "default") -> Tuple[float, Dict[str, Any]]:
    """Score one image against one formal visual spec."""

    image = np.asarray(image)
    if image.dtype != np.uint8:
        image = np.clip(np.rint(image), 0, 255).astype(np.uint8)
    errors = validate_spec(spec)
    if errors:
        return 0.0, {
            "visual_logic": [0.0],
            "visual_logic_strict": [0.0],
            "visual_logic_loose": [0.0],
            "visual_logic_valid_spec": [0.0],
            "visual_logic_background": [0.0],
            "visual_logic_forbidden": [0.0],
            "visual_logic_forbidden_violations": [0.0],
            "visual_logic_extra_components": [0.0],
            "visual_logic_object_count_score": [0.0],
            "visual_logic_shape_score": [0.0],
            "visual_logic_size_score": [0.0],
            "visual_logic_layout_score": [0.0],
            "visual_logic_relation_score": [0.0],
            "visual_logic_relation_strict": [0.0],
        }

    background_color = (spec.get("background") or {}).get("color")
    is_benchmark = _is_vvr_bench_spec(spec)
    is_benchmark_v2 = _is_vvr_bench_v2(spec)
    is_benchmark_v3 = _is_vvr_bench_v3(spec)
    is_benchmark_v4 = _is_vvr_bench_v4(spec)
    relations = spec.get("relations", [])
    component_shape_version = (
        3
        if is_benchmark_v4 and not relations
        else (2 if is_benchmark_v2 else 1)
    )
    palette = _color_palette(spec)
    hue_intervals = VVR_BENCH_HUE_INTERVALS_V4 if is_benchmark_v4 else VVR_BENCH_HUE_INTERVALS
    expected_shapes_by_color: Dict[str, set[str]] = {}
    for item in spec.get("objects", []):
        expected_shapes_by_color.setdefault(str(item.get("color")), set()).add(
            str(item.get("shape", "circle"))
        )
    expected_shape_by_color = {
        color: next(iter(shapes))
        for color, shapes in expected_shapes_by_color.items()
        if len(shapes) == 1
    }
    components_by_color = {
        color: find_color_components(
            image,
            color,
            background_color,
            palette=palette,
            include_boundary=is_benchmark,
            adaptive_color=is_benchmark_v2,
            candidate_colors=tuple(palette),
            shape_version=component_shape_version,
            hue_intervals=hue_intervals,
        )
        for color in palette
    }
    foreground_components = (
        find_foreground_components(image, shape_version=2)
        if is_benchmark_v4
        else []
    )
    color_pixel_shares: Dict[str, float] = {}
    if is_benchmark_v4:
        color_pixel_counts = _adaptive_color_pixel_counts(image, tuple(palette), hue_intervals)
        total_color_pixels = max(sum(color_pixel_counts.values()), 1)
        color_pixel_shares = {
            color: count / total_color_pixels
            for color, count in color_pixel_counts.items()
        }
    count_components_by_color = (
        _suppress_cross_color_fragments(
            {
                color: find_color_components(
                    image,
                    color,
                    background_color,
                    palette=palette,
                    include_boundary=is_benchmark,
                    adaptive_color=True,
                    candidate_colors=tuple(palette),
                    shape_version=component_shape_version,
                    multiplicity_version=3 if is_benchmark_v4 else component_shape_version,
                    shape_hint=expected_shape_by_color.get(color) if is_benchmark_v4 else None,
                    closing_radius=0,
                    hue_intervals=hue_intervals,
                )
                for color in palette
            },
            require_similar_extent=is_benchmark_v4,
        )
        if is_benchmark_v2
        else components_by_color
    )
    geometry_components_by_color = (
        _suppress_aliased_geometry_candidates(components_by_color, spec.get("objects", []))
        if is_benchmark_v2
        else components_by_color
    )
    matched_component_ids = set()
    object_results: List[Dict[str, Any]] = []
    object_scores: List[float] = []
    id_to_components: Dict[str, List[Component]] = {}
    id_to_color_components: Dict[str, List[Component]] = {}
    id_to_observed_components: Dict[str, List[Component]] = {}
    id_to_relation_components: Dict[str, List[Component]] = {}
    id_to_candidate_components: Dict[str, List[Component]] = {}
    id_to_shape: Dict[str, str] = {}
    id_to_color: Dict[str, str] = {}
    containment_container_ids = {
        str(
            relation["object"]
            if relation.get("type") in {"inside", "each_inside"}
            else relation["subject"]
        )
        for relation in relations
        if relation.get("type") in {"inside", "contains", "each_inside", "each_contains"}
    }
    for obj in spec.get("objects", []):
        shape = obj.get("shape", "circle")
        observed_components = _countable_components(count_components_by_color.get(obj["color"], []))
        color_components = _countable_components(
            components_by_color.get(obj["color"], []),
            min_dimension=6,
            min_color_confidence=0.25,
            min_relative_area=0.005,
        )
        components = _countable_components(
            geometry_components_by_color.get(obj["color"], []),
            min_dimension=6,
            min_color_confidence=0.25,
            min_relative_area=0.005,
        )
        if is_benchmark_v4 and len(expected_shapes_by_color.get(str(obj["color"]), set())) > 1:
            # One connected component cannot satisfy two same-color shape
            # groups. Assign it to the requested shape with strongest
            # geometric support before computing count or relation evidence.
            expected_shapes = expected_shapes_by_color[str(obj["color"])]
            observed_components = [
                component
                for component in observed_components
                if _component_matches_exclusive_expected_shape(
                    component, shape, expected_shapes
                )
            ]
            color_components = [
                component
                for component in color_components
                if _component_matches_exclusive_expected_shape(
                    component, shape, expected_shapes
                )
            ]
            components = [
                component
                for component in components
                if _component_matches_exclusive_expected_shape(
                    component, shape, expected_shapes
                )
            ]
        if (
            is_benchmark_v4
            and shape in {"circle", "square"}
            and str(obj["id"]) in containment_container_ids
        ):
            # A valid container is often rendered as a hollow colored frame.
            # Generic count filtering rejects its intentionally low fill ratio,
            # so recover only shape-supported, single-hole outlines.
            outline_components = (
                _outline_circle_components(
                    geometry_components_by_color.get(obj["color"], [])
                )
                if shape == "circle"
                else _outline_square_components(
                    geometry_components_by_color.get(obj["color"], [])
                )
            )
            observed_components = _dedupe_component_candidates(
                [*observed_components, *outline_components],
                shape,
            )
            color_components = _dedupe_component_candidates(
                [*color_components, *outline_components],
                shape,
            )
            components = _dedupe_component_candidates(
                [*components, *outline_components],
                shape,
            )
        fallback_components: List[Component] = []
        fallback_observed_components: List[Component] = []
        current_match_quality = max((_component_match_quality(component, shape) for component in components), default=0.0)
        # A single-color fallback can recover heavily shifted colors, but it
        # also admits neighboring hues. Only use it when the globally
        # partitioned color mask has no credible candidate.
        if is_benchmark_v2 and current_match_quality < 0.95:
            fallback_candidate_colors = (
                ("green", "cyan", "blue")
                if is_benchmark_v4 and obj["color"] == "cyan"
                else (obj["color"],)
            )
            fallback_components = find_color_components(
                image,
                obj["color"],
                background_color,
                palette=palette,
                include_boundary=True,
                adaptive_color=True,
                candidate_colors=fallback_candidate_colors,
                shape_version=component_shape_version,
                shape_hint=shape,
                hue_intervals=hue_intervals,
            )
            fallback_components = _countable_components(
                fallback_components,
                min_dimension=6,
                min_color_confidence=0.25,
                min_relative_area=0.005,
            )
            fallback_components = _filter_fallback_aliases(
                fallback_components,
                target_color=obj["color"],
                components_by_color=components_by_color,
                expected_shape_by_color=expected_shape_by_color,
                strict_named_color_identity=is_benchmark_v4,
            )
            fallback_observed_components = _countable_components(
                find_color_components(
                    image,
                    obj["color"],
                    background_color,
                    palette=palette,
                    include_boundary=True,
                    adaptive_color=True,
                    candidate_colors=fallback_candidate_colors,
                    shape_version=component_shape_version,
                    multiplicity_version=3 if is_benchmark_v4 else component_shape_version,
                    closing_radius=0,
                    hue_intervals=hue_intervals,
                )
            )
            if is_benchmark_v4:
                fallback_observed_components = _filter_fallback_aliases(
                    fallback_observed_components,
                    target_color=obj["color"],
                    components_by_color=components_by_color,
                    expected_shape_by_color=expected_shape_by_color,
                    strict_named_color_identity=True,
                )
            if (
                not is_benchmark_v4
                and
                fallback_observed_components
                and (
                    not observed_components
                    or (
                        sum(component.multiplicity for component in fallback_observed_components)
                        < sum(component.multiplicity for component in observed_components)
                        and _mean_shape_support(fallback_observed_components, shape)
                        >= _mean_shape_support(observed_components, shape) + 0.20
                    )
                )
            ):
                observed_components = fallback_observed_components
            fallback_quality = max((_component_match_quality(component, shape) for component in fallback_components), default=0.0)
            credible_global_shape = any(
                component.shape_scores.get(shape, 0.0) >= _shape_presence_threshold(shape)
                for component in components
            )
            if fallback_quality > 0.0 and not (is_benchmark_v4 and credible_global_shape):
                components = _dedupe_component_candidates([*components, *fallback_components], shape)
        if shape:
            components = sorted(
                components,
                key=(
                    (lambda comp: _component_match_quality(comp, shape))
                    if is_benchmark_v2
                    else (lambda comp: comp.shape_scores.get(shape, 0.0))
                ),
                reverse=True,
            )
        count_mode = str(obj.get("count_mode", "exact"))
        count_is_exact = count_mode == "exact"
        target_count = int(obj.get("count", 1))
        selection_limit = (
            target_count
            if count_is_exact
            else int(obj.get("max_count", 10))
        )
        selected = components[:selection_limit]
        id_to_components[obj["id"]] = selected
        id_to_color_components[obj["id"]] = color_components
        identity_components = (
            [
                component
                for component in components
                if _component_identity_compatible(component, shape, image.shape[:2])
            ]
            if is_benchmark_v4
            else components
        )
        id_to_relation_components[obj["id"]] = identity_components[:selection_limit]
        id_to_candidate_components[obj["id"]] = components[:4]
        id_to_shape[obj["id"]] = shape
        id_to_color[obj["id"]] = obj["color"]
        if is_benchmark_v2:
            if is_benchmark_v4:
                global_count_components = [
                    component
                    for component in observed_components
                    if _component_count_compatible(component, shape, image.shape[:2])
                ]
                fallback_count_components = [
                    component
                    for component in fallback_observed_components
                    if _component_count_compatible(component, shape, image.shape[:2])
                ]
                # Preserve the exclusive color-partition count. A fallback may
                # recover a missing group or merge fragments of one object, but
                # it cannot inflate an already-observed count with neighboring
                # colors or wrong-shape components.
                count_components = observed_components
                observed_count = _effective_component_count(observed_components, shape)
                fallback_count = _effective_component_count(fallback_count_components, shape)
                raw_fallback_count = _effective_component_count(
                    fallback_observed_components,
                    shape,
                )
                if not observed_components and fallback_count_components:
                    count_components = fallback_count_components
                elif _fallback_repairs_fragmented_count(
                    fallback_count_components,
                    observed_components,
                    shape,
                ):
                    count_components = fallback_count_components
                elif (
                    count_is_exact
                    and
                    observed_count < target_count
                    and observed_count < raw_fallback_count <= target_count
                ):
                    count_components = fallback_observed_components
                # Preserve the previously validated count-relation grounding.
                # The fallback arbitration above is specific to exact count.
                id_to_observed_components[obj["id"]] = (
                    global_count_components
                    if global_count_components
                    else _substantive_count_components(observed_components)
                )
            else:
                count_components = observed_components
                id_to_observed_components[obj["id"]] = observed_components
        else:
            count_components = components
            id_to_observed_components[obj["id"]] = [
                component
                for component in components
                if component.shape_scores.get(shape, 0.0) >= 0.25
            ]
        for comp in selected:
            matched_component_ids.add(id(comp))
        count_pred = sum(component.multiplicity for component in count_components) if is_benchmark_v2 else len(count_components)
        if is_benchmark_v4:
            count_pred = _estimate_repeated_group_count(count_components, shape)
        if count_is_exact:
            count_error, count_score = _score_exact_count(count_pred, target_count)
        else:
            count_error = 0.0 if count_pred >= 1 else 1.0
            count_score = 1.0 if count_pred >= 1 else 0.0
        color_presence_score = min(1.0, float(count_pred))
        color_presence_strict = count_pred >= 1
        if (
            is_benchmark_v4
            and len(spec.get("objects", [])) > 1
            and not color_presence_strict
            and any(
                _component_identity_compatible(component, shape, image.shape[:2])
                for component in fallback_components
            )
        ):
            color_presence_score = 1.0
            color_presence_strict = True
        if is_benchmark_v4 and len(spec.get("objects", [])) == 1 and not spec.get("relations"):
            # Atomic color claims require the named hue to dominate the
            # chromatic foreground. This accepts large/cropped valid objects
            # while rejecting tiny neighboring-hue highlights and artifacts.
            color_share = color_pixel_shares.get(obj["color"], 0.0)
            color_presence_score = min(1.0, color_share / 0.40)
            color_presence_strict = color_share >= 0.40
        shape_score = _score_shape_attribute(
            shape,
            selected,
            allow_occluded_triangle=is_benchmark_v4,
        )
        binding_score = _score_color_shape_binding(count_score, shape_score)
        binding_presence_score = min(
            color_presence_score,
            min(1.0, shape_score / max(_shape_presence_threshold(shape), 1e-6)),
        )
        binding_presence_strict = color_presence_strict and _shape_binding_supported(
            selected,
            shape,
            allow_occluded_triangle=is_benchmark_v4,
        )
        binding_components = list(count_components)
        binding_all_strict = bool(
            binding_components
            and all(
                _component_all_binding_compatible(component, shape)
                or (
                    is_benchmark_v4
                    and shape == "square"
                    and (
                        bool(_outline_square_components([component]))
                        or (
                            str(obj["id"]) in containment_container_ids
                            and _square_container_exterior_supported(component)
                        )
                    )
                )
                or (
                    is_benchmark_v4
                    and shape == "triangle"
                    and _is_occluded_triangle_component(component)
                )
                for component in binding_components
            )
        )
        layout_components = (
            id_to_relation_components[obj["id"]]
            if is_benchmark_v4 and (obj.get("layout") or {}).get("type") == "regions"
            else selected
        )
        layout_score = _score_layout(
            obj.get("layout"),
            layout_components,
            image.shape[:2],
            benchmark_version=4 if is_benchmark_v4 else 1,
        )
        size_score = _score_size(obj.get("size"), selected, image.shape[:2], benchmark=is_benchmark)
        score = 0.45 * count_score + 0.20 * shape_score + 0.20 * layout_score + 0.15 * size_score
        object_scores.append(score)
        object_results.append(
            {
                "id": obj["id"],
                "color": obj["color"],
                "shape": shape,
                "target_count": float(target_count),
                "count_mode": count_mode,
                "pred_count": float(count_pred),
                "count_error": float(count_error),
                "count_score": float(count_score),
                "color_presence_score": float(color_presence_score),
                "color_presence_strict": bool(color_presence_strict),
                "shape_score": float(shape_score),
                "size": obj.get("size"),
                "size_score": float(size_score),
                "binding_score": float(binding_score),
                "binding_presence_score": float(binding_presence_score),
                "binding_presence_strict": bool(binding_presence_strict),
                "binding_all_strict": binding_all_strict,
                "selected_component_count": len(selected),
                "selected_max_hole_count": max(
                    (component.hole_count for component in selected), default=0
                ),
                "selected_mean_fill_ratio": float(
                    np.mean([component.fill_ratio for component in selected])
                )
                if selected
                else 0.0,
                "selected_mean_aspect_ratio": float(
                    np.mean(
                        [
                            max(
                                component.aspect_ratio,
                                1.0 / max(component.aspect_ratio, 1e-6),
                            )
                            for component in selected
                        ]
                    )
                )
                if selected
                else 0.0,
                "selected_mean_square_score": float(
                    np.mean(
                        [component.shape_scores.get("square", 0.0) for component in selected]
                    )
                )
                if selected
                else 0.0,
                "layout_score": float(layout_score),
            }
        )

    if is_benchmark_v4:
        indistinguishable_ids = _indistinguishable_cyan_blue_bindings(
            image,
            spec.get("objects", []),
            id_to_components,
        )
        for result in object_results:
            if str(result["id"]) in indistinguishable_ids:
                result["binding_presence_strict"] = False

    between_components: Dict[str, List[Component]] = {}
    if is_benchmark_v4 and any(relation.get("type") == "between" for relation in relations):
        for obj in spec.get("objects", []):
            candidates = _countable_components(
                find_color_components(
                    image,
                    obj["color"],
                    background_color,
                    palette=palette,
                    include_boundary=True,
                    adaptive_color=True,
                    candidate_colors=tuple(palette),
                    shape_version=3,
                    hue_intervals=hue_intervals,
                ),
                min_dimension=6,
                min_color_confidence=0.25,
                min_relative_area=0.005,
            )
            shape = str(obj.get("shape", "circle"))
            candidates = sorted(
                candidates,
                key=lambda component: _component_match_quality(component, shape),
                reverse=True,
            )
            component_limit = (
                int(obj.get("count", 1))
                if obj.get("count_mode", "exact") == "exact"
                else int(obj.get("max_count", 10))
            )
            between_components[str(obj["id"])] = [
                component
                for component in candidates
                if _component_identity_compatible(component, shape, image.shape[:2])
                and _components_have_shape_identity([component], shape)
            ][:component_limit]

    relation_results = []
    relation_scores = []
    spec_objects_by_id = {str(obj["id"]): obj for obj in spec.get("objects", [])}
    for relation in spec.get("relations", []):
        relation_for_score = relation
        if relation.get("type") in {"each_inside", "each_contains"}:
            relation_for_score = dict(relation)
            relation_for_score["expected_subject_count"] = int(
                spec_objects_by_id[str(relation["subject"])]["count"]
            )
            relation_for_score["expected_object_count"] = int(
                spec_objects_by_id[str(relation["object"])]["count"]
            )
        relation_components = id_to_relation_components if is_benchmark_v4 else id_to_components
        between_v3_grounded = bool(
            is_benchmark_v4
            and relation.get("type") == "between"
            and all(between_components.get(object_id) for object_id in _relation_object_ids(relation))
        )
        if between_v3_grounded:
            relation_components = between_components
        aliased_extremum_subject = (
            is_benchmark_v4
            and relation.get("type") in {"largest", "smallest"}
            and _extremum_subject_aliases_reference(relation, relation_components)
        )
        if is_benchmark_v4 and relation.get("type") in {
            "inside",
            "contains",
            "each_inside",
            "each_contains",
        }:
            relation_components = {
                object_id: (
                    id_to_relation_components.get(object_id, [])
                    or [
                        component
                        for component in id_to_components.get(object_id, [])
                        if _topology_shape_supported(component, id_to_shape[object_id])
                    ]
                )
                for object_id in id_to_components
            }
        if is_benchmark_v4 and relation.get("type") in {"touching", "separate", "not_touching"}:
            relation_components = {
                object_id: [
                    component
                    for component in id_to_components.get(object_id, [])
                    if _topology_shape_supported(component, id_to_shape[object_id])
                ]
                for object_id in id_to_components
            }
        if is_benchmark_v4 and relation.get("type") in {
            "leftmost",
            "rightmost",
            "topmost",
            "bottommost",
            "largest",
            "smallest",
        }:
            relation_for_score, relation_components = _ground_extreme_against_visible_objects(
                relation,
                relation_components,
                count_components_by_color,
                background_color,
            )
        if is_benchmark_v2 and relation.get("type") in ("closer_than", "farther_than"):
            relation_components = _ground_comparative_distance_relation(
                relation,
                relation_components,
                (
                    {
                        object_id: [
                            component
                            for component in candidates
                            if _component_identity_compatible(
                                component,
                                id_to_shape[object_id],
                                image.shape[:2],
                            )
                        ]
                        for object_id, candidates in id_to_candidate_components.items()
                    }
                    if is_benchmark_v4
                    else id_to_candidate_components
                ),
                id_to_shape,
            )
        benchmark_version = 4 if is_benchmark_v4 else (3 if is_benchmark_v3 else (2 if is_benchmark_v2 else 1))
        aliased_to_stronger_color = (
            is_benchmark_v4
            and relation.get("type") in {"closer_than", "farther_than"}
            and _relation_referent_aliases_stronger_object(
                relation,
                relation_components,
                id_to_components,
                id_to_shape,
            )
        )
        missing_shape_identity = (
            is_benchmark_v4
            and (
                (
                    relation.get("type")
                    in {
                        "larger_than",
                        "smaller_than",
                        "same_size",
                        "all_larger_than",
                        "all_smaller_than",
                        "all_same_size",
                        "not_all_same_size",
                        "closer_than",
                        "farther_than",
                        "between",
                        "touching",
                        "separate",
                        "not_touching",
                    }
                    and (
                        _relation_has_missing_shape_identity(
                            relation,
                            relation_components,
                            id_to_shape,
                        )
                        or (
                            not between_v3_grounded
                            and _relation_has_strong_shape_conflict(
                                relation,
                                id_to_color_components,
                                id_to_shape,
                                [],
                                relation_components,
                            )
                        )
                    )
                )
                or (
                    relation.get("type") in {"left_of", "right_of", "above", "below"}
                    and (
                        _relation_all_referents_lack_shape_identity(
                            relation,
                            relation_components,
                            id_to_shape,
                        )
                        or _relation_has_strong_shape_conflict(
                            relation,
                            id_to_color_components,
                            id_to_shape,
                            foreground_components,
                            relation_components,
                        )
                    )
                )
            )
        )
        if (
            is_benchmark_v4
            and relation.get("type")
            in {
                "left_of",
                "right_of",
                "above",
                "below",
                "larger_than",
                "smaller_than",
                "same_size",
                "all_larger_than",
                "all_smaller_than",
                "all_same_size",
                "not_all_same_size",
                "closer_than",
                "farther_than",
                "between",
            }
            and any(
                not relation_components.get(object_id)
                or (
                    not id_to_color_components.get(object_id)
                    and id_to_color.get(object_id) != "cyan"
                )
                for object_id in _relation_object_ids(relation)
            )
        ):
            score = 0.0
            result = {
                "type": relation["type"],
                "subject": relation["subject"],
                "object": relation.get("object"),
                "score": 0.0,
                "strict_pass": False,
                "missing_color_referent": True,
            }
        elif aliased_to_stronger_color or aliased_extremum_subject or missing_shape_identity:
            score = 0.0
            result = {
                "type": relation["type"],
                "subject": relation["subject"],
                "object": relation.get("object"),
                "score": 0.0,
                "strict_pass": False,
                "aliased_to_stronger_color": bool(aliased_to_stronger_color),
                "aliased_extremum_subject": bool(aliased_extremum_subject),
                "missing_shape_identity": bool(missing_shape_identity),
            }
        else:
            score, result = _score_relation(
                relation_for_score,
                relation_components,
                observed_id_to_components=id_to_observed_components,
                object_shapes=id_to_shape,
                image_hw=image.shape[:2],
                benchmark_version=benchmark_version,
            )
        relation_scores.append(score)
        relation_results.append(result)

    forbidden_score, forbidden_details = _score_forbidden(spec, components_by_color)
    if is_benchmark_v3:
        selected_components = [
            (component, id_to_shape[object_id])
            for object_id, components in id_to_components.items()
            for component in components
        ]
        extra_score, extra_count = _score_semantic_extra_components(
            components_by_color,
            matched_component_ids,
            selected_components,
        )
        background_score = _background_score_v3(image, background_color)
    else:
        extra_score, extra_count = _score_extra_components(components_by_color, matched_component_ids)
        background_score = _background_score(image, background_color)

    terms = []
    if object_scores:
        terms.append(0.55 * float(np.mean(object_scores)))
    if relation_scores:
        terms.append(0.15 * float(np.mean(relation_scores)))
    terms.append(0.15 * forbidden_score)
    terms.append(0.10 * extra_score)
    terms.append(0.05 * background_score)
    base_score = float(sum(terms) / sum([0.55, 0.15 if relation_scores else 0.0, 0.15, 0.10, 0.05]))

    object_count_score = float(np.mean([o["count_score"] for o in object_results])) if object_results else 0.0
    shape_score = float(np.mean([o["shape_score"] for o in object_results])) if object_results else 0.0
    binding_score = float(np.mean([o["binding_score"] for o in object_results])) if object_results else 0.0
    binding_strict = (
        float(np.mean([1.0 if o["binding_presence_strict"] else 0.0 for o in object_results]))
        if object_results
        else 0.0
    )
    binding_all_strict = (
        float(np.mean([1.0 if o["binding_all_strict"] else 0.0 for o in object_results]))
        if object_results
        else 0.0
    )
    atomic_shape_score = shape_score
    if is_benchmark_v2 and any(item.get("shape") == "triangle" for item in spec.get("objects", [])):
        foreground_components = find_foreground_components(
            image,
            background_color,
            shape_hint="triangle",
            shape_version=component_shape_version,
        )
        atomic_shape_scores = []
        for item, result in zip(spec.get("objects", []), object_results):
            shape_name = str(item.get("shape", "circle"))
            if shape_name == "triangle":
                atomic_shape_scores.append(
                    max(
                        (component.shape_scores.get(shape_name, 0.0) for component in foreground_components),
                        default=0.0,
                    )
                )
            else:
                atomic_shape_scores.append(float(result["shape_score"]))
        atomic_shape_score = float(np.mean(atomic_shape_scores)) if atomic_shape_scores else shape_score
    if is_benchmark_v4 and len(spec.get("objects", [])) == 1 and not spec.get("relations"):
        shape_name = str(spec["objects"][0].get("shape", "circle"))
        foreground_components = find_foreground_components(
            image,
            background_color,
            shape_hint=shape_name,
            shape_version=component_shape_version,
        )
        atomic_shape_score = max(
            [
                *(component.shape_scores.get(shape_name, 0.0) for component in foreground_components),
                float(object_results[0]["shape_score"]),
            ],
            default=0.0,
        )
    size_score = float(np.mean([o["size_score"] for o in object_results])) if object_results else 1.0
    layout_score = float(np.mean([o["layout_score"] for o in object_results])) if object_results else 0.0
    relation_score = float(np.mean(relation_scores)) if relation_scores else 1.0
    relation_strict = float(np.mean([1.0 if r["strict_pass"] else 0.0 for r in relation_results])) if relation_results else 1.0
    shape_strict_flags = [obj["shape_score"] >= 0.55 for obj in object_results]
    if is_benchmark_v4:
        shape_strict_flags = [
            float(obj["shape_score"]) >= _shape_presence_threshold(str(obj["shape"]))
            for obj in object_results
        ]
        if len(spec.get("objects", [])) == 1 and not spec.get("relations"):
            shape_strict_flags = [
                atomic_shape_score
                >= _shape_presence_threshold(str(spec["objects"][0].get("shape", "circle")))
            ]
    elif is_benchmark_v3:
        shape_strict_flags = [
            _v3_shape_strict_pass(obj, id_to_components, object_results)
            for obj in object_results
        ]
    shape_strict_pass = all(shape_strict_flags)

    target_total = sum(int(item.get("count", 0)) for item in spec.get("objects", []))
    count_gate = 0.10 + 0.90 * object_count_score
    relation_gate = 1.0 if not relation_scores else 0.25 + 0.75 * relation_score
    extra_gate = max(0.0, 1.0 - float(extra_count) / max(float(target_total), 1.0))
    forbidden_gate = forbidden_score
    sparse_terms = [
        object_count_score >= 0.99,
        shape_strict_pass,
        layout_score >= 0.99,
        relation_strict >= 0.99,
        forbidden_score >= 0.99,
        extra_count == 0,
        background_score >= 0.70,
    ]
    sparse_pass_fraction = float(np.mean(sparse_terms))
    sparse_pass = sparse_pass_fraction >= 1.0
    gated_score = base_score * count_gate * extra_gate * relation_gate * forbidden_gate
    if reward_mode == "sparse":
        score = base_score if sparse_pass else 0.0
    elif reward_mode in ("gated", "soft_gate"):
        score = gated_score
    elif reward_mode == "default":
        score = base_score
    else:
        raise ValueError(f"unsupported visual verifier reward_mode: {reward_mode}")

    strict_pass = (
        all(
            obj["count_error"] == 0.0
            and shape_is_strict
            and obj["layout_score"] >= 0.99
            and (obj["size"] is None or obj["size_score"] >= 0.99)
            for obj, shape_is_strict in zip(object_results, shape_strict_flags)
        )
        and all(result["strict_pass"] for result in relation_results)
        and forbidden_score >= 0.99
        and extra_count == 0
        and background_score >= 0.70
    )
    loose_pass = score >= 0.75
    details = {
        "visual_logic": [score],
        "visual_logic_strict": [1.0 if strict_pass else 0.0],
        "visual_logic_loose": [1.0 if loose_pass else 0.0],
        "visual_logic_valid_spec": [1.0],
        "visual_logic_background": [background_score],
        "visual_logic_forbidden": [forbidden_score],
        "visual_logic_forbidden_violations": [float(sum(1 for item in forbidden_details if item.get("score", 1.0) < 1.0))],
        "visual_logic_extra_components": [float(extra_count)],
        "visual_logic_object_count_score": [object_count_score],
        "visual_logic_shape_score": [shape_score],
        "visual_logic_color_shape_binding_score": [binding_score],
        "visual_logic_color_shape_binding_strict": [binding_strict],
        "visual_logic_color_shape_binding_all_strict": [binding_all_strict],
        "visual_logic_atomic_shape_match": [atomic_shape_score],
        "visual_logic_size_score": [size_score],
        "visual_logic_layout_score": [layout_score],
        "visual_logic_relation_score": [relation_score],
        "visual_logic_relation_strict": [relation_strict],
        "visual_logic_base": [base_score],
        "visual_logic_gated_score": [gated_score],
        "visual_logic_count_gate": [count_gate],
        "visual_logic_extra_gate": [extra_gate],
        "visual_logic_relation_gate": [relation_gate],
        "visual_logic_forbidden_gate": [forbidden_gate],
        "visual_logic_sparse_pass_fraction": [sparse_pass_fraction],
        "visual_logic_sparse_pass": [1.0 if sparse_pass else 0.0],
    }
    if is_benchmark_v4:
        for result in object_results:
            object_id = re.sub(r"[^a-zA-Z0-9_]+", "_", str(result["id"]))
            details[f"visual_logic_object_{object_id}_shape_score"] = [
                float(result["shape_score"])
            ]
            details[f"visual_logic_object_{object_id}_binding_all_strict"] = [
                1.0 if result["binding_all_strict"] else 0.0
            ]
            for diagnostic in (
                "selected_component_count",
                "selected_max_hole_count",
                "selected_mean_fill_ratio",
                "selected_mean_aspect_ratio",
                "selected_mean_square_score",
            ):
                details[f"visual_logic_object_{object_id}_{diagnostic}"] = [
                    float(result[diagnostic])
                ]
    predicate_scores: Dict[str, List[float]] = {}
    predicate_strict: Dict[str, List[float]] = {}
    for result in relation_results:
        predicate_type = str(result.get("type", "unknown"))
        predicate_scores.setdefault(predicate_type, []).append(float(result.get("score", 0.0)))
        predicate_strict.setdefault(predicate_type, []).append(1.0 if result.get("strict_pass") else 0.0)
    for predicate_type, values in predicate_scores.items():
        details[f"visual_logic_predicate_{predicate_type}"] = [float(np.mean(values))]
        details[f"visual_logic_predicate_{predicate_type}_strict"] = [
            float(np.mean(predicate_strict[predicate_type]))
        ]
    if object_results:
        object_shape_scores = [float(obj["shape_score"]) for obj in object_results]
        object_shape_strict = list(shape_strict_flags)
        if is_benchmark_v4:
            if len(spec.get("objects", [])) == 1 and not spec.get("relations"):
                object_shape_scores = [atomic_shape_score]
        object_predicates = {
            "exact_count": (
                [float(obj["count_score"]) for obj in object_results],
                [float(obj["count_error"]) == 0.0 for obj in object_results],
            ),
            "color_attribute": (
                [float(obj["color_presence_score"]) for obj in object_results],
                [bool(obj["color_presence_strict"]) for obj in object_results],
            ),
            "shape_attribute": (
                object_shape_scores,
                object_shape_strict,
            ),
            "color_shape_binding": (
                [float(obj["binding_presence_score"]) for obj in object_results],
                [bool(obj["binding_presence_strict"]) for obj in object_results],
            ),
        }
        if is_benchmark_v4 and len(spec.get("objects", [])) == 1 and not spec.get("relations"):
            color_score = float(object_results[0]["color_presence_score"])
            color_strict = bool(object_results[0]["color_presence_strict"])
            shape_name = str(spec["objects"][0].get("shape", "circle"))
            shape_strict = atomic_shape_score >= _shape_presence_threshold(shape_name)
            object_predicates["color_shape_binding"] = (
                [min(color_score, min(1.0, atomic_shape_score / _shape_presence_threshold(shape_name)))],
                [color_strict and shape_strict],
            )
        for predicate_type, (values, strict_values) in object_predicates.items():
            details[f"visual_logic_predicate_{predicate_type}"] = [float(np.mean(values))]
            details[f"visual_logic_predicate_{predicate_type}_strict"] = [
                float(np.mean(strict_values))
            ]
        for predicate_type, layout_type in (
            ("absolute_region", "regions"),
            ("grid_occupancy", "grid_cells"),
        ):
            layout_values = [
                float(result["layout_score"])
                for spec_object, result in zip(spec.get("objects", []), object_results)
                if (spec_object.get("layout") or {}).get("type") == layout_type
            ]
            if layout_values:
                details[f"visual_logic_predicate_{predicate_type}"] = [float(np.mean(layout_values))]
                details[f"visual_logic_predicate_{predicate_type}_strict"] = [
                    float(np.mean([value >= 0.99 for value in layout_values]))
                ]
    return score, details


def _score_exact_count(observed_count: int, target_count: int) -> Tuple[int, float]:
    error = abs(observed_count - target_count)
    score = max(0.0, 1.0 - error / max(target_count, 1))
    return error, float(score)


def _effective_component_count(
    components: Sequence[Component],
    requested_shape: Optional[str],
) -> int:
    """Count merged instances without splitting shadow-heavy square masks."""

    return sum(
        (
            component.confidence_peak_multiplicity
            if component.confidence_peak_multiplicity > 1
            else (
                1
                if requested_shape == "square"
                and component.shape_scores.get("square", 0.0) < 0.20
                else component.multiplicity
            )
        )
        for component in components
    )


def _substantive_count_components(
    components: Sequence[Component],
    min_relative_area: float = 0.15,
) -> List[Component]:
    """Remove small shadow/fringe fragments when shape evidence is absent."""

    if not components:
        return []
    largest_area = max(component.area for component in components)
    return [
        component
        for component in components
        if component.area >= min_relative_area * largest_area
    ]


def _estimate_repeated_group_count(
    components: Sequence[Component],
    requested_shape: Optional[str] = None,
) -> int:
    """Estimate count when several similarly rendered instances merge.

    A visibly isolated component supplies an area unit. Larger connected
    components count as multiple instances only when their areas are close to
    integer multiples of that unit; otherwise the ordinary component
    multiplicities are retained.
    """

    observed = _effective_component_count(components, requested_shape)
    if len(components) < 3:
        return observed
    areas = sorted(float(component.area) for component in components)
    unit = areas[0]
    if unit <= 0.0 or areas[1] < 1.6 * unit or areas[-1] > 3.5 * unit:
        return observed
    ratios = [area / unit for area in areas]
    rounded = [max(1, int(round(ratio))) for ratio in ratios]
    if any(abs(ratio - count) > 0.36 for ratio, count in zip(ratios, rounded)):
        return observed
    return max(observed, sum(rounded))


def _fallback_repairs_fragmented_count(
    fallback_components: Sequence[Component],
    observed_components: Sequence[Component],
    requested_shape: str,
) -> bool:
    """Accept a lower fallback count only when it reunites split masks."""

    if not fallback_components or not observed_components:
        return False
    fallback_count = _effective_component_count(fallback_components, requested_shape)
    observed_count = _effective_component_count(observed_components, requested_shape)
    if fallback_count >= observed_count:
        return False
    if (
        _mean_shape_support(fallback_components, requested_shape)
        < _mean_shape_support(observed_components, requested_shape) + 0.20
    ):
        return False
    return all(
        any(
            _bbox_containment_fraction(fallback, observed) >= 0.90
            for fallback in fallback_components
        )
        for observed in observed_components
    )


def _score_shape_attribute(
    shape: str,
    components: Sequence[Component],
    *,
    allow_occluded_triangle: bool = False,
) -> float:
    if not components:
        return 0.0
    return float(
        np.mean(
            [
                _effective_shape_score(component, shape)
                if allow_occluded_triangle
                else component.shape_scores.get(shape, 0.0)
                for component in components
            ]
        )
    )


def _shape_binding_supported(
    components: Sequence[Component],
    requested_shape: str,
    *,
    allow_occluded_triangle: bool = False,
) -> bool:
    """Require the requested shape to survive a strong competing-shape test."""

    if not components:
        return False
    for component in components:
        if (
            allow_occluded_triangle
            and requested_shape == "triangle"
            and _is_occluded_triangle_component(component)
        ):
            continue
        requested = component.shape_scores.get(requested_shape, 0.0)
        competing = max(
            (
                score
                for shape, score in component.shape_scores.items()
                if shape != requested_shape
            ),
            default=0.0,
        )
        if (
            requested < _shape_presence_threshold(requested_shape)
            or requested < competing - 0.20
        ):
            return False
    return True


def _is_occluded_triangle_component(component: Component) -> bool:
    """Recognize a triangle whose interior is removed by another object."""

    aspect_ratio = max(component.aspect_ratio, 1.0 / max(component.aspect_ratio, 1e-6))
    return bool(
        component.hole_count >= 1
        and aspect_ratio <= 1.80
        and 0.25 <= component.fill_ratio <= 0.62
        and component.solidity >= 0.55
        and component.hull_vertex_count <= 24
        and component.shape_scores.get("triangle", 0.0) >= 0.10
    )


def _effective_shape_score(component: Component, requested_shape: str) -> float:
    score = float(component.shape_scores.get(requested_shape, 0.0))
    if requested_shape == "triangle" and _is_occluded_triangle_component(component):
        return max(score, 0.55)
    return score


def _shape_presence_threshold(shape: str) -> float:
    """Minimum evidence that a component depicts the requested shape in v4."""

    return 0.20 if shape == "triangle" else 0.40


def _component_shape_compatible(component: Component, requested_shape: str) -> bool:
    """Reject only components with strong evidence for a conflicting shape."""

    requested = float(component.shape_scores.get(requested_shape, 0.0))
    competing = max(
        (
            float(score)
            for shape, score in component.shape_scores.items()
            if shape != requested_shape
        ),
        default=0.0,
    )
    return bool(
        requested >= _shape_presence_threshold(requested_shape)
        or competing < 0.65
        or requested >= competing - 0.30
    )


def _component_all_binding_compatible(component: Component, requested_shape: str) -> bool:
    """Require each member of a quantified group to support the named shape."""

    requested = float(component.shape_scores.get(requested_shape, 0.0))
    competing = max(
        (
            float(score)
            for shape, score in component.shape_scores.items()
            if shape != requested_shape
        ),
        default=0.0,
    )
    return bool(
        requested >= _shape_presence_threshold(requested_shape)
        and requested >= competing - 0.10
    )


def _component_matches_exclusive_expected_shape(
    component: Component,
    requested_shape: str,
    expected_shapes: Sequence[str],
) -> bool:
    """Assign a same-color component to exactly one requested shape group."""

    ranked = sorted(
        (
            (_effective_shape_score(component, shape), shape)
            for shape in set(expected_shapes)
        ),
        key=lambda item: (-item[0], item[1]),
    )
    if not ranked or ranked[0][1] != requested_shape:
        return False
    return _component_all_binding_compatible(component, requested_shape)


def _component_identity_compatible(
    component: Component,
    requested_shape: str,
    image_hw: Tuple[int, int],
) -> bool:
    """Apply basic geometric sanity checks when a predicate names an object."""

    if requested_shape == "triangle":
        triangle_score = component.shape_scores.get("triangle", 0.0)
        aspect_ratio = max(component.aspect_ratio, 1.0 / max(component.aspect_ratio, 1e-6))
        if triangle_score >= 0.55 and aspect_ratio <= 2.50:
            return True
        if component.multiplicity > 1 and triangle_score >= 0.35:
            return True
        if _is_occluded_triangle_component(component):
            return True
        legacy_occluded_triangle = bool(
            aspect_ratio <= 1.80
            and 0.25 <= component.fill_ratio <= 0.62
            and component.solidity >= 0.55
            and component.hull_vertex_count <= 24
        )
        if triangle_score < 0.03 and legacy_occluded_triangle:
            return True
        if triangle_score < 0.03 and aspect_ratio <= 1.80:
            return False
        if aspect_ratio > 1.80:
            height, width = image_hw
            x0, y0, x1, y1 = component.bbox
            touches_frame = x0 <= 2 or y0 <= 2 or x1 >= width - 3 or y1 >= height - 3
            cropped_triangle = touches_frame and component.fill_ratio < 0.85
            if not cropped_triangle:
                return False
    return _component_shape_compatible(component, requested_shape)


def _component_count_compatible(
    component: Component,
    requested_shape: str,
    image_hw: Tuple[int, int],
) -> bool:
    """Require positive shape evidence before a component contributes to count."""

    requested = float(component.shape_scores.get(requested_shape, 0.0))
    if requested >= _shape_presence_threshold(requested_shape):
        return True
    competing = max(
        (float(score) for shape, score in component.shape_scores.items() if shape != requested_shape),
        default=0.0,
    )
    height, width = image_hw
    x0, y0, x1, y1 = component.bbox
    touches_frame = x0 <= 2 or y0 <= 2 or x1 >= width - 3 or y1 >= height - 3
    aspect_ratio = max(component.aspect_ratio, 1.0 / max(component.aspect_ratio, 1e-6))
    if (
        requested_shape == "square"
        and (aspect_ratio <= 1.80 or (touches_frame and aspect_ratio <= 2.50))
        and component.fill_ratio >= 0.75
        and component.solidity >= 0.85
        and competing < 0.70
    ):
        return True
    return bool(
        touches_frame
        and component.fill_ratio < 0.85
        and competing < 0.65
        and (requested >= 0.03 or competing < 0.15)
    )


def _topology_shape_supported(component: Component, requested_shape: str) -> bool:
    """Allow occluded topology referents while rejecting obvious wrong shapes."""

    if component.shape_scores.get(requested_shape, 0.0) >= 0.10:
        return True
    if requested_shape != "square":
        return False
    aspect_ratio = max(component.aspect_ratio, 1.0 / max(component.aspect_ratio, 1e-6))
    return bool(aspect_ratio <= 1.20 and component.fill_ratio >= 0.45)


def _outline_circle_components(components: Sequence[Component]) -> List[Component]:
    """Recover hollow circular containers from their visible outer boundary."""

    recovered: List[Component] = []
    for component in components:
        if component.boundary is None or len(component.boundary) < 12:
            continue
        x0, y0, x1, y1 = component.bbox
        width = x1 - x0 + 1
        height = y1 - y0 + 1
        aspect_ratio = max(component.aspect_ratio, 1.0 / max(component.aspect_ratio, 1e-6))
        boundary = np.asarray(component.boundary, dtype=np.float32)
        center = np.asarray(component.centroid, dtype=np.float32)
        radii = np.linalg.norm(boundary - center, axis=1)
        radial_variation = float(np.std(radii) / max(np.mean(radii), 1e-6))
        if (
            component.area >= 80
            and min(width, height) >= 8
            and component.color_confidence >= 0.40
            and component.hole_count >= 1
            and 0.03 <= component.fill_ratio <= 0.35
            and aspect_ratio <= 1.20
            and radial_variation <= 0.08
        ):
            recovered.append(component)
    return recovered


def _outline_square_components(components: Sequence[Component]) -> List[Component]:
    """Recover hollow square frames without admitting arbitrary thin fragments."""

    recovered: List[Component] = []
    for component in components:
        x0, y0, x1, y1 = component.bbox
        width = x1 - x0 + 1
        height = y1 - y0 + 1
        aspect_ratio = max(component.aspect_ratio, 1.0 / max(component.aspect_ratio, 1e-6))
        if (
            component.area >= 80
            and min(width, height) >= 8
            and component.color_confidence >= 0.40
            and component.hole_count >= 1
            and 0.03 <= component.fill_ratio <= 0.35
            and aspect_ratio <= 1.25
            and component.shape_scores.get("square", 0.0) >= 0.80
        ):
            recovered.append(component)
    return recovered


def _square_container_exterior_supported(component: Component) -> bool:
    """Recognize a square exterior independently of interior occlusions."""

    if component.boundary is None or len(component.boundary) == 0:
        return False
    x0, y0, x1, y1 = component.bbox
    width = x1 - x0 + 1
    height = y1 - y0 + 1
    aspect_ratio = max(component.aspect_ratio, 1.0 / max(component.aspect_ratio, 1e-6))
    if not (
        component.area >= 80
        and min(width, height) >= 20
        and component.color_confidence >= 0.40
        and component.hole_count >= 1
        and 0.10 <= component.fill_ratio <= 0.90
        and aspect_ratio <= 1.20
    ):
        return False

    boundary = np.asarray(component.boundary, dtype=np.float32)
    corners = np.asarray(
        ((x0, y0), (x0, y1), (x1, y0), (x1, y1)),
        dtype=np.float32,
    )
    tolerance = max(3.0, 0.12 * min(width, height))
    return all(
        float(np.min(np.linalg.norm(boundary - corner, axis=1))) <= tolerance
        for corner in corners
    )


def _relation_object_ids(relation: Dict[str, Any]) -> List[str]:
    """Return every object group needed to decide a relation."""

    object_ids = [
        relation.get("subject"),
        relation.get("object"),
        relation.get("anchor_a"),
        relation.get("anchor_b"),
    ]
    references = relation.get("references")
    if isinstance(references, list):
        object_ids.extend(references)
    return list(dict.fromkeys(str(object_id) for object_id in object_ids if object_id is not None))


def _v3_shape_strict_pass(
    result: Dict[str, Any],
    id_to_components: Dict[str, List[Component]],
    object_results: Sequence[Dict[str, Any]],
) -> bool:
    """Allow partial shape evidence only when another requested object occludes it."""

    if float(result["shape_score"]) >= 0.55:
        return True
    if float(result["shape_score"]) < 0.14:
        return False
    components = id_to_components.get(str(result["id"]), [])
    if not components:
        return False
    other_ids = [str(other["id"]) for other in object_results if other["id"] != result["id"]]
    return any(
        _bbox_intersection_fraction(component, other_component) >= 0.05
        for component in components
        for other_id in other_ids
        for other_component in id_to_components.get(other_id, [])
    )


def _score_color_shape_binding(count_score: float, shape_score: float) -> float:
    return float(count_score * shape_score)


def score_video_spec(video: np.ndarray, spec: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """Score a synthetic video spec using per-frame component tracks.

    Expected input layout is THWC. If a single image is passed, it is treated as a
    one-frame video so temporal predicates fail gracefully rather than crashing.
    """

    video = np.asarray(video)
    if video.ndim == 3:
        video = video[None, ...]
    if video.dtype != np.uint8:
        video = np.clip(np.rint(video), 0, 255).astype(np.uint8)
    errors = validate_spec(spec)
    if errors:
        return 0.0, {
            "visual_logic": [0.0],
            "visual_logic_strict": [0.0],
            "visual_logic_loose": [0.0],
            "visual_logic_valid_spec": [0.0],
            "visual_logic_temporal_score": [0.0],
            "visual_logic_temporal_strict": [0.0],
        }
    if video.shape[0] == 0:
        return 0.0, {
            "visual_logic": [0.0],
            "visual_logic_strict": [0.0],
            "visual_logic_loose": [0.0],
            "visual_logic_valid_spec": [1.0],
            "visual_logic_temporal_score": [0.0],
            "visual_logic_temporal_strict": [0.0],
            "visual_logic_video_detection_rate": [0.0],
            "visual_logic_motion_score": [0.0],
            "visual_logic_relative_motion_score": [0.0],
            "visual_logic_acceleration_score": [0.0],
            "visual_logic_size_change_score": [0.0],
            "visual_logic_video_count_score": [0.0],
        }

    first_score, first_details = score_image_spec(video[0], {**spec, "modality": "image"})
    frame_components = _build_frame_components(video, spec)
    tracks = _build_tracks(video, spec, frame_components=frame_components)
    object_specs = {obj["id"]: obj for obj in spec.get("objects", [])}
    temporal_results = []
    temporal_scores = []
    for temporal in spec.get("temporal", []):
        score, result = _score_temporal(
            temporal,
            tracks,
            video.shape[0],
            frame_components=frame_components,
            object_specs=object_specs,
        )
        temporal_scores.append(score)
        temporal_results.append(result)
    temporal_score = float(np.mean(temporal_scores)) if temporal_scores else 1.0
    score = 0.55 * first_score + 0.45 * temporal_score
    strict_pass = first_details["visual_logic_strict"][0] >= 1.0 and all(r["strict_pass"] for r in temporal_results)
    loose_pass = score >= 0.75
    details = dict(first_details)
    details.update(
        {
            "visual_logic": [float(score)],
            "visual_logic_strict": [1.0 if strict_pass else 0.0],
            "visual_logic_loose": [1.0 if loose_pass else 0.0],
            "visual_logic_temporal_score": [temporal_score],
            "visual_logic_temporal_strict": [float(np.mean([1.0 if r["strict_pass"] else 0.0 for r in temporal_results])) if temporal_results else 1.0],
            "visual_logic_video_detection_rate": [_temporal_result_mean(temporal_results, "detection_rate", default=1.0)],
            "visual_logic_motion_score": [
                _temporal_type_score(
                    temporal_results,
                    {"moves_direction", "stationary", "velocity_range", "trajectory_shape", "temporal_order"},
                )
            ],
            "visual_logic_relative_motion_score": [
                _temporal_type_score(temporal_results, {"relative_distance_change", "relative_position_change"})
            ],
            "visual_logic_acceleration_score": [_temporal_type_score(temporal_results, {"acceleration_trend"})],
            "visual_logic_size_change_score": [_temporal_type_score(temporal_results, {"size_change"})],
            "visual_logic_video_count_score": [_temporal_type_score(temporal_results, {"count_consistency"})],
        }
    )
    return float(score), details


def find_color_components(
    image: np.ndarray,
    color: str,
    background_color: Optional[str] = None,
    min_area: int = 80,
    *,
    palette: Optional[Dict[str, Color]] = None,
    include_boundary: bool = False,
    adaptive_color: bool = False,
    candidate_colors: Optional[Sequence[str]] = None,
    shape_version: int = 1,
    multiplicity_version: Optional[int] = None,
    closing_radius: Optional[int] = None,
    shape_hint: Optional[str] = None,
    hue_intervals: Optional[Dict[str, Tuple[float, float]]] = None,
) -> List[Component]:
    confidence_map = None
    if adaptive_color:
        mask, confidence_map = _adaptive_color_mask_and_confidence(
            image,
            color,
            background_color,
            palette=palette,
            candidate_colors=candidate_colors,
            hue_intervals=hue_intervals,
        )
        mask = _binary_dilation(_binary_erosion(mask, 1), 1)
        mask = close_small_mask_gaps(mask, radius=2 if closing_radius is None else closing_radius)
        min_area = min(min_area, max(40, int(round(mask.size * 0.00015))))
    else:
        mask = mask_for_color(image, color, background_color, palette=palette)
        mask = close_small_mask_gaps(mask, radius=1 if closing_radius is None else closing_radius)
    components = connected_components(
        mask,
        color=color,
        min_area=min_area,
        include_boundary=include_boundary,
        shape_version=shape_version,
        multiplicity_version=multiplicity_version,
        confidence_map=confidence_map,
        shape_hint=shape_hint,
    )
    if adaptive_color:
        max_area = 0.80 * mask.size
        components = [component for component in components if component.area <= max_area]
    return components


def close_small_mask_gaps(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """Close one- or two-pixel holes/seams without merging clearly separated objects."""

    mask = np.asarray(mask, dtype=bool)
    if radius <= 0 or mask.size == 0:
        return mask
    return _binary_erosion(_binary_dilation(mask, radius), radius)


def _binary_dilation(mask: np.ndarray, radius: int) -> np.ndarray:
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    out = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            out |= padded[dy : dy + h, dx : dx + w]
    return out


def _binary_erosion(mask: np.ndarray, radius: int) -> np.ndarray:
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    out = np.ones_like(mask, dtype=bool)
    h, w = mask.shape
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            out &= padded[dy : dy + h, dx : dx + w]
    return out


def mask_for_color(
    image: np.ndarray,
    color: str,
    background_color: Optional[str] = None,
    *,
    palette: Optional[Dict[str, Color]] = None,
) -> np.ndarray:
    palette = palette or COLOR_RGB
    arr = image.astype(np.float32)
    target = np.array(palette[color], dtype=np.float32)
    dist = np.linalg.norm(arr - target, axis=-1)
    palette_colors = np.array([palette[name] for name in palette], dtype=np.float32)
    palette_dist = np.linalg.norm(arr[..., None, :] - palette_colors[None, None, :, :], axis=-1)
    target_idx = list(palette).index(color)
    nearest_color = np.argmin(palette_dist, axis=-1)
    saturated = arr.max(axis=-1) - arr.min(axis=-1) > 40
    mask = (dist < 95) & saturated & (nearest_color == target_idx)
    if background_color in BACKGROUND_RGB:
        bg = np.array(BACKGROUND_RGB[background_color], dtype=np.float32)
        bg_dist = np.linalg.norm(arr - bg, axis=-1)
        mask = mask & (bg_dist > 75)
    return mask


def _rgb_to_hsv_arrays(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rgb = image.astype(np.float32) / 255.0
    maximum = rgb.max(axis=-1)
    minimum = rgb.min(axis=-1)
    chroma = maximum - minimum
    hue = np.zeros_like(maximum)
    nonzero = chroma > 1e-6
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    red_max = nonzero & (maximum == red)
    green_max = nonzero & (maximum == green)
    blue_max = nonzero & (maximum == blue)
    hue[red_max] = ((green[red_max] - blue[red_max]) / chroma[red_max]) % 6.0
    hue[green_max] = (blue[green_max] - red[green_max]) / chroma[green_max] + 2.0
    hue[blue_max] = (red[blue_max] - green[blue_max]) / chroma[blue_max] + 4.0
    hue *= 60.0
    saturation = np.divide(chroma, maximum, out=np.zeros_like(chroma), where=maximum > 1e-6)
    return hue, saturation, maximum, chroma * 255.0


def _circular_hue_distance(hue: np.ndarray, anchor: float) -> np.ndarray:
    return np.abs((hue - anchor + 180.0) % 360.0 - 180.0)


def _hue_interval_mask(
    hue: np.ndarray,
    color: str,
    hue_intervals: Optional[Dict[str, Tuple[float, float]]] = None,
) -> np.ndarray:
    low, high = (hue_intervals or VVR_BENCH_HUE_INTERVALS)[color]
    if low > high:
        return (hue >= low) | (hue < high)
    return (hue >= low) & (hue < high)


def _color_assignment_mask(
    hue: np.ndarray,
    saturation: np.ndarray,
    value: np.ndarray,
    color: str,
    hue_intervals: Dict[str, Tuple[float, float]],
) -> np.ndarray:
    """Return the exclusive perceptual color assignment for one V4 color."""

    assigned = _hue_interval_mask(hue, color, hue_intervals)
    if hue_intervals != VVR_BENCH_HUE_INTERVALS_V4:
        return assigned
    # Diffusion models often render pink as a light, desaturated red rather
    # than magenta. Keep saturated red in the red class while assigning this
    # pale-red region to pink. Value is required so dark red shading cannot be
    # relabeled as pink.
    pale_red = (
        ((hue >= 355.0) | (hue < 15.0))
        & (saturation >= 0.08)
        & (saturation <= 0.35)
        & (value >= 0.65)
    )
    if color == "pink":
        return assigned | pale_red
    if color == "red":
        return assigned & ~pale_red
    # Around 240 degrees, saturation and brightness determine whether people
    # perceive a generated object as electric blue or indigo/purple. Keep
    # saturated bright pixels blue and assign darker or muted pixels purple.
    blue_violet = (
        (hue >= 240.0)
        & (hue < 245.0)
        & ((saturation < 0.85) | (value < 0.85))
    )
    if color == "purple":
        return assigned | blue_violet
    if color == "blue":
        return assigned & ~blue_violet
    return assigned


def _maximum_hue_distance(
    color: str,
    hue_intervals: Optional[Dict[str, Tuple[float, float]]],
) -> float:
    if color == "green" and hue_intervals == VVR_BENCH_HUE_INTERVALS_V4:
        return 70.0
    return 55.0


def _estimated_background(image: np.ndarray) -> Tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    border_width = max(4, min(height, width) // 64)
    border = np.concatenate(
        [
            image[:border_width].reshape(-1, 3),
            image[-border_width:].reshape(-1, 3),
            image[:, :border_width].reshape(-1, 3),
            image[:, -border_width:].reshape(-1, 3),
        ],
        axis=0,
    ).astype(np.float32)
    background = np.median(border, axis=0)
    distances = np.linalg.norm(border - background, axis=-1)
    # Objects can be cropped by an image boundary. Estimate background noise
    # from the closest 80% of border pixels so those objects do not raise the
    # foreground threshold for the entire image.
    cutoff = float(np.quantile(distances, 0.80))
    background_pixels = distances[distances <= cutoff]
    noise = float(np.quantile(background_pixels, 0.95)) if len(background_pixels) else 0.0
    return background, min(24.0, max(8.0, noise + 4.0))


def adaptive_mask_for_color(
    image: np.ndarray,
    color: str,
    background_color: Optional[str] = None,
    *,
    palette: Optional[Dict[str, Color]] = None,
    candidate_colors: Optional[Sequence[str]] = None,
    hue_intervals: Optional[Dict[str, Tuple[float, float]]] = None,
) -> np.ndarray:
    """Segment a named color while tolerating model-specific shading.

    V1 used fixed RGB balls and a hard saturation cutoff. V2 estimates the
    rendered background from the image border, then assigns sufficiently
    contrasted chromatic pixels to perceptual hue anchors. Assignment remains
    deterministic and uses only pixels, the declared background, and the
    finite color vocabulary.
    """

    mask, _ = _adaptive_color_mask_and_confidence(
        image,
        color,
        background_color,
        palette=palette,
        candidate_colors=candidate_colors,
        hue_intervals=hue_intervals,
    )
    return mask


def _adaptive_color_pixel_counts(
    image: np.ndarray,
    candidate_colors: Sequence[str],
    hue_intervals: Dict[str, Tuple[float, float]],
) -> Dict[str, int]:
    """Count exclusive perceptual colors with one shared HSV conversion."""

    candidates = [name for name in candidate_colors if name in VVR_BENCH_HUE_DEGREES]
    background, contrast_threshold = _estimated_background(image)
    rgb_distance = np.linalg.norm(image.astype(np.float32) - background, axis=-1)
    hue, saturation, value, chroma = _rgb_to_hsv_arrays(image)
    background_hue, background_saturation, background_value, _ = _rgb_to_hsv_arrays(
        background.reshape(1, 1, 3)
    )
    foreground = (rgb_distance > contrast_threshold) & (chroma > 5.0) & (saturation > 0.03)
    counts: Dict[str, int] = {}
    for color in candidates:
        color_foreground = foreground
        background_matches_target = bool(
            background_saturation[0, 0] >= 0.05
            and _color_assignment_mask(
                background_hue,
                background_saturation,
                background_value,
                color,
                hue_intervals,
            )[0, 0]
        )
        if background_matches_target:
            color_foreground = foreground & (
                (np.abs(saturation - float(background_saturation[0, 0])) > 0.06)
                | (np.abs(value - float(background_value[0, 0])) > 0.06)
                | (rgb_distance > 24.0)
            )
        target_distance = _circular_hue_distance(hue, VVR_BENCH_HUE_DEGREES[color])
        mask = (
            color_foreground
            & _color_assignment_mask(hue, saturation, value, color, hue_intervals)
            & (target_distance <= _maximum_hue_distance(color, hue_intervals))
        )
        counts[color] = int(np.count_nonzero(mask))
    return counts


def find_foreground_components(
    image: np.ndarray,
    background_color: Optional[str] = None,
    *,
    shape_hint: Optional[str] = None,
    shape_version: int = 2,
) -> List[Component]:
    """Find salient chromatic objects without assuming their named color."""

    del background_color
    background, contrast_threshold = _estimated_background(image)
    rgb_distance = np.linalg.norm(image.astype(np.float32) - background, axis=-1)
    _, saturation, _, chroma = _rgb_to_hsv_arrays(image)
    mask = (rgb_distance > contrast_threshold) & (chroma > 5.0) & (saturation > 0.03)
    mask = _binary_dilation(_binary_erosion(mask, 1), 1)
    mask = close_small_mask_gaps(mask, radius=2)
    confidence = (
        0.60 * np.clip(rgb_distance / 40.0, 0.0, 1.0)
        + 0.40 * np.clip(saturation / 0.50, 0.0, 1.0)
    ).astype(np.float32)
    return connected_components(
        mask,
        color="foreground",
        min_area=max(40, int(round(mask.size * 0.00015))),
        include_boundary=True,
        shape_version=shape_version,
        confidence_map=confidence,
        shape_hint=shape_hint,
    )


def _adaptive_color_mask_and_confidence(
    image: np.ndarray,
    color: str,
    background_color: Optional[str] = None,
    *,
    palette: Optional[Dict[str, Color]] = None,
    candidate_colors: Optional[Sequence[str]] = None,
    hue_intervals: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    del background_color
    palette = palette or VVR_BENCH_COLOR_RGB
    hue_intervals = hue_intervals or VVR_BENCH_HUE_INTERVALS
    candidates = [name for name in (candidate_colors or tuple(palette)) if name in VVR_BENCH_HUE_DEGREES]
    if color not in candidates:
        empty = np.zeros(image.shape[:2], dtype=np.float32)
        return empty.astype(bool), empty
    background, contrast_threshold = _estimated_background(image)
    rgb_distance = np.linalg.norm(image.astype(np.float32) - background, axis=-1)
    hue, saturation, value, chroma = _rgb_to_hsv_arrays(image)
    anchors = np.asarray([VVR_BENCH_HUE_DEGREES[name] for name in candidates], dtype=np.float32)
    hue_distances = np.stack([_circular_hue_distance(hue, float(anchor)) for anchor in anchors], axis=-1)
    target_index = candidates.index(color)
    target_distance = hue_distances[..., target_index]
    if set(candidates) == set(hue_intervals):
        target_assignment = _color_assignment_mask(
            hue,
            saturation,
            value,
            color,
            hue_intervals,
        )
    else:
        target_assignment = np.argmin(hue_distances, axis=-1) == target_index

    background_hue, background_saturation, background_value, _ = _rgb_to_hsv_arrays(
        background.reshape(1, 1, 3)
    )
    if set(candidates) == set(hue_intervals):
        background_matches_target = bool(
            background_saturation[0, 0] >= 0.05
            and _color_assignment_mask(
                background_hue,
                background_saturation,
                background_value,
                color,
                hue_intervals,
            )[0, 0]
        )
    else:
        background_nearest = int(
            np.argmin(
                [_circular_hue_distance(background_hue, float(anchor))[0, 0] for anchor in anchors]
            )
        )
        background_matches_target = bool(
            background_saturation[0, 0] >= 0.05
            and target_index == background_nearest
        )
    foreground = (rgb_distance > contrast_threshold) & (chroma > 5.0) & (saturation > 0.03)
    if background_matches_target:
        foreground &= (
            (np.abs(saturation - float(background_saturation[0, 0])) > 0.06)
            | (np.abs(value - float(background_value[0, 0])) > 0.06)
            | (rgb_distance > 24.0)
        )
    hue_radius = _maximum_hue_distance(color, hue_intervals)
    mask = foreground & target_assignment & (target_distance <= hue_radius)
    hue_confidence = np.clip(1.0 - target_distance / hue_radius, 0.0, 1.0)
    saturation_confidence = np.clip(saturation / 0.50, 0.0, 1.0)
    contrast_confidence = np.clip(rgb_distance / 40.0, 0.0, 1.0)
    confidence = 0.65 * hue_confidence + 0.20 * saturation_confidence + 0.15 * contrast_confidence
    return mask, confidence.astype(np.float32)


def _countable_components(
    components: Sequence[Component],
    *,
    min_dimension: int = 8,
    min_color_confidence: float = 0.40,
    min_relative_area: float = 0.02,
) -> List[Component]:
    result: List[Component] = []
    for component in components:
        x0, y0, x1, y1 = component.bbox
        width = x1 - x0 + 1
        height = y1 - y0 + 1
        if (
            component.area < 80
            or min(width, height) < min_dimension
            or component.fill_ratio < 0.18
            or component.color_confidence < min_color_confidence
        ):
            continue
        result.append(component)
    if result:
        largest_area = max(component.area for component in result)
        result = [component for component in result if component.area >= min_relative_area * largest_area]
    filtered = []
    for component in result:
        x0, y0, x1, y1 = component.bbox
        nested = False
        for other in result:
            if other is component or other.area < 4 * component.area:
                continue
            ox0, oy0, ox1, oy1 = other.bbox
            cx, cy = component.centroid
            if ox0 <= cx <= ox1 and oy0 <= cy <= oy1 and x0 >= ox0 - 2 and y0 >= oy0 - 2 and x1 <= ox1 + 2 and y1 <= oy1 + 2:
                nested = True
                break
        if not nested:
            filtered.append(component)
    return filtered


def _mean_shape_support(components: Sequence[Component], shape: str) -> float:
    total_area = sum(component.area for component in components)
    if total_area <= 0:
        return 0.0
    return float(
        sum(component.area * component.shape_scores.get(shape, 0.0) for component in components)
        / total_area
    )


def _component_match_quality(component: Component, shape: str) -> float:
    size_support = min(1.0, math.log1p(component.area) / math.log(10000.0))
    return (
        0.35 * component.color_confidence
        + 0.30 * component.shape_scores.get(shape, 0.0)
        + 0.35 * size_support
    )


def _dedupe_component_candidates(components: Sequence[Component], shape: str) -> List[Component]:
    ranked = sorted(components, key=lambda component: _component_match_quality(component, shape), reverse=True)
    kept: List[Component] = []
    for component in ranked:
        if any(_components_alias([component], [other]) for other in kept):
            continue
        kept.append(component)
    return kept


def _filter_fallback_aliases(
    components: Sequence[Component],
    *,
    target_color: str,
    components_by_color: Dict[str, List[Component]],
    expected_shape_by_color: Dict[str, str],
    strict_named_color_identity: bool = False,
) -> List[Component]:
    """Drop fallback regions better explained by another expected object.

    Single-color fallback masks deliberately tolerate hue shifts. Near a
    neighboring color, that can segment the same object twice. A fallback is
    rejected only when the globally assigned component has substantially
    stronger color evidence and supports that other object's expected shape.
    """

    kept: List[Component] = []
    for component in components:
        aliased = False
        for other_color, other_components in components_by_color.items():
            if other_color == target_color or other_color not in expected_shape_by_color:
                continue
            expected_shape = expected_shape_by_color[other_color]
            for other in other_components:
                if (
                    strict_named_color_identity
                    and _bbox_containment_fraction(component, other) >= 0.90
                    and component.area >= 1.25 * other.area
                    and other.area >= 0.30 * component.area
                ):
                    aliased = True
                    break
                overlaps_same_object = (
                    _components_alias_same_referent([component], [other])
                    if strict_named_color_identity
                    else _bbox_intersection_fraction(component, other) >= 0.65
                )
                if not overlaps_same_object:
                    continue
                if (
                    other.color_confidence >= component.color_confidence + 0.08
                    and (
                        strict_named_color_identity
                        or other.shape_scores.get(expected_shape, 0.0) >= 0.50
                    )
                ):
                    aliased = True
                    break
            if aliased:
                break
        if not aliased:
            kept.append(component)
    return kept


def _bbox_intersection_fraction(first: Component, second: Component) -> float:
    ax0, ay0, ax1, ay1 = first.bbox
    bx0, by0, bx1, by1 = second.bbox
    width = max(0, min(ax1, bx1) - max(ax0, bx0) + 1)
    height = max(0, min(ay1, by1) - max(ay0, by0) + 1)
    intersection = width * height
    first_box_area = (ax1 - ax0 + 1) * (ay1 - ay0 + 1)
    second_box_area = (bx1 - bx0 + 1) * (by1 - by0 + 1)
    return intersection / max(min(first_box_area, second_box_area), 1)


def _bbox_larger_intersection_fraction(first: Component, second: Component) -> float:
    ax0, ay0, ax1, ay1 = first.bbox
    bx0, by0, bx1, by1 = second.bbox
    width = max(0, min(ax1, bx1) - max(ax0, bx0) + 1)
    height = max(0, min(ay1, by1) - max(ay0, by0) + 1)
    intersection = width * height
    first_box_area = (ax1 - ax0 + 1) * (ay1 - ay0 + 1)
    second_box_area = (bx1 - bx0 + 1) * (by1 - by0 + 1)
    return intersection / max(max(first_box_area, second_box_area), 1)


def _bbox_containment_fraction(container: Component, contained: Component) -> float:
    """Fraction of the contained component's box covered by the container."""

    ax0, ay0, ax1, ay1 = container.bbox
    bx0, by0, bx1, by1 = contained.bbox
    width = max(0, min(ax1, bx1) - max(ax0, bx0) + 1)
    height = max(0, min(ay1, by1) - max(ay0, by0) + 1)
    return width * height / max((bx1 - bx0 + 1) * (by1 - by0 + 1), 1)


def _suppress_cross_color_fragments(
    components_by_color: Dict[str, List[Component]],
    *,
    require_similar_extent: bool = False,
) -> Dict[str, List[Component]]:
    cleaned: Dict[str, List[Component]] = {}
    all_components = [
        (color, component)
        for color, components in components_by_color.items()
        for component in components
    ]
    for color, components in components_by_color.items():
        kept = []
        for component in components:
            fragment = any(
                other_color != color
                and other.area >= 3.0 * component.area
                and _bbox_intersection_fraction(component, other) >= 0.70
                and (
                    not require_similar_extent
                    or _bbox_larger_intersection_fraction(component, other) >= 0.65
                )
                for other_color, other in all_components
            )
            if not fragment:
                kept.append(component)
        cleaned[color] = kept
    return cleaned


def _suppress_aliased_geometry_candidates(
    components_by_color: Dict[str, List[Component]],
    object_specs: Sequence[Dict[str, Any]],
) -> Dict[str, List[Component]]:
    expected_shape = {str(item.get("color")): str(item.get("shape", "circle")) for item in object_specs}
    cleaned: Dict[str, List[Component]] = {}
    for color, components in components_by_color.items():
        kept = []
        for component in components:
            aliased_to_stronger = False
            for other_color, other_components in components_by_color.items():
                if other_color == color or other_color not in expected_shape:
                    continue
                for other in other_components:
                    if not _components_alias([component], [other]):
                        continue
                    own_shape = component.shape_scores.get(expected_shape.get(color, ""), 0.0)
                    other_shape = other.shape_scores.get(expected_shape[other_color], 0.0)
                    if other.area >= 1.5 * component.area and other_shape >= own_shape and own_shape < 0.20:
                        aliased_to_stronger = True
                        break
                if aliased_to_stronger:
                    break
            if not aliased_to_stronger:
                kept.append(component)
        cleaned[color] = kept
    return cleaned


def _components_alias(first: Sequence[Component], second: Sequence[Component]) -> bool:
    if len(first) != 1 or len(second) != 1:
        return False
    a, b = first[0], second[0]
    if _bbox_intersection_fraction(a, b) < 0.80:
        return False
    extent = max(1.0, min(_component_extent(a), _component_extent(b)))
    return math.dist(a.centroid, b.centroid) <= 0.15 * extent


def _components_alias_same_referent(first: Sequence[Component], second: Sequence[Component]) -> bool:
    """Detect duplicate masks for one object without conflating nested objects."""

    if len(first) != 1 or len(second) != 1:
        return False
    a, b = first[0], second[0]
    if _bbox_intersection_fraction(a, b) < 0.80:
        return False
    extent_ratio = min(_component_extent(a), _component_extent(b)) / max(
        _component_extent(a),
        _component_extent(b),
        1.0,
    )
    if extent_ratio < 0.65:
        return False
    extent = max(1.0, min(_component_extent(a), _component_extent(b)))
    return math.dist(a.centroid, b.centroid) <= 0.22 * extent


def connected_components(
    mask: np.ndarray,
    color: str,
    min_area: int = 80,
    *,
    include_boundary: bool = False,
    shape_version: int = 1,
    multiplicity_version: Optional[int] = None,
    confidence_map: Optional[np.ndarray] = None,
    shape_hint: Optional[str] = None,
) -> List[Component]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: List[Component] = []
    for y in range(h):
        xs = np.flatnonzero(mask[y] & ~seen[y])
        for x0 in xs:
            if seen[y, x0] or not mask[y, x0]:
                continue
            stack = [(y, int(x0))]
            seen[y, x0] = True
            pixels = []
            while stack:
                cy, cx = stack.pop()
                pixels.append((cy, cx))
                for ny in (cy - 1, cy, cy + 1):
                    for nx in (cx - 1, cx, cx + 1):
                        if ny < 0 or ny >= h or nx < 0 or nx >= w:
                            continue
                        if seen[ny, nx] or not mask[ny, nx]:
                            continue
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            if len(pixels) < min_area:
                continue
            ys = np.array([p[0] for p in pixels], dtype=np.float32)
            xs_arr = np.array([p[1] for p in pixels], dtype=np.float32)
            y_min, y_max = int(ys.min()), int(ys.max())
            x_min, x_max = int(xs_arr.min()), int(xs_arr.max())
            bbox_w = max(1, x_max - x_min + 1)
            bbox_h = max(1, y_max - y_min + 1)
            fill_ratio = float(len(pixels) / (bbox_w * bbox_h))
            aspect_ratio = float(bbox_w / bbox_h)
            pixel_set = set(pixels)
            boundary = None
            boundary_points: List[Tuple[float, float]] = []
            if include_boundary:
                boundary_points = [
                    (float(px), float(py))
                    for py, px in pixels
                    if any((py + dy, px + dx) not in pixel_set for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)))
                ]
                boundary = np.asarray(boundary_points, dtype=np.float32)
            hull = _convex_hull(boundary_points) if include_boundary else []
            solidity = _component_solidity(hull, len(pixels)) if include_boundary else 1.0
            hole_count = _component_hole_count(pixel_set, (x_min, y_min, x_max, y_max)) if include_boundary else 0
            bbox_center = ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0)
            extent = math.sqrt(bbox_w * bbox_h)
            robust_width = float(np.quantile(xs_arr, 0.95) - np.quantile(xs_arr, 0.05) + 1.0)
            robust_height = float(np.quantile(ys, 0.95) - np.quantile(ys, 0.05) + 1.0)
            robust_extent = math.sqrt(max(robust_width, 1.0) * max(robust_height, 1.0))
            centroid_offset = math.hypot(float(xs_arr.mean()) - bbox_center[0], float(ys.mean()) - bbox_center[1]) / max(extent, 1.0)
            shape_scores = classify_shape(
                len(pixels),
                bbox_w,
                bbox_h,
                fill_ratio,
                solidity=solidity,
                hull_vertex_count=len(hull),
                hole_count=hole_count,
                centroid_offset=centroid_offset,
                benchmark=include_boundary,
                benchmark_version=shape_version,
            )
            if (
                (
                    (shape_version >= 2 and shape_hint == "triangle")
                    or (shape_version >= 3 and shape_hint is not None)
                )
                and confidence_map is not None
                and len(pixels) >= 80
            ):
                confidence_values = np.asarray(
                    [confidence_map[int(py), int(px)] for py, px in pixels],
                    dtype=np.float32,
                )
                core_threshold = max(0.72, float(np.quantile(confidence_values, 0.75)))
                core_mask = np.zeros((bbox_h, bbox_w), dtype=bool)
                for py, px in pixels:
                    if confidence_map[int(py), int(px)] >= core_threshold:
                        core_mask[int(py) - y_min, int(px) - x_min] = True
                core_components = connected_components(
                    core_mask,
                    color=color,
                    min_area=max(20, int(round(len(pixels) * 0.08))),
                    include_boundary=True,
                    shape_version=shape_version,
                )
                if core_components:
                    core = max(core_components, key=lambda component: component.area)
                    shape_scores = {
                        shape: max(score, core.shape_scores.get(shape, 0.0))
                        for shape, score in shape_scores.items()
                    }
            effective_multiplicity_version = (
                shape_version if multiplicity_version is None else multiplicity_version
            )
            multiplicity = (
                _estimate_touching_multiplicity(
                    pixel_set,
                    (x_min, y_min, x_max, y_max),
                    hole_count=hole_count,
                    shape_version=effective_multiplicity_version,
                )
                if shape_version >= 2
                else 1
            )
            confidence_peak_multiplicity = 1
            if (
                effective_multiplicity_version >= 3
                and multiplicity == 1
                and confidence_map is not None
                and shape_hint is not None
                and fill_ratio < 0.70
                and shape_scores.get(shape_hint, 0.0) < 0.03
                and max(
                    (score for shape, score in shape_scores.items() if shape != shape_hint),
                    default=0.0,
                )
                < 0.65
            ):
                confidence_values = np.asarray(
                    [confidence_map[int(py), int(px)] for py, px in pixels],
                    dtype=np.float32,
                )
                confidence_core = np.zeros((bbox_h, bbox_w), dtype=bool)
                confidence_threshold = float(np.quantile(confidence_values, 0.60))
                for py, px in pixels:
                    if confidence_map[int(py), int(px)] >= confidence_threshold:
                        confidence_core[int(py) - y_min, int(px) - x_min] = True
                core_areas = [
                    core_area
                    for core_area in _mask_component_areas(confidence_core)
                    if core_area >= max(8, int(round(len(pixels) * 0.03)))
                ]
                if (
                    1 < len(core_areas) <= 8
                    and sum(core_areas) >= 0.25 * len(pixels)
                    and max(core_areas) <= 5 * min(core_areas)
                ):
                    confidence_peak_multiplicity = len(core_areas)
                    multiplicity = confidence_peak_multiplicity
            color_confidence = (
                float(np.mean([confidence_map[int(py), int(px)] for py, px in pixels]))
                if confidence_map is not None
                else 1.0
            )
            components.append(
                Component(
                    color=color,
                    area=len(pixels),
                    bbox=(x_min, y_min, x_max, y_max),
                    centroid=(float(xs_arr.mean()), float(ys.mean())),
                    aspect_ratio=aspect_ratio,
                    fill_ratio=fill_ratio,
                    shape_scores=shape_scores,
                    boundary=boundary,
                    solidity=solidity,
                    hull_vertex_count=len(hull),
                    hole_count=hole_count,
                    centroid_offset=centroid_offset,
                    multiplicity=multiplicity,
                    robust_extent=robust_extent if shape_version >= 2 else 0.0,
                    color_confidence=color_confidence,
                    confidence_peak_multiplicity=confidence_peak_multiplicity,
                )
            )
    return components


def _estimate_touching_multiplicity(
    pixels: set[Tuple[int, int]],
    bbox: Tuple[int, int, int, int],
    *,
    hole_count: int = 0,
    shape_version: int = 2,
) -> int:
    """Estimate visibly repeated same-color objects joined at their boundaries.

    Count prompts frequently produce touching circles that ordinary connected
    components merge. Progressive erosion separates narrow contact necks. The
    estimate is bounded by the component's axis ratio so a rough single object
    cannot fragment into an arbitrary number of instances.
    """

    x0, y0, x1, y1 = bbox
    if hole_count > 0:
        return 1
    width = x1 - x0 + 1
    height = y1 - y0 + 1
    axis_ratio = max(width, height) / max(min(width, height), 1)
    local = np.zeros((height, width), dtype=bool)
    for y, x in pixels:
        local[y - y0, x - x0] = True
    if shape_version >= 3 and axis_ratio < 1.35:
        depth = np.zeros_like(local, dtype=np.int16)
        current = local.copy()
        level = 0
        while current.any() and level < min(width, height) // 2:
            level += 1
            current = _binary_erosion(current, 1)
            depth[current] = level
        maximum_depth = int(depth.max())
        if maximum_depth > 0:
            core = depth >= int(math.ceil(0.65 * maximum_depth))
            areas = _mask_component_areas(core)
            substantial = [area for area in areas if area >= max(8, int(round(len(pixels) * 0.03)))]
            if 1 < len(substantial) <= 4 and max(substantial) <= 5 * min(substantial):
                return len(substantial)
        return 1
    if axis_ratio < 1.35:
        return 1
    max_parts = min(8, max(1, int(math.ceil(axis_ratio - 0.15))))
    if max_parts <= 1:
        return 1
    best = 1
    current = local
    for _ in range(3):
        current = _binary_erosion(current, 1)
        areas = _mask_component_areas(current)
        substantial = [area for area in areas if area >= max(8, int(round(len(pixels) * 0.06)))]
        if 1 < len(substantial) <= max_parts and max(substantial) <= 5 * min(substantial):
            best = max(best, len(substantial))
    profile = local.sum(axis=1 if height >= width else 0).astype(np.float32)
    smoothing_width = max(3, int(round(min(width, height) * 0.07)))
    if smoothing_width % 2 == 0:
        smoothing_width += 1
    smooth = np.convolve(profile, np.ones(smoothing_width) / smoothing_width, mode="same")
    peak_floor = 0.65 * float(smooth.max()) if len(smooth) else 0.0
    min_separation = max(6, int(round(min(width, height) * 0.35)))
    peaks: List[int] = []
    for index in range(1, max(1, len(smooth) - 1)):
        if smooth[index] < peak_floor or smooth[index] < smooth[index - 1] or smooth[index] < smooth[index + 1]:
            continue
        if not peaks or index - peaks[-1] >= min_separation:
            peaks.append(index)
        elif smooth[index] > smooth[peaks[-1]]:
            peaks[-1] = index
    prominent_peaks: List[int] = []
    for peak in peaks:
        if not prominent_peaks:
            prominent_peaks.append(peak)
            continue
        previous = prominent_peaks[-1]
        valley = float(smooth[previous : peak + 1].min())
        if valley <= 0.72 * min(float(smooth[previous]), float(smooth[peak])):
            prominent_peaks.append(peak)
        elif smooth[peak] > smooth[previous]:
            prominent_peaks[-1] = peak
    if 1 < len(prominent_peaks) <= max_parts:
        best = max(best, len(prominent_peaks))
    return best


def _mask_component_areas(mask: np.ndarray) -> List[int]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    areas: List[int] = []
    for y in range(height):
        for x in np.flatnonzero(mask[y] & ~seen[y]):
            if seen[y, x]:
                continue
            stack = [(y, int(x))]
            seen[y, x] = True
            area = 0
            while stack:
                cy, cx = stack.pop()
                area += 1
                for ny in (cy - 1, cy, cy + 1):
                    for nx in (cx - 1, cx, cx + 1):
                        if ny < 0 or ny >= height or nx < 0 or nx >= width:
                            continue
                        if seen[ny, nx] or not mask[ny, nx]:
                            continue
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            areas.append(area)
    return areas


def _convex_hull(points: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    points = sorted(set(points))
    if len(points) < 3:
        return points

    def cross(origin: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])

    lower: List[Tuple[float, float]] = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: List[Tuple[float, float]] = []
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def _component_solidity(hull: Sequence[Tuple[float, float]], pixel_area: int) -> float:
    """Return component area divided by its convex-hull area.

    Pixel centers slightly underestimate hull area, so the denominator receives
    a one-pixel perimeter correction. The value is used only by the versioned
    benchmark shape classifier; legacy circle/square scores are unchanged.
    """

    if len(hull) < 3:
        return 0.0
    twice_area = abs(
        sum(
            hull[idx][0] * hull[(idx + 1) % len(hull)][1]
            - hull[(idx + 1) % len(hull)][0] * hull[idx][1]
            for idx in range(len(hull))
        )
    )
    hull_area = max(twice_area / 2.0 + math.sqrt(pixel_area), 1.0)
    return float(min(1.0, pixel_area / hull_area))


def _component_hole_count(
    pixels: set[Tuple[int, int]],
    bbox: Tuple[int, int, int, int],
) -> int:
    x0, y0, x1, y1 = bbox
    width = x1 - x0 + 3
    height = y1 - y0 + 3
    occupied = np.zeros((height, width), dtype=bool)
    for y, x in pixels:
        occupied[y - y0 + 1, x - x0 + 1] = True
    exterior = np.zeros_like(occupied)
    stack = [(0, 0)]
    exterior[0, 0] = True
    while stack:
        y, x = stack.pop()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if ny < 0 or ny >= height or nx < 0 or nx >= width:
                continue
            if exterior[ny, nx] or occupied[ny, nx]:
                continue
            exterior[ny, nx] = True
            stack.append((ny, nx))
    unseen = ~occupied & ~exterior
    holes = 0
    for y in range(height):
        for x in range(width):
            if not unseen[y, x]:
                continue
            holes += 1
            stack = [(y, x)]
            unseen[y, x] = False
            while stack:
                cy, cx = stack.pop()
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < height and 0 <= nx < width and unseen[ny, nx]:
                        unseen[ny, nx] = False
                        stack.append((ny, nx))
    return holes


def _component_extent(component: Component) -> float:
    if component.robust_extent > 0.0:
        return component.robust_extent
    x0, y0, x1, y1 = component.bbox
    return float(math.sqrt(max(x1 - x0 + 1, 1) * max(y1 - y0 + 1, 1)))


def _component_visual_center(component: Component) -> Tuple[float, float]:
    x0, y0, x1, y1 = component.bbox
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def _indistinguishable_cyan_blue_bindings(
    image: np.ndarray,
    object_specs: Sequence[Dict[str, Any]],
    id_to_components: Dict[str, List[Component]],
) -> set[str]:
    """Flag separate cyan/blue referents rendered with the same visible hue."""

    by_color = {str(item.get("color")): str(item.get("id")) for item in object_specs}
    if "cyan" not in by_color or "blue" not in by_color:
        return set()
    cyan_id, blue_id = by_color["cyan"], by_color["blue"]
    cyan_components = id_to_components.get(cyan_id, [])
    blue_components = id_to_components.get(blue_id, [])
    if len(cyan_components) != 1 or len(blue_components) != 1:
        return set()
    cyan_component, blue_component = cyan_components[0], blue_components[0]
    if _bbox_intersection_fraction(cyan_component, blue_component) >= 0.20:
        return set()

    hue, saturation, value, _ = _rgb_to_hsv_arrays(image)

    def median_hue(component: Component) -> Optional[float]:
        x0, y0, x1, y1 = component.bbox
        local_hue = hue[y0 : y1 + 1, x0 : x1 + 1]
        local_saturation = saturation[y0 : y1 + 1, x0 : x1 + 1]
        local_value = value[y0 : y1 + 1, x0 : x1 + 1]
        chromatic = (local_saturation >= 0.20) & (local_value >= 0.15)
        if not chromatic.any():
            return None
        return float(np.median(local_hue[chromatic]))

    cyan_hue = median_hue(cyan_component)
    blue_hue = median_hue(blue_component)
    if cyan_hue is None or blue_hue is None:
        return set()
    if float(_circular_hue_distance(np.asarray(cyan_hue), blue_hue)) <= 6.0:
        return {cyan_id, blue_id}
    return set()


def _mean_visual_center(components: Sequence[Component]) -> Tuple[float, float]:
    centers = [_component_visual_center(component) for component in components]
    return (
        float(np.mean([center[0] for center in centers])),
        float(np.mean([center[1] for center in centers])),
    )


def _score_size(
    size: Optional[str],
    components: Sequence[Component],
    image_hw: Tuple[int, int],
    *,
    benchmark: bool = False,
) -> float:
    if not size:
        return 1.0
    if not components:
        return 0.0
    h, w = image_hw
    image_extent = max(1, min(h, w))
    extent_ratios = np.array([_component_extent(comp) / image_extent for comp in components], dtype=np.float32)
    mean_extent_ratio = float(extent_ratios.mean())
    if benchmark:
        if size == "small":
            if mean_extent_ratio <= 0.16:
                return 1.0
            return float(max(0.0, 1.0 - (mean_extent_ratio - 0.16) / 0.14))
        if size == "large":
            if mean_extent_ratio >= 0.30:
                return 1.0
            return float(max(0.0, 1.0 - (0.30 - mean_extent_ratio) / 0.14))
        return 0.0
    image_area = max(1, h * w)
    ratios = np.array([comp.area / image_area for comp in components], dtype=np.float32)
    mean_ratio = float(ratios.mean())
    if size == "small":
        return _interval_score(mean_ratio, 0.002, 0.025)
    if size == "medium":
        return _interval_score(mean_ratio, 0.018, 0.070)
    if size == "large":
        return _interval_score(mean_ratio, 0.055, 0.180)
    return 0.0


def _interval_score(value: float, low: float, high: float) -> float:
    if low <= value <= high:
        return 1.0
    center = (low + high) / 2.0
    half_width = max((high - low) / 2.0, 1e-6)
    return float(max(0.0, 1.0 - abs(value - center) / (3.0 * half_width)))


def classify_shape(
    area: int,
    bbox_w: int,
    bbox_h: int,
    fill_ratio: float,
    *,
    solidity: float = 1.0,
    hull_vertex_count: int = 0,
    hole_count: int = 0,
    centroid_offset: float = 0.0,
    benchmark: bool = False,
    benchmark_version: int = 1,
) -> Dict[str, float]:
    del area
    aspect_score = max(0.0, 1.0 - abs(math.log(max(bbox_w, 1) / max(bbox_h, 1))) / math.log(2.0))
    circle_fill_score = max(0.0, 1.0 - abs(fill_ratio - 0.785) / 0.35)
    square_fill_score = max(0.0, 1.0 - abs(fill_ratio - 1.0) / 0.45)
    triangle_fill_score = max(0.0, 1.0 - abs(fill_ratio - 0.50) / 0.28)
    convex_score = min(1.0, max(0.0, (solidity - 0.82) / 0.14))
    if not benchmark:
        return {
            "circle": float(aspect_score * circle_fill_score),
            "square": float(aspect_score * square_fill_score),
            "triangle": float(max(0.0, aspect_score) * triangle_fill_score * convex_score),
        }

    if benchmark_version >= 2:
        aspect_ratio = max(bbox_w, bbox_h) / max(min(bbox_w, bbox_h), 1)
        circle_aspect = min(1.0, max(0.0, (2.15 - aspect_ratio) / 0.35))
        circle_fill = min(1.0, max(0.0, 1.0 - abs(fill_ratio - 0.76) / 0.24))
        circle_solidity = min(1.0, max(0.0, (solidity - 0.76) / 0.18))
        circle_boundary = min(1.0, max(0.0, (hull_vertex_count - 10) / 10.0))
        outlined_circle = 1.0 if hole_count == 1 and hull_vertex_count >= 14 and aspect_ratio <= 2.15 else 0.0
        circle_score = circle_aspect * max(circle_fill * circle_solidity * circle_boundary, outlined_circle)

        square_aspect = min(1.0, max(0.0, (1.45 - aspect_ratio) / 0.15))
        axis_square = min(1.0, max(0.0, (fill_ratio - 0.78) / 0.14))
        # A rotated square occupies roughly half of its axis-aligned bounding
        # box. High convexity and a centered centroid distinguish it from a
        # triangle with a similar fill ratio.
        rotated_square = (
            min(1.0, max(0.0, 1.0 - abs(fill_ratio - 0.52) / 0.18))
            * min(1.0, max(0.0, (solidity - 0.76) / 0.18))
            * min(1.0, max(0.0, (0.10 - centroid_offset) / 0.06))
        )
        outlined_square = 1.0 if hole_count == 1 and aspect_ratio <= 1.45 and centroid_offset <= 0.12 else 0.0
        square_score = square_aspect * max(axis_square, rotated_square, outlined_square)

        if benchmark_version >= 3:
            # Shadows and softly rendered edges can lower rectangular fill
            # without changing the perceived square. Corner-rich low-vertex
            # contours recover those squares while keeping smooth circles out.
            rectilinear_square = min(
                min(1.0, max(0.0, (1.18 - aspect_ratio) / 0.08)),
                min(1.0, max(0.0, (fill_ratio - 0.68) / 0.12)),
                min(1.0, max(0.0, (solidity - 0.82) / 0.10)),
                min(1.0, max(0.0, (70.0 - hull_vertex_count) / 30.0)),
            )
            square_score = max(square_score, rectilinear_square)

            soft_circle = min(
                min(1.0, max(0.0, (1.18 - aspect_ratio) / 0.10)),
                min(1.0, max(0.0, (fill_ratio - 0.55) / 0.12)),
                min(1.0, max(0.0, (0.92 - fill_ratio) / 0.12)),
                min(1.0, max(0.0, (solidity - 0.82) / 0.10)),
            )
            circle_score = max(circle_score, soft_circle)

        triangle_center = min(1.0, max(0.0, (centroid_offset - 0.035) / 0.055))
        triangle_fill = min(1.0, max(0.0, 1.0 - abs(fill_ratio - 0.50) / 0.25))
        triangle_solidity = min(1.0, max(0.0, (solidity - 0.60) / 0.20))
        triangle_score = triangle_fill * triangle_center * triangle_solidity
        return {
            "circle": float(circle_score),
            "square": float(square_score),
            "triangle": float(triangle_score),
        }

    aspect_ratio = max(bbox_w, bbox_h) / max(min(bbox_w, bbox_h), 1)
    aspect_tolerance = min(1.0, max(0.0, (2.10 - aspect_ratio) / 0.20))
    solidity_tolerance = min(1.0, max(0.0, (solidity - 0.90) / 0.05))
    ellipse_fill = min(1.0, max(0.0, 1.0 - abs(fill_ratio - 0.785) / 0.18))
    round_vertex_support = max(
        min(1.0, max(0.0, (hull_vertex_count - 20) / 6.0)),
        1.0 if hull_vertex_count <= 12 and 0.76 <= fill_ratio <= 0.84 else 0.0,
    )
    filled_round_score = ellipse_fill * round_vertex_support * solidity_tolerance
    outlined_round_score = 1.0 if hole_count == 1 and hull_vertex_count >= 20 and aspect_ratio <= 2.10 else 0.0
    circle_score = aspect_tolerance * max(filled_round_score, outlined_round_score)

    centered_score = min(1.0, max(0.0, (0.10 - centroid_offset) / 0.06))
    filled_quadrilateral_support = max(
        min(1.0, max(0.0, (fill_ratio - 0.84) / 0.08)),
        1.0 if hull_vertex_count <= 12 and 0.44 <= fill_ratio <= 0.62 else 0.0,
    )
    filled_square_score = centered_score * solidity_tolerance * filled_quadrilateral_support
    outlined_square_score = (
        1.0
        if hole_count == 1
        and hull_vertex_count <= 12
        and aspect_ratio <= 1.30
        and centroid_offset <= 0.10
        else 0.0
    )
    square_score = aspect_tolerance * max(filled_square_score, outlined_square_score)

    triangle_center_score = min(1.0, max(0.0, (centroid_offset - 0.045) / 0.055))
    triangle_convex_score = min(1.0, max(0.0, (solidity - 0.80) / 0.08))
    triangle_score = triangle_fill_score * triangle_center_score * triangle_convex_score
    return {
        "circle": float(circle_score),
        "square": float(square_score),
        "triangle": float(triangle_score),
    }


def _score_layout(
    layout: Optional[Dict[str, Any]],
    components: Sequence[Component],
    image_hw: Tuple[int, int],
    *,
    benchmark_version: int = 1,
) -> float:
    if not layout:
        return 1.0
    if not components:
        return 0.0
    layout_type = layout.get("type")
    if layout_type == "regions":
        required = list(layout.get("regions", []))
        if benchmark_version >= 4:
            return _region_coverage_score(required, components, image_hw)
        observed = [_region_for_centroid(comp.centroid, image_hw) for comp in components]
        return _set_coverage_score(required, observed)
    if layout_type == "grid_cells":
        rows = int(layout["rows"])
        cols = int(layout["cols"])
        required = [tuple(cell) for cell in layout.get("cells", [])]
        observed = [_grid_cell_for_centroid(comp.centroid, image_hw, rows, cols) for comp in components]
        return _set_coverage_score(required, observed)
    if layout_type in ("horizontal_row", "vertical_column", "diagonal_down"):
        return _pattern_score(layout_type, components)
    return 0.0


def _region_membership(
    region: str,
    centroid: Tuple[float, float],
    image_hw: Tuple[int, int],
) -> bool:
    """Interpret cardinal region words as overlapping perceptual zones."""

    height, width = image_hw
    x = centroid[0] / max(width, 1)
    y = centroid[1] / max(height, 1)
    left, right = x <= 0.45, x >= 0.55
    top, bottom = y <= 0.45, y >= 0.55
    center_x, center_y = 0.30 <= x <= 0.70, 0.30 <= y <= 0.70
    tests = {
        "top_left": top and left,
        "top": top,
        "top_right": top and right,
        "left": left,
        "center": center_x and center_y,
        "right": right,
        "bottom_left": bottom and left,
        "bottom": bottom,
        "bottom_right": bottom and right,
    }
    return bool(tests.get(region, False))


def _region_coverage_score(
    required: Sequence[str],
    components: Sequence[Component],
    image_hw: Tuple[int, int],
) -> float:
    if not required:
        return 1.0
    if not components:
        return 0.0
    matched_regions = sum(
        any(_region_membership(region, component.centroid, image_hw) for component in components)
        for region in required
    )
    matched_components = sum(
        any(_region_membership(region, component.centroid, image_hw) for region in required)
        for component in components
    )
    coverage = matched_regions / len(required)
    purity = matched_components / len(components)
    return float(min(coverage, purity))


def _set_coverage_score(required: Sequence[Any], observed: Sequence[Any]) -> float:
    if not required:
        return 1.0
    required_set = set(required)
    observed_set = set(observed)
    coverage = len(required_set & observed_set) / len(required_set)
    duplicate_penalty = max(0, len(observed) - len(observed_set)) / max(len(observed), 1)
    extra_penalty = max(0, len(observed_set - required_set)) / max(len(observed_set), 1)
    return float(max(0.0, coverage - 0.25 * duplicate_penalty - 0.25 * extra_penalty))


def _region_for_centroid(centroid: Tuple[float, float], image_hw: Tuple[int, int]) -> str:
    h, w = image_hw
    x, y = centroid
    col = 0 if x < w / 3 else 2 if x >= 2 * w / 3 else 1
    row = 0 if y < h / 3 else 2 if y >= 2 * h / 3 else 1
    names = [
        ["top_left", "top", "top_right"],
        ["left", "center", "right"],
        ["bottom_left", "bottom", "bottom_right"],
    ]
    return names[row][col]


def _grid_cell_for_centroid(centroid: Tuple[float, float], image_hw: Tuple[int, int], rows: int, cols: int) -> Tuple[int, int]:
    h, w = image_hw
    x, y = centroid
    row = min(rows - 1, max(0, int(y / max(h, 1) * rows)))
    col = min(cols - 1, max(0, int(x / max(w, 1) * cols)))
    return row, col


def _pattern_score(layout_type: str, components: Sequence[Component]) -> float:
    if len(components) <= 1:
        return 1.0
    xs = np.array([comp.centroid[0] for comp in components], dtype=np.float32)
    ys = np.array([comp.centroid[1] for comp in components], dtype=np.float32)
    span_x = float(max(xs.max() - xs.min(), 1.0))
    span_y = float(max(ys.max() - ys.min(), 1.0))
    if layout_type == "horizontal_row":
        return float(max(0.0, 1.0 - span_y / max(span_x, 1.0)))
    if layout_type == "vertical_column":
        return float(max(0.0, 1.0 - span_x / max(span_y, 1.0)))
    if layout_type == "diagonal_down":
        if span_x < 1.0 or span_y < 1.0:
            return 0.0
        corr = np.corrcoef(xs, ys)[0, 1]
        return float(max(0.0, corr))
    return 0.0


def _score_relation(
    relation: Dict[str, Any],
    id_to_components: Dict[str, List[Component]],
    *,
    observed_id_to_components: Optional[Dict[str, List[Component]]] = None,
    object_shapes: Optional[Dict[str, str]] = None,
    image_hw: Optional[Tuple[int, int]] = None,
    benchmark_version: int = 1,
) -> Tuple[float, Dict[str, Any]]:
    observed_id_to_components = observed_id_to_components or id_to_components
    subject = id_to_components.get(relation["subject"], [])
    obj = id_to_components.get(relation.get("object"), [])
    result = {
        "type": relation["type"],
        "subject": relation["subject"],
        "object": relation.get("object"),
        "score": 0.0,
        "strict_pass": False,
    }
    rel_type = relation["type"]
    if rel_type in (
        "same_count",
        "more_than_count",
        "fewer_than_count",
        "twice_as_many",
        "times_as_many",
    ):
        return _score_count_comparison(
            relation,
            observed_id_to_components,
            result,
            benchmark_version=benchmark_version,
            object_shapes=object_shapes,
        )
    if rel_type in ("all_left_of", "all_right_of", "all_above", "all_below"):
        return _score_all_to_all_direction(relation, id_to_components, result)
    if rel_type in (
        "all_same_row",
        "all_same_column",
        "not_all_same_row",
        "not_all_same_column",
    ):
        return _score_group_alignment(
            relation,
            id_to_components,
            result,
            image_hw=image_hw,
        )
    if rel_type == "not_all_same_size" or (
        rel_type == "all_same_size" and relation.get("object") is None
    ):
        return _score_within_group_size_relation(relation, id_to_components, result)
    if (
        benchmark_version >= 4
        and rel_type
        in {
            "larger_than",
            "smaller_than",
            "same_size",
            "all_larger_than",
            "all_smaller_than",
            "all_same_size",
            "left_of",
            "right_of",
            "above",
            "below",
            "closer_than",
            "farther_than",
            "between",
            "same_row",
            "same_column",
            "touching",
            "separate",
            "not_touching",
        }
        and _relation_has_aliased_referents(relation, id_to_components)
    ):
        result.update({"aliased_referents": True})
        return 0.0, result
    if rel_type in (
        "larger_than",
        "smaller_than",
        "same_size",
        "all_larger_than",
        "all_smaller_than",
        "all_same_size",
        "largest",
        "smallest",
    ):
        return _score_size_relation(relation, id_to_components, result, benchmark_version=benchmark_version)
    if rel_type in ("closer_than", "farther_than"):
        return _score_comparative_distance(relation, id_to_components, result, benchmark_version=benchmark_version)
    benchmark_boundaries = any(component.boundary is not None for component in subject + obj)
    if rel_type in ("touching", "separate") or (rel_type == "not_touching" and benchmark_boundaries):
        return _score_boundary_relation(
            relation,
            id_to_components,
            result,
            benchmark_version=benchmark_version,
        )
    if rel_type in ("same_row", "same_column"):
        return _score_alignment_relation(
            relation,
            id_to_components,
            result,
            image_hw=image_hw,
            benchmark_version=benchmark_version,
        )
    if rel_type in ("leftmost", "rightmost", "topmost", "bottommost"):
        return _score_extreme_position(
            relation,
            id_to_components,
            result,
            image_hw=image_hw,
            benchmark_version=benchmark_version,
        )
    if rel_type == "between":
        return _score_between_relation(
            relation,
            id_to_components,
            result,
            benchmark_version=benchmark_version,
        )
    if not subject or not obj:
        return 0.0, result
    if benchmark_version >= 4 and rel_type in ("left_of", "right_of", "above", "below"):
        nested = any(
            _bbox_intersection_fraction(first, second) >= 0.98
            and math.dist(first.centroid, second.centroid)
            <= 0.80 * min(_component_extent(first), _component_extent(second))
            and max(first.shape_scores.values(), default=0.0) >= 0.40
            and max(second.shape_scores.values(), default=0.0) >= 0.40
            for first in subject
            for second in obj
        )
        if nested:
            result.update({"score": 0.0, "strict_pass": False, "nested_referents": True})
            return 0.0, result
    subj_x = float(np.mean([comp.centroid[0] for comp in subject]))
    subj_y = float(np.mean([comp.centroid[1] for comp in subject]))
    obj_x = float(np.mean([comp.centroid[0] for comp in obj]))
    obj_y = float(np.mean([comp.centroid[1] for comp in obj]))
    margin = float(relation.get("margin_px", 24.0))
    if rel_type == "left_of":
        delta = obj_x - subj_x
    elif rel_type == "right_of":
        delta = subj_x - obj_x
    elif rel_type == "above":
        delta = obj_y - subj_y
    elif rel_type == "below":
        delta = subj_y - obj_y
    elif rel_type == "inside":
        score = _inside_score(_union_bbox(subject), _union_bbox(obj), margin=relation.get("margin_px", 0.0))
        result.update({"score": float(score), "strict_pass": bool(score >= 0.99)})
        return float(score), result
    elif rel_type == "contains":
        score = _inside_score(_union_bbox(obj), _union_bbox(subject), margin=relation.get("margin_px", 0.0))
        result.update({"score": float(score), "strict_pass": bool(score >= 0.99)})
        return float(score), result
    elif rel_type in ("each_inside", "each_contains"):
        return _score_distinct_containment(relation, subject, obj, result)
    elif rel_type in ("near", "far", "not_touching"):
        score, strict, measurement = _score_distance_relation(rel_type, subject, obj, relation)
        result.update({"score": float(score), "strict_pass": bool(strict), **measurement})
        return float(score), result
    else:
        delta = -margin
    score = min(1.0, max(0.0, delta / max(margin, 1.0)))
    result.update({"score": float(score), "strict_pass": bool(delta >= margin), "margin": delta})
    return float(score), result


def _relation_has_aliased_referents(
    relation: Dict[str, Any],
    id_to_components: Dict[str, List[Component]],
) -> bool:
    relation_ids = _relation_object_ids(relation)
    for index, first_id in enumerate(relation_ids):
        for second_id in relation_ids[index + 1 :]:
            if _components_alias_same_referent(
                id_to_components.get(first_id, []),
                id_to_components.get(second_id, []),
            ):
                return True
    return False


def _extremum_subject_aliases_reference(
    relation: Dict[str, Any],
    id_to_components: Dict[str, List[Component]],
) -> bool:
    """Reject an extremum whose subject is the same pixels as a named peer."""

    subject_id = str(relation["subject"])
    subject = id_to_components.get(subject_id, [])
    return any(
        reference_id != subject_id
        and _components_alias_same_referent(
            subject,
            id_to_components.get(reference_id, []),
        )
        for reference_id in _relation_references(relation, id_to_components)
    )


def _relation_has_missing_shape_identity(
    relation: Dict[str, Any],
    id_to_components: Dict[str, List[Component]],
    object_shapes: Dict[str, str],
) -> bool:
    """Reject named referents with no pixel evidence for their requested shape."""

    for object_id in _relation_object_ids(relation):
        components = id_to_components.get(object_id, [])
        requested_shape = object_shapes.get(object_id)
        if not _components_have_shape_identity(components, requested_shape):
            return True
    return False


def _relation_all_referents_lack_shape_identity(
    relation: Dict[str, Any],
    id_to_components: Dict[str, List[Component]],
    object_shapes: Dict[str, str],
) -> bool:
    """Detect directional claims formed only from shape-less color fragments."""

    relation_ids = _relation_object_ids(relation)
    return bool(relation_ids) and all(
        not _components_have_shape_identity(
            id_to_components.get(object_id, []),
            object_shapes.get(object_id),
        )
        for object_id in relation_ids
    )


def _relation_has_strong_shape_conflict(
    relation: Dict[str, Any],
    id_to_color_components: Dict[str, List[Component]],
    object_shapes: Dict[str, str],
    foreground_components: Sequence[Component],
    selected_components: Optional[Dict[str, List[Component]]] = None,
) -> bool:
    """Reject a named referent whose color mask clearly depicts another shape."""

    for object_id in _relation_object_ids(relation):
        requested_shape = object_shapes.get(object_id)
        if requested_shape is None:
            continue
        selected = (selected_components or {}).get(object_id, [])
        candidates = id_to_color_components.get(object_id, [])
        if selected:
            selected_global_candidates = [
                component
                for component in candidates
                if any(
                    _components_alias_same_referent([component], [chosen])
                    for chosen in selected
                )
            ]
            if selected_global_candidates:
                candidates = selected_global_candidates
        for component in candidates:
            requested = component.shape_scores.get(requested_shape, 0.0)
            competing = max(
                (
                    score
                    for shape, score in component.shape_scores.items()
                    if shape != requested_shape
                ),
                default=0.0,
            )
            if requested < 0.03 and competing >= 0.65:
                return True
            for foreground in foreground_components:
                if (
                    _bbox_intersection_fraction(component, foreground) < 0.70
                    or _bbox_larger_intersection_fraction(component, foreground) < 0.70
                ):
                    continue
                foreground_requested = foreground.shape_scores.get(requested_shape, 0.0)
                foreground_competing = max(
                    (
                        score
                        for shape, score in foreground.shape_scores.items()
                        if shape != requested_shape
                    ),
                    default=0.0,
                )
                if foreground_requested < 0.03 and foreground_competing >= 0.65:
                    return True
    return False


def _components_have_shape_identity(
    components: Sequence[Component],
    requested_shape: Optional[str],
) -> bool:
    if not components or requested_shape is None:
        return False
    requested_score = max(
        (component.shape_scores.get(requested_shape, 0.0) for component in components),
        default=0.0,
    )
    if requested_score >= 0.03:
        return True
    for component in components:
        aspect_ratio = max(component.aspect_ratio, 1.0 / max(component.aspect_ratio, 1e-6))
        if requested_shape == "square":
            plausible_geometry = bool(
                aspect_ratio <= 1.18
                or (
                    aspect_ratio <= 1.35
                    and component.fill_ratio >= 0.65
                    and component.solidity >= 0.65
                )
            )
        elif requested_shape == "circle":
            plausible_geometry = bool(
                aspect_ratio <= 1.18
                and 0.42 <= component.fill_ratio <= 0.92
                and component.solidity >= 0.55
            )
        else:
            # Zero-score triangles reaching this point were already accepted
            # as plausible crops by _component_identity_compatible.
            plausible_geometry = True
        if plausible_geometry:
            return True
    return False


def _relation_referent_aliases_stronger_object(
    relation: Dict[str, Any],
    relation_components: Dict[str, List[Component]],
    all_components: Dict[str, List[Component]],
    object_shapes: Dict[str, str],
) -> bool:
    """Reject a relation referent explained better by another named object."""

    relation_ids = set(_relation_object_ids(relation))
    for relation_id in relation_ids:
        for component in relation_components.get(relation_id, []):
            for other_id, candidates in all_components.items():
                if other_id == relation_id:
                    continue
                if object_shapes.get(other_id) != object_shapes.get(relation_id):
                    continue
                for other in candidates:
                    same_physical_shape = bool(
                        _components_alias_same_referent([component], [other])
                        or (
                            max(
                                _bbox_containment_fraction(component, other),
                                _bbox_containment_fraction(other, component),
                            )
                            >= 0.80
                            and math.dist(component.centroid, other.centroid)
                            <= 0.35 * max(_component_extent(component), _component_extent(other))
                        )
                    )
                    if (
                        same_physical_shape
                        and other.color_confidence >= component.color_confidence + 0.08
                    ):
                        return True
    return False


def _ground_extreme_against_visible_objects(
    relation: Dict[str, Any],
    selected: Dict[str, List[Component]],
    components_by_color: Dict[str, List[Component]],
    background_color: Optional[str],
) -> Tuple[Dict[str, Any], Dict[str, List[Component]]]:
    """Ground a colored-object extremum against every visible color component."""

    subject = selected.get(str(relation["subject"]), [])
    size_extremum = relation.get("type") in {"largest", "smallest"}
    min_reference_area = (
        0.0
        if size_extremum
        else 0.02 * max((component.area for component in subject), default=0)
    )
    grounded = dict(selected)
    reference_ids: List[str] = []
    index = 0
    background_chromatic_color = {
        "pale cyan": "cyan",
        "pale pink": "pink",
    }.get(background_color)
    for color, components in components_by_color.items():
        if color == background_chromatic_color:
            continue
        for component in _countable_components(
            components,
            min_dimension=6,
            min_color_confidence=0.25,
            min_relative_area=0.005,
        ):
            if component.area < min_reference_area:
                continue
            aliases_subject = any(
                _components_alias_same_referent([subject_component], [component])
                for subject_component in subject
            )
            if aliases_subject:
                continue
            reference_id = f"__visible_{index:03d}"
            grounded[reference_id] = [component]
            reference_ids.append(reference_id)
            index += 1
    relation_copy = dict(relation)
    relation_copy["references"] = reference_ids
    return relation_copy, grounded


def _score_count_comparison(
    relation: Dict[str, Any],
    observed: Dict[str, List[Component]],
    result: Dict[str, Any],
    *,
    benchmark_version: int = 1,
    object_shapes: Optional[Dict[str, str]] = None,
) -> Tuple[float, Dict[str, Any]]:
    object_shapes = object_shapes or {}
    subject_id = relation["subject"]
    object_id = relation.get("object")
    if benchmark_version >= 4:
        subject_count = _effective_component_count(
            observed.get(subject_id, []),
            object_shapes.get(subject_id),
        )
        object_count = _effective_component_count(
            observed.get(object_id, []),
            object_shapes.get(object_id),
        )
    else:
        subject_count = sum(component.multiplicity for component in observed.get(subject_id, []))
        object_count = sum(component.multiplicity for component in observed.get(object_id, []))
    rel_type = relation["type"]
    if benchmark_version >= 4 and (subject_count == 0 or object_count == 0):
        score = 0.0
        strict = False
    elif rel_type == "same_count":
        error = abs(subject_count - object_count)
        score = max(0.0, 1.0 - error / max(subject_count, object_count, 1))
        strict = error == 0
    elif rel_type == "more_than_count":
        delta = subject_count - object_count
        score = min(1.0, max(0.0, 0.5 + delta / (2.0 * max(subject_count, object_count, 1))))
        strict = delta >= int(relation.get("min_difference", 1))
    elif rel_type == "fewer_than_count":
        delta = object_count - subject_count
        score = min(1.0, max(0.0, 0.5 + delta / (2.0 * max(subject_count, object_count, 1))))
        strict = delta >= int(relation.get("min_difference", 1))
    else:
        multiplier = float(relation.get("multiplier", 2.0))
        target = multiplier * object_count
        error = abs(subject_count - target)
        score = max(0.0, 1.0 - error / max(target, 1.0))
        strict = error <= 1e-6
    result.update(
        {
            "score": float(score),
            "strict_pass": bool(strict),
            "subject_count": float(subject_count),
            "object_count": float(object_count),
        }
    )
    return float(score), result


def _mean_component_extent(components: Sequence[Component]) -> float:
    if not components:
        return 0.0
    return float(np.mean([_component_extent(component) for component in components]))


def _relation_references(
    relation: Dict[str, Any],
    id_to_components: Dict[str, List[Component]],
) -> List[str]:
    references = relation.get("references")
    if isinstance(references, list):
        return [str(reference) for reference in references]
    return [object_id for object_id in id_to_components if object_id != relation.get("subject")]


def _score_within_group_size_relation(
    relation: Dict[str, Any],
    id_to_components: Dict[str, List[Component]],
    result: Dict[str, Any],
) -> Tuple[float, Dict[str, Any]]:
    """Score whether all visible instances in one group share the same size."""

    subject = id_to_components.get(relation["subject"], [])
    if len(subject) < 2:
        return 0.0, result
    extents = [_component_extent(component) for component in subject]
    ratio = max(extents) / max(min(extents), 1.0)
    max_ratio = float(relation.get("max_size_ratio", 1.08))
    same_score = max(
        0.0,
        1.0 - (ratio - 1.0) / max(max_ratio - 1.0, 1e-6),
    )
    if relation["type"] == "not_all_same_size":
        score = 1.0 - same_score
        strict = ratio > max_ratio
    else:
        score = same_score
        strict = ratio <= max_ratio
    result.update(
        {
            "score": float(score),
            "strict_pass": bool(strict),
            "minimum_extent": float(min(extents)),
            "maximum_extent": float(max(extents)),
            "size_ratio": float(ratio),
            "comparison": "within_group_extent_range",
        }
    )
    return float(score), result


def _score_size_relation(
    relation: Dict[str, Any],
    id_to_components: Dict[str, List[Component]],
    result: Dict[str, Any],
    *,
    benchmark_version: int = 1,
) -> Tuple[float, Dict[str, Any]]:
    subject = id_to_components.get(relation["subject"], [])
    rel_type = relation["type"]
    if not subject:
        return 0.0, result
    if rel_type in ("all_larger_than", "all_smaller_than", "all_same_size"):
        obj = id_to_components.get(relation.get("object"), [])
        if not obj:
            return 0.0, result
        subject_extents = [_component_extent(component) for component in subject]
        object_extents = [_component_extent(component) for component in obj]
        subject_min = min(subject_extents)
        subject_max = max(subject_extents)
        object_min = min(object_extents)
        object_max = max(object_extents)
        if rel_type == "all_larger_than":
            min_ratio = float(relation.get("min_size_ratio", 1.05))
            ratio = subject_min / max(object_max, 1.0)
            score = min(1.0, max(0.0, (ratio - 1.0) / max(min_ratio - 1.0, 1e-6)))
            strict = ratio >= min_ratio
        elif rel_type == "all_smaller_than":
            min_ratio = float(relation.get("min_size_ratio", 1.05))
            ratio = object_min / max(subject_max, 1.0)
            score = min(1.0, max(0.0, (ratio - 1.0) / max(min_ratio - 1.0, 1e-6)))
            strict = ratio >= min_ratio
        else:
            max_ratio = float(relation.get("max_size_ratio", 1.08))
            all_extents = subject_extents + object_extents
            ratio = max(all_extents) / max(min(all_extents), 1.0)
            score = max(
                0.0,
                1.0 - (ratio - 1.0) / max(max_ratio - 1.0, 1e-6),
            )
            strict = ratio <= max_ratio
        result.update(
            {
                "score": float(score),
                "strict_pass": bool(strict),
                "subject_min_extent": float(subject_min),
                "subject_max_extent": float(subject_max),
                "object_min_extent": float(object_min),
                "object_max_extent": float(object_max),
                "size_ratio": float(ratio),
                "comparison": "worst_case_all_to_all",
            }
        )
        return float(score), result
    subject_extent = _mean_component_extent(subject)
    default_min_ratio = 1.05 if benchmark_version >= 3 else (1.08 if benchmark_version >= 2 else 1.35)
    min_ratio = float(relation.get("min_size_ratio", default_min_ratio))
    if rel_type in ("largest", "smallest"):
        reference_ids = _relation_references(relation, id_to_components)
        reference_extents = [
            _mean_component_extent(id_to_components.get(reference, []))
            for reference in reference_ids
            if id_to_components.get(reference)
        ]
        result["references"] = reference_ids
        if not reference_extents:
            return 0.0, result
        if rel_type == "largest":
            comparison_extent = max(reference_extents)
            ratio = subject_extent / max(comparison_extent, 1.0)
        else:
            comparison_extent = min(reference_extents)
            ratio = comparison_extent / max(subject_extent, 1.0)
        score = min(1.0, max(0.0, (ratio - 1.0) / max(min_ratio - 1.0, 1e-6)))
        strict = ratio >= min_ratio
    else:
        obj = id_to_components.get(relation.get("object"), [])
        if not obj:
            return 0.0, result
        comparison_extent = _mean_component_extent(obj)
        if rel_type == "larger_than":
            ratio = subject_extent / max(comparison_extent, 1.0)
            score = min(1.0, max(0.0, (ratio - 1.0) / max(min_ratio - 1.0, 1e-6)))
            strict = ratio >= min_ratio
        elif rel_type == "smaller_than":
            ratio = comparison_extent / max(subject_extent, 1.0)
            score = min(1.0, max(0.0, (ratio - 1.0) / max(min_ratio - 1.0, 1e-6)))
            strict = ratio >= min_ratio
        else:
            ratio = max(subject_extent, comparison_extent) / max(min(subject_extent, comparison_extent), 1.0)
            default_max_ratio = 1.08 if benchmark_version >= 4 else (1.25 if benchmark_version >= 2 else 1.15)
            max_ratio = float(relation.get("max_size_ratio", default_max_ratio))
            score = max(0.0, 1.0 - (ratio - 1.0) / max(max_ratio - 1.0, 1e-6))
            strict = ratio <= max_ratio
    result.update(
        {
            "score": float(score),
            "strict_pass": bool(strict),
            "subject_extent": float(subject_extent),
            "comparison_extent": float(comparison_extent),
            "size_ratio": float(ratio),
        }
    )
    return float(score), result


def _component_boundary_distance(a: Component, b: Component) -> float:
    if a.boundary is None or b.boundary is None or not len(a.boundary) or not len(b.boundary):
        return _bbox_gap(a.bbox, b.bbox)
    first, second = (a.boundary, b.boundary) if len(a.boundary) <= len(b.boundary) else (b.boundary, a.boundary)
    best_squared = float("inf")
    for start in range(0, len(first), 256):
        chunk = first[start : start + 256]
        delta = chunk[:, None, :] - second[None, :, :]
        best_squared = min(best_squared, float(np.min(np.sum(delta * delta, axis=-1))))
        if best_squared <= 1.0:
            break
    return float(math.sqrt(best_squared))


def _boundary_distance(a: Sequence[Component], b: Sequence[Component]) -> float:
    if not a or not b:
        return float("inf")
    return min(_component_boundary_distance(first, second) for first in a for second in b)


def _score_comparative_distance(
    relation: Dict[str, Any],
    id_to_components: Dict[str, List[Component]],
    result: Dict[str, Any],
    *,
    benchmark_version: int = 1,
) -> Tuple[float, Dict[str, Any]]:
    subject = id_to_components.get(relation["subject"], [])
    anchor_a = id_to_components.get(relation.get("anchor_a"), [])
    anchor_b = id_to_components.get(relation.get("anchor_b"), [])
    result.update({"anchor_a": relation.get("anchor_a"), "anchor_b": relation.get("anchor_b")})
    if not subject or not anchor_a or not anchor_b:
        return 0.0, result
    distance_a = _boundary_distance(subject, anchor_a)
    distance_b = _boundary_distance(subject, anchor_b)
    if relation["type"] == "farther_than":
        nearer_distance, farther_distance = distance_b, distance_a
    else:
        nearer_distance, farther_distance = distance_a, distance_b
    default_max_ratio = 0.90 if benchmark_version >= 2 else 0.70
    default_min_margin = 4.0 if benchmark_version >= 2 else 8.0
    max_ratio = float(relation.get("max_distance_ratio", default_max_ratio))
    min_margin = float(relation.get("min_margin_px", default_min_margin))
    ratio = nearer_distance / max(farther_distance, 1e-6)
    advantage = farther_distance - nearer_distance
    score = min(1.0, max(0.0, (1.0 - ratio) / max(1.0 - max_ratio, 1e-6)))
    strict = ratio <= max_ratio and advantage >= min_margin
    result.update(
        {
            "score": float(score),
            "strict_pass": bool(strict),
            "distance_to_anchor_a": float(distance_a),
            "distance_to_anchor_b": float(distance_b),
            "near_far_ratio": float(ratio),
            "distance_margin": float(advantage),
        }
    )
    return float(score), result


def _ground_comparative_distance_relation(
    relation: Dict[str, Any],
    selected: Dict[str, List[Component]],
    candidates: Dict[str, List[Component]],
    object_shapes: Dict[str, str],
) -> Dict[str, List[Component]]:
    """Resolve near-tied object candidates using the full relation structure."""

    object_ids = [relation.get("subject"), relation.get("anchor_a"), relation.get("anchor_b")]
    if any(object_id is None for object_id in object_ids):
        return selected
    candidate_sets: List[List[Component]] = []
    for object_id in object_ids:
        object_id = str(object_id)
        pool = candidates.get(object_id, [])
        shape = object_shapes.get(object_id, "circle")
        if not pool:
            return selected
        best_quality = _component_match_quality(pool[0], shape)
        plausible = [
            component
            for component in pool
            if _component_match_quality(component, shape) >= max(0.40, best_quality - 0.05)
        ]
        candidate_sets.append(plausible or [pool[0]])

    best_objective = float("-inf")
    best_assignment: Optional[Tuple[Component, ...]] = None
    for assignment in product(*candidate_sets):
        trial = dict(selected)
        qualities = []
        for object_id, component in zip(object_ids, assignment):
            object_id = str(object_id)
            trial[object_id] = [component]
            qualities.append(_component_match_quality(component, object_shapes.get(object_id, "circle")))
        relation_score, _ = _score_comparative_distance(
            relation,
            trial,
            {"type": relation["type"], "score": 0.0, "strict_pass": False},
            benchmark_version=2,
        )
        objective = float(np.mean(qualities)) + 0.10 * relation_score
        if objective > best_objective:
            best_objective = objective
            best_assignment = assignment

    if best_assignment is None:
        return selected
    grounded = dict(selected)
    for object_id, component in zip(object_ids, best_assignment):
        grounded[str(object_id)] = [component]
    return grounded


def _score_all_to_all_direction(
    relation: Dict[str, Any],
    id_to_components: Dict[str, List[Component]],
    result: Dict[str, Any],
) -> Tuple[float, Dict[str, Any]]:
    """Score whether every subject centroid precedes every object centroid."""

    subject = id_to_components.get(relation["subject"], [])
    obj = id_to_components.get(relation.get("object"), [])
    if not subject or not obj:
        return 0.0, result

    relation_type = str(relation["type"])
    if relation_type == "all_left_of":
        gap = min(component.centroid[0] for component in obj) - max(
            component.centroid[0] for component in subject
        )
    elif relation_type == "all_right_of":
        gap = min(component.centroid[0] for component in subject) - max(
            component.centroid[0] for component in obj
        )
    elif relation_type == "all_above":
        gap = min(component.centroid[1] for component in obj) - max(
            component.centroid[1] for component in subject
        )
    else:
        gap = min(component.centroid[1] for component in subject) - max(
            component.centroid[1] for component in obj
        )

    margin = float(relation.get("margin_px", 12.0))
    score = min(1.0, max(0.0, float(gap) / max(margin, 1.0)))
    result.update(
        {
            "score": score,
            "strict_pass": bool(gap >= margin),
            "minimum_centroid_gap": float(gap),
            "required_margin": margin,
            "quantifier": "all",
        }
    )
    return score, result


def _score_boundary_relation(
    relation: Dict[str, Any],
    id_to_components: Dict[str, List[Component]],
    result: Dict[str, Any],
    *,
    benchmark_version: int = 1,
) -> Tuple[float, Dict[str, Any]]:
    subject = id_to_components.get(relation["subject"], [])
    obj = id_to_components.get(relation.get("object"), [])
    if not subject or not obj:
        return 0.0, result
    if benchmark_version >= 4:
        subject_center = _mean_visual_center(subject)
        object_center = _mean_visual_center(obj)
        center_distance = math.dist(subject_center, object_center)
        subject_radius = math.sqrt(sum(component.area for component in subject) / math.pi)
        object_radius = math.sqrt(sum(component.area for component in obj) / math.pi)
        equivalent_distance = center_distance - subject_radius - object_radius
        pixel_boundary_distance = min(
            _component_boundary_distance(first, second)
            for first in subject
            for second in obj
        )
        # Exclusive color masks are one pixel apart at exact raster contact.
        # Equivalent-area geometry resolves that quantization regime; a
        # larger measured boundary is direct evidence of visible separation.
        strongly_overlapping_support = equivalent_distance <= -0.05 * max(
            subject_radius + object_radius,
            1.0,
        )
        decision_distance = float(relation.get("decision_distance_px", 0.27))
        raster_tangent = pixel_boundary_distance <= 2.0 and equivalent_distance <= 1.0
        distance = (
            min(equivalent_distance, decision_distance)
            if raster_tangent
            else equivalent_distance
            if pixel_boundary_distance <= 1.0
            or (pixel_boundary_distance <= 3.0 and strongly_overlapping_support)
            else pixel_boundary_distance
        )
        transition = float(relation.get("transition_px", 3.0))
        result.update(
            {
                "center_distance": float(center_distance),
                "subject_equivalent_radius": float(subject_radius),
                "object_equivalent_radius": float(object_radius),
                "equivalent_boundary_distance": float(equivalent_distance),
                "pixel_boundary_distance": float(pixel_boundary_distance),
                "raster_tangent": bool(raster_tangent),
            }
        )
    elif benchmark_version >= 2:
        subject_center = _mean_visual_center(subject)
        object_center = _mean_visual_center(obj)
        center_distance = math.dist(subject_center, object_center)
        subject_radius = math.sqrt(sum(component.area for component in subject) / math.pi)
        object_radius = math.sqrt(sum(component.area for component in obj) / math.pi)
        distance = center_distance - subject_radius - object_radius
        decision_distance = float(relation.get("decision_distance_px", 1.0))
        transition = float(relation.get("transition_px", 3.0))
        result.update(
            {
                "center_distance": float(center_distance),
                "subject_equivalent_radius": float(subject_radius),
                "object_equivalent_radius": float(object_radius),
            }
        )
    else:
        distance = _boundary_distance(subject, obj)
        decision_distance = float(relation.get("decision_distance_px", 2.5))
        transition = float(relation.get("transition_px", 1.5))
    contact_score = min(
        1.0,
        max(0.0, 0.5 + (decision_distance - distance) / max(2.0 * transition, 1e-6)),
    )
    if relation["type"] == "touching":
        score = contact_score
        strict = distance <= decision_distance
    else:
        score = 1.0 - contact_score
        strict = distance > decision_distance
    result.update(
        {
            "score": float(score),
            "strict_pass": bool(strict),
            "boundary_distance": float(distance),
            "contact_score": float(contact_score),
            "decision_distance": decision_distance,
        }
    )
    return float(score), result


def _score_alignment_relation(
    relation: Dict[str, Any],
    id_to_components: Dict[str, List[Component]],
    result: Dict[str, Any],
    *,
    image_hw: Optional[Tuple[int, int]],
    benchmark_version: int = 1,
) -> Tuple[float, Dict[str, Any]]:
    subject = id_to_components.get(relation["subject"], [])
    obj = id_to_components.get(relation.get("object"), [])
    if not subject or not obj:
        return 0.0, result
    subject_center = _mean_visual_center(subject)
    object_center = _mean_visual_center(obj)
    axis = 1 if relation["type"] == "same_row" else 0
    offset = abs(subject_center[axis] - object_center[axis])
    tolerance_fraction = 0.04
    default_tolerance = 12.0 if image_hw is None else tolerance_fraction * min(image_hw)
    if benchmark_version >= 2:
        default_tolerance = max(
            default_tolerance,
            0.55
            * max(
                max((_component_extent(component) for component in subject), default=0.0),
                max((_component_extent(component) for component in obj), default=0.0),
            ),
        )
    tolerance = float(relation.get("tolerance_px", default_tolerance))
    score = max(0.0, 1.0 - max(0.0, offset - tolerance) / max(2.0 * tolerance, 1.0))
    strict = offset <= tolerance + 0.5
    if strict:
        score = 1.0
    result.update({"score": float(score), "strict_pass": bool(strict), "axis_offset": float(offset), "tolerance": tolerance})
    return float(score), result


def _score_group_alignment(
    relation: Dict[str, Any],
    id_to_components: Dict[str, List[Component]],
    result: Dict[str, Any],
    *,
    image_hw: Optional[Tuple[int, int]],
) -> Tuple[float, Dict[str, Any]]:
    """Score an all-aligned group predicate or its exact logical negation."""

    subject = id_to_components.get(relation["subject"], [])
    if len(subject) < 2:
        result.update({"detected_instances": len(subject), "strict_pass": False})
        return 0.0, result

    relation_type = str(relation["type"])
    axis = 1 if relation_type.endswith("same_row") else 0
    coordinates = [float(component.centroid[axis]) for component in subject]
    span = max(coordinates) - min(coordinates)
    tolerance_fraction = 0.04
    default_tolerance = 12.0 if image_hw is None else tolerance_fraction * min(image_hw)
    tolerance = float(
        relation.get(
            "tolerance_px",
            max(
                default_tolerance,
                0.55 * max((_component_extent(component) for component in subject), default=0.0),
            ),
        )
    )
    alignment_strict = span <= tolerance + 0.5
    alignment_score = max(
        0.0,
        1.0 - max(0.0, span - tolerance) / max(2.0 * tolerance, 1.0),
    )
    if alignment_strict:
        alignment_score = 1.0
    negated = relation_type.startswith("not_all_")
    strict = not alignment_strict if negated else alignment_strict
    score = 1.0 - alignment_score if negated else alignment_score
    result.update(
        {
            "score": float(score),
            "strict_pass": bool(strict),
            "axis_span": float(span),
            "tolerance": tolerance,
            "detected_instances": len(subject),
            "all_aligned_strict": bool(alignment_strict),
            "quantifier": "not_all" if negated else "all",
        }
    )
    return float(score), result


def _score_extreme_position(
    relation: Dict[str, Any],
    id_to_components: Dict[str, List[Component]],
    result: Dict[str, Any],
    *,
    image_hw: Optional[Tuple[int, int]],
    benchmark_version: int = 1,
) -> Tuple[float, Dict[str, Any]]:
    subject = id_to_components.get(relation["subject"], [])
    reference_ids = _relation_references(relation, id_to_components)
    references = [id_to_components[reference] for reference in reference_ids if id_to_components.get(reference)]
    result["references"] = reference_ids
    if not subject or not references:
        return 0.0, result
    subject_center = _mean_visual_center(subject)
    reference_centers = [_mean_visual_center(reference) for reference in references]
    rel_type = relation["type"]
    if benchmark_version >= 2:
        subject_box = _union_bbox(subject)
        reference_boxes = [_union_bbox(reference) for reference in references]
        if rel_type == "leftmost":
            margin = min(box[0] for box in reference_boxes) - subject_box[0]
        elif rel_type == "rightmost":
            margin = subject_box[2] - max(box[2] for box in reference_boxes)
        elif rel_type == "topmost":
            margin = min(box[1] for box in reference_boxes) - subject_box[1]
        else:
            margin = subject_box[3] - max(box[3] for box in reference_boxes)
    elif rel_type == "leftmost":
        margin = min(center[0] for center in reference_centers) - subject_center[0]
    elif rel_type == "rightmost":
        margin = subject_center[0] - max(center[0] for center in reference_centers)
    elif rel_type == "topmost":
        margin = min(center[1] for center in reference_centers) - subject_center[1]
    else:
        margin = subject_center[1] - max(center[1] for center in reference_centers)
    if benchmark_version >= 4:
        required_margin = max(3.0, 0.04 * _mean_component_extent(subject))
    elif benchmark_version >= 2:
        default_margin = 3.0
    else:
        default_margin = 12.0 if image_hw is None else 0.04 * min(image_hw)
    if benchmark_version < 4:
        required_margin = default_margin
    required_margin = float(relation.get("margin_px", required_margin))
    score = min(1.0, max(0.0, margin / max(required_margin, 1.0)))
    strict = margin >= required_margin
    result.update({"score": float(score), "strict_pass": bool(strict), "extreme_margin": float(margin)})
    return float(score), result


def _score_between_relation(
    relation: Dict[str, Any],
    id_to_components: Dict[str, List[Component]],
    result: Dict[str, Any],
    *,
    benchmark_version: int = 1,
) -> Tuple[float, Dict[str, Any]]:
    subject = id_to_components.get(relation["subject"], [])
    anchor_a = id_to_components.get(relation.get("anchor_a"), [])
    anchor_b = id_to_components.get(relation.get("anchor_b"), [])
    result.update({"anchor_a": relation.get("anchor_a"), "anchor_b": relation.get("anchor_b")})
    if not subject or not anchor_a or not anchor_b:
        return 0.0, result
    s = np.array(_mean_centroid(subject), dtype=np.float32)
    a = np.array(_mean_centroid(anchor_a), dtype=np.float32)
    b = np.array(_mean_centroid(anchor_b), dtype=np.float32)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-6:
        return 0.0, result
    t = float(np.dot(s - a, ab) / denom)
    projection = a + t * ab
    distance_to_line = float(np.linalg.norm(s - projection))
    anchor_distance = float(np.linalg.norm(ab))
    line_score = max(0.0, 1.0 - distance_to_line / max(anchor_distance * 0.25, 1.0))
    interval_score = 1.0 if 0.15 <= t <= 0.85 else max(0.0, 1.0 - min(abs(t - 0.15), abs(t - 0.85)) / 0.35)
    score = float(line_score * interval_score)
    default_threshold = 0.70 if benchmark_version >= 4 else 0.80
    strict_threshold = float(relation.get("between_score_threshold", default_threshold))
    result.update(
        {
            "score": score,
            "strict_pass": bool(score >= strict_threshold),
            "projection_t": t,
            "distance_to_line": distance_to_line,
        }
    )
    return score, result


def _score_distance_relation(
    rel_type: str,
    subject: Sequence[Component],
    obj: Sequence[Component],
    relation: Dict[str, Any],
) -> Tuple[float, bool, Dict[str, Any]]:
    sx, sy = _mean_centroid(subject)
    ox, oy = _mean_centroid(obj)
    center_dist = float(math.hypot(sx - ox, sy - oy))
    subj_box = _union_bbox(subject)
    obj_box = _union_bbox(obj)
    edge_gap = _bbox_gap(subj_box, obj_box)
    near_px = float(relation.get("near_px", 96.0))
    far_px = float(relation.get("far_px", 160.0))
    if rel_type == "near":
        score = max(0.0, 1.0 - center_dist / max(near_px, 1.0))
        strict = center_dist <= near_px
    elif rel_type == "far":
        score = min(1.0, center_dist / max(far_px, 1.0))
        strict = center_dist >= far_px
    else:
        min_gap = float(relation.get("min_gap_px", 8.0))
        score = min(1.0, max(0.0, edge_gap / max(min_gap, 1.0)))
        strict = edge_gap >= min_gap
    return score, strict, {"center_distance": center_dist, "edge_gap": edge_gap}


def _best_distinct_assignment(
    scores: Sequence[Sequence[float]],
) -> Tuple[float, List[Tuple[int, int, float]]]:
    """Return the maximum-score one-to-one assignment, allowing unmatched rows."""

    if not scores:
        return 0.0, []
    candidate_count = len(scores[0])
    states: Dict[int, Tuple[float, List[Tuple[int, int, float]]]] = {0: (0.0, [])}
    for required_index, row in enumerate(scores):
        next_states = dict(states)
        for mask, (total, pairs) in states.items():
            for candidate_index in range(candidate_count):
                if mask & (1 << candidate_index):
                    continue
                value = float(row[candidate_index])
                next_mask = mask | (1 << candidate_index)
                proposal = (total + value, [*pairs, (required_index, candidate_index, value)])
                if proposal[0] > next_states.get(next_mask, (-1.0, []))[0]:
                    next_states[next_mask] = proposal
        states = next_states
    return max(states.values(), key=lambda item: item[0])


def _maximum_strict_matching(
    scores: Sequence[Sequence[float]], threshold: float = 0.99
) -> List[Tuple[int, int]]:
    """Return a maximum-cardinality one-to-one matching over strict edges."""

    if not scores:
        return []
    candidate_to_required: Dict[int, int] = {}

    def augment(required_index: int, visited: set[int]) -> bool:
        for candidate_index, value in enumerate(scores[required_index]):
            if value < threshold or candidate_index in visited:
                continue
            visited.add(candidate_index)
            previous = candidate_to_required.get(candidate_index)
            if previous is None or augment(previous, visited):
                candidate_to_required[candidate_index] = required_index
                return True
        return False

    for required_index in range(len(scores)):
        augment(required_index, set())
    return sorted(
        ((required_index, candidate_index) for candidate_index, required_index in candidate_to_required.items()),
        key=lambda pair: pair[0],
    )


def _score_distinct_containment(
    relation: Dict[str, Any],
    subject: Sequence[Component],
    obj: Sequence[Component],
    result: Dict[str, Any],
) -> Tuple[float, Dict[str, Any]]:
    """Score per-instance containment with distinct counterpart matching."""

    margin = float(relation.get("margin_px", 0.0))
    if relation["type"] == "each_inside":
        required = subject
        candidates = obj
        scores = [
            [_inside_score(inner.bbox, outer.bbox, margin) for outer in candidates]
            for inner in required
        ]
    else:
        required = subject
        candidates = obj
        scores = [
            [_inside_score(inner.bbox, outer.bbox, margin) for inner in candidates]
            for outer in required
        ]
    best_total, best_pairs = _best_distinct_assignment(scores)
    strict_pairs = _maximum_strict_matching(scores)
    expected_required_count = int(relation.get("expected_subject_count", len(required)))
    score = best_total / max(expected_required_count, 1)
    strict = (
        len(required) == expected_required_count
        and len(strict_pairs) == expected_required_count
    )
    result.update(
        {
            "score": float(score),
            "strict_pass": strict,
            "required_count": len(required),
            "expected_required_count": expected_required_count,
            "candidate_count": len(candidates),
            "strict_matched_count": len(strict_pairs),
            "matched_pairs": [
                {
                    "required_index": required_index,
                    "candidate_index": candidate_index,
                    "score": pair_score,
                }
                for required_index, candidate_index, pair_score in best_pairs
                if pair_score > 0.0
            ],
        }
    )
    return float(score), result


def _inside_score(inner: Tuple[int, int, int, int], outer: Tuple[int, int, int, int], margin: float = 0.0) -> float:
    ix0, iy0, ix1, iy1 = inner
    ox0, oy0, ox1, oy1 = outer
    checks = [
        ix0 >= ox0 + margin,
        iy0 >= oy0 + margin,
        ix1 <= ox1 - margin,
        iy1 <= oy1 - margin,
    ]
    if all(checks):
        return 1.0
    violations = [
        max(0.0, ox0 + margin - ix0),
        max(0.0, oy0 + margin - iy0),
        max(0.0, ix1 - (ox1 - margin)),
        max(0.0, iy1 - (oy1 - margin)),
    ]
    return float(max(0.0, 1.0 - sum(violations) / max(_bbox_diag(outer), 1.0)))


def _mean_centroid(components: Sequence[Component]) -> Tuple[float, float]:
    return (
        float(np.mean([comp.centroid[0] for comp in components])),
        float(np.mean([comp.centroid[1] for comp in components])),
    )


def _union_bbox(components: Sequence[Component]) -> Tuple[int, int, int, int]:
    return (
        min(comp.bbox[0] for comp in components),
        min(comp.bbox[1] for comp in components),
        max(comp.bbox[2] for comp in components),
        max(comp.bbox[3] for comp in components),
    )


def _bbox_gap(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    dx = max(bx0 - ax1, ax0 - bx1, 0)
    dy = max(by0 - ay1, ay0 - by1, 0)
    return float(math.hypot(dx, dy))


def _bbox_diag(box: Tuple[int, int, int, int]) -> float:
    x0, y0, x1, y1 = box
    return float(math.hypot(x1 - x0, y1 - y0))


def _build_frame_components(video: np.ndarray, spec: Dict[str, Any]) -> Dict[str, List[List[Component]]]:
    background_color = (spec.get("background") or {}).get("color")
    frame_components: Dict[str, List[List[Component]]] = {obj["id"]: [] for obj in spec.get("objects", [])}
    for frame in video:
        for obj in spec.get("objects", []):
            candidates = find_color_components(frame, obj["color"], background_color)
            shape = obj.get("shape", "circle")
            candidates = [comp for comp in candidates if comp.shape_scores.get(shape, 0.0) >= 0.25]
            candidates = sorted(candidates, key=lambda comp: comp.area, reverse=True)
            frame_components[obj["id"]].append(candidates)
    return frame_components


def _build_tracks(
    video: np.ndarray,
    spec: Dict[str, Any],
    frame_components: Optional[Dict[str, List[List[Component]]]] = None,
) -> Dict[str, List[Optional[Component]]]:
    if frame_components is None:
        frame_components = _build_frame_components(video, spec)
    tracks: Dict[str, List[Optional[Component]]] = {obj["id"]: [] for obj in spec.get("objects", [])}
    previous: Dict[str, Optional[Component]] = {obj["id"]: None for obj in spec.get("objects", [])}
    for frame_idx in range(len(video)):
        for obj in spec.get("objects", []):
            candidates = frame_components.get(obj["id"], [[]])[frame_idx]
            if not candidates:
                tracks[obj["id"]].append(None)
                previous[obj["id"]] = None
                continue
            prev = previous[obj["id"]]
            if prev is None:
                selected = max(candidates, key=lambda comp: comp.area)
            else:
                selected = min(candidates, key=lambda comp: _centroid_distance(comp.centroid, prev.centroid))
            tracks[obj["id"]].append(selected)
            previous[obj["id"]] = selected
    return tracks


def _score_temporal(
    temporal: Dict[str, Any],
    tracks: Dict[str, List[Optional[Component]]],
    total_frames: int,
    frame_components: Optional[Dict[str, List[List[Component]]]] = None,
    object_specs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[float, Dict[str, Any]]:
    obj_id = temporal["object"]
    track = tracks.get(obj_id, [])
    present = [comp for comp in track if comp is not None]
    result = {"type": temporal["type"], "object": obj_id, "score": 0.0, "strict_pass": False}
    if total_frames <= 0 or not track:
        return 0.0, result
    detection_rate = len(present) / total_frames
    if temporal["type"] == "persistent":
        min_rate = float(temporal.get("min_detection_rate", 0.85))
        score = min(1.0, detection_rate / max(min_rate, 1e-6))
        result.update({"score": score, "strict_pass": bool(detection_rate >= min_rate), "detection_rate": detection_rate})
        return float(score), result
    if temporal["type"] == "count_consistency":
        score, strict, count_details = _count_consistency_score(
            obj_id,
            temporal,
            frame_components or {},
            object_specs or {},
            total_frames,
        )
        result.update(
            {
                "score": float(score),
                "strict_pass": bool(strict),
                "detection_rate": detection_rate,
                **count_details,
            }
        )
        return float(score), result
    points = [(idx, comp.centroid) for idx, comp in enumerate(track) if comp is not None]
    if len(points) < 2:
        result.update({"detection_rate": detection_rate})
        return 0.0, result
    start = np.array(points[0][1], dtype=np.float32)
    end = np.array(points[-1][1], dtype=np.float32)
    disp = end - start
    frame_span = max(points[-1][0] - points[0][0], 1)
    velocities = _track_velocities(points)
    temp_type = temporal["type"]
    if temp_type == "moves_direction":
        direction = temporal.get("direction", "right")
        score, strict = _direction_score(disp, direction, float(temporal.get("min_displacement_px", 24.0)))
        result.update({"direction": direction, "displacement": disp.tolist()})
    elif temp_type == "stationary":
        max_disp = float(temporal.get("max_displacement_px", 16.0))
        distance = float(np.linalg.norm(disp))
        score = max(0.0, 1.0 - distance / max(max_disp, 1.0))
        strict = distance <= max_disp
        result.update({"displacement_distance": distance})
    elif temp_type == "velocity_range":
        speed = float(np.linalg.norm(disp) / frame_span)
        min_v = float(temporal.get("min_px_per_frame", 0.0))
        max_v = float(temporal.get("max_px_per_frame", 1e9))
        score = _interval_score(speed, min_v, max_v)
        strict = min_v <= speed <= max_v
        result.update({"speed_px_per_frame": speed, "min_px_per_frame": min_v, "max_px_per_frame": max_v})
    elif temp_type == "size_change":
        score, strict, size_details = _size_change_score(track, temporal)
        result.update(size_details)
    elif temp_type == "acceleration_trend":
        if len(velocities) < 2:
            score, strict = 0.0, False
        else:
            first = float(np.mean(velocities[: max(1, len(velocities) // 2)]))
            second = float(np.mean(velocities[max(1, len(velocities) // 2) :]))
            trend = temporal.get("trend", "speed_up")
            delta = second - first if trend == "speed_up" else first - second
            min_delta = float(temporal.get("min_delta_px_per_frame", 0.5))
            score = min(1.0, max(0.0, delta / max(min_delta, 1e-6)))
            strict = delta >= min_delta
            result.update({"first_speed": first, "second_speed": second, "speed_delta": delta, "trend": trend})
    elif temp_type == "trajectory_shape":
        shape = temporal.get("shape", "horizontal")
        score, strict = _trajectory_shape_score(points, shape)
        result.update({"shape": shape})
    elif temp_type == "temporal_order":
        score, strict = _temporal_order_score(points, temporal)
    elif temp_type == "relative_distance_change":
        ref_id = temporal.get("reference_object")
        score, strict, relative_details = _relative_distance_change_score(track, tracks.get(ref_id, []), temporal)
        result.update({"reference_object": ref_id, **relative_details})
    elif temp_type == "relative_position_change":
        ref_id = temporal.get("reference_object")
        score, strict, relative_details = _relative_position_change_score(track, tracks.get(ref_id, []), temporal)
        result.update({"reference_object": ref_id, **relative_details})
    else:
        score, strict = 0.0, False
    result.update({"score": float(score), "strict_pass": bool(strict), "detection_rate": detection_rate})
    return float(score), result


def _count_consistency_score(
    obj_id: str,
    temporal: Dict[str, Any],
    frame_components: Dict[str, List[List[Component]]],
    object_specs: Dict[str, Dict[str, Any]],
    total_frames: int,
) -> Tuple[float, bool, Dict[str, Any]]:
    obj_spec = object_specs.get(obj_id, {})
    target_count = int(temporal.get("count", obj_spec.get("count", 1)))
    min_pass_rate = float(temporal.get("min_frame_pass_rate", 0.85))
    per_frame_components = frame_components.get(obj_id, [])
    if total_frames <= 0 or not per_frame_components:
        return 0.0, False, {
            "target_count": float(target_count),
            "frame_pass_rate": 0.0,
            "mean_pred_count": 0.0,
        }
    frame_scores = []
    frame_passes = []
    pred_counts = []
    for comps in per_frame_components[:total_frames]:
        pred = len(comps)
        pred_counts.append(float(pred))
        err = abs(pred - target_count)
        frame_scores.append(max(0.0, 1.0 - err / max(target_count, 1)))
        frame_passes.append(pred == target_count)
    frame_pass_rate = float(np.mean(frame_passes)) if frame_passes else 0.0
    score = float(np.mean(frame_scores)) if frame_scores else 0.0
    return score, frame_pass_rate >= min_pass_rate, {
        "target_count": float(target_count),
        "frame_pass_rate": frame_pass_rate,
        "mean_pred_count": float(np.mean(pred_counts)) if pred_counts else 0.0,
        "min_frame_pass_rate": min_pass_rate,
    }


def _size_change_score(
    track: Sequence[Optional[Component]],
    temporal: Dict[str, Any],
) -> Tuple[float, bool, Dict[str, Any]]:
    area_points = [(idx, float(comp.area)) for idx, comp in enumerate(track) if comp is not None]
    if len(area_points) < 2:
        return 0.0, False, {
            "size_change_ratio": 1.0,
            "start_area": 0.0,
            "end_area": 0.0,
        }
    window = max(1, int(math.ceil(len(area_points) * float(temporal.get("endpoint_fraction", 0.25)))))
    start_area = float(np.mean([area for _, area in area_points[:window]]))
    end_area = float(np.mean([area for _, area in area_points[-window:]]))
    min_ratio = float(temporal.get("min_area_ratio", 1.35))
    trend = temporal.get("trend", "grow")
    if trend in ("grow", "larger", "increase"):
        ratio = end_area / max(start_area, 1.0)
    elif trend in ("shrink", "smaller", "decrease"):
        ratio = start_area / max(end_area, 1.0)
    else:
        return 0.0, False, {
            "size_change_ratio": 1.0,
            "start_area": start_area,
            "end_area": end_area,
            "trend": trend,
        }
    score = min(1.0, max(0.0, (ratio - 1.0) / max(min_ratio - 1.0, 1e-6)))
    return float(score), bool(ratio >= min_ratio), {
        "size_change_ratio": float(ratio),
        "start_area": start_area,
        "end_area": end_area,
        "min_area_ratio": min_ratio,
        "trend": trend,
    }


def _centroid_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def _track_velocities(points: Sequence[Tuple[int, Tuple[float, float]]]) -> List[float]:
    velocities = []
    for (idx_a, point_a), (idx_b, point_b) in zip(points[:-1], points[1:]):
        dt = max(idx_b - idx_a, 1)
        velocities.append(_centroid_distance(point_a, point_b) / dt)
    return velocities


def _direction_score(displacement: np.ndarray, direction: str, min_displacement: float) -> Tuple[float, bool]:
    dx, dy = float(displacement[0]), float(displacement[1])
    if direction == "right":
        forward, orth = dx, abs(dy)
    elif direction == "left":
        forward, orth = -dx, abs(dy)
    elif direction == "down":
        forward, orth = dy, abs(dx)
    elif direction == "up":
        forward, orth = -dy, abs(dx)
    else:
        return 0.0, False
    forward_score = min(1.0, max(0.0, forward / max(min_displacement, 1.0)))
    orth_penalty = min(1.0, orth / max(abs(forward), min_displacement, 1.0))
    score = max(0.0, forward_score * (1.0 - 0.5 * orth_penalty))
    return float(score), bool(forward >= min_displacement and orth <= max(abs(forward), 1.0))


def _trajectory_shape_score(points: Sequence[Tuple[int, Tuple[float, float]]], shape: str) -> Tuple[float, bool]:
    coords = np.array([point for _, point in points], dtype=np.float32)
    xs = coords[:, 0]
    ys = coords[:, 1]
    span_x = float(max(xs.max() - xs.min(), 1.0))
    span_y = float(max(ys.max() - ys.min(), 1.0))
    if shape == "horizontal":
        score = max(0.0, 1.0 - span_y / max(span_x, 1.0))
    elif shape == "vertical":
        score = max(0.0, 1.0 - span_x / max(span_y, 1.0))
    elif shape == "diagonal_down":
        score = max(0.0, float(np.corrcoef(xs, ys)[0, 1])) if len(points) >= 3 else 0.0
    elif shape == "diagonal_up":
        score = max(0.0, -float(np.corrcoef(xs, ys)[0, 1])) if len(points) >= 3 else 0.0
    elif shape == "circle":
        center = coords.mean(axis=0)
        radii = np.linalg.norm(coords - center, axis=1)
        radius_mean = float(radii.mean())
        radial_consistency = max(0.0, 1.0 - float(radii.std()) / max(radius_mean, 1.0))
        angle_span = _angle_span(coords - center)
        score = radial_consistency * min(1.0, angle_span / math.pi)
    else:
        score = 0.0
    return float(score), bool(score >= 0.75)


def _angle_span(centered_points: np.ndarray) -> float:
    angles = np.unwrap(np.arctan2(centered_points[:, 1], centered_points[:, 0]))
    return float(angles.max() - angles.min())


def _temporal_order_score(points: Sequence[Tuple[int, Tuple[float, float]]], temporal: Dict[str, Any]) -> Tuple[float, bool]:
    phases = temporal.get("phases", [])
    if len(phases) < 2 or len(points) < 4:
        return 0.0, False
    mid = len(points) // 2
    first_disp = np.array(points[mid - 1][1], dtype=np.float32) - np.array(points[0][1], dtype=np.float32)
    second_disp = np.array(points[-1][1], dtype=np.float32) - np.array(points[mid][1], dtype=np.float32)
    first_score, first_pass = _direction_score(first_disp, phases[0].get("direction", "right"), float(phases[0].get("min_displacement_px", 12.0)))
    second_score, second_pass = _direction_score(second_disp, phases[1].get("direction", "up"), float(phases[1].get("min_displacement_px", 12.0)))
    score = float((first_score + second_score) / 2.0)
    return score, bool(first_pass and second_pass)


def _paired_track_points(
    track: Sequence[Optional[Component]],
    reference_track: Sequence[Optional[Component]],
) -> List[Tuple[int, Tuple[float, float], Tuple[float, float]]]:
    pairs = []
    for idx, (comp, ref_comp) in enumerate(zip(track, reference_track)):
        if comp is None or ref_comp is None:
            continue
        pairs.append((idx, comp.centroid, ref_comp.centroid))
    return pairs


def _relative_distance_change_score(
    track: Sequence[Optional[Component]],
    reference_track: Sequence[Optional[Component]],
    temporal: Dict[str, Any],
) -> Tuple[float, bool, Dict[str, Any]]:
    pairs = _paired_track_points(track, reference_track)
    if len(pairs) < 2:
        return 0.0, False, {
            "paired_detection_rate": 0.0,
            "start_distance": 0.0,
            "end_distance": 0.0,
            "distance_change": 0.0,
        }
    distances = np.array(
        [_centroid_distance(point, reference_point) for _, point, reference_point in pairs],
        dtype=np.float32,
    )
    endpoint_fraction = float(temporal.get("endpoint_fraction", 0.25))
    window = max(1, int(math.ceil(len(distances) * endpoint_fraction)))
    start_distance = float(np.mean(distances[:window]))
    end_distance = float(np.mean(distances[-window:]))
    trend = temporal.get("trend", "toward")
    if trend in ("toward", "closer", "approach"):
        change = start_distance - end_distance
    elif trend in ("away", "farther", "separate"):
        change = end_distance - start_distance
    else:
        return 0.0, False, {
            "paired_detection_rate": len(pairs) / max(len(track), 1),
            "start_distance": start_distance,
            "end_distance": end_distance,
            "distance_change": 0.0,
            "trend": trend,
        }
    min_change = float(temporal.get("min_distance_change_px", 24.0))
    score = min(1.0, max(0.0, change / max(min_change, 1e-6)))
    return float(score), bool(change >= min_change), {
        "paired_detection_rate": len(pairs) / max(len(track), 1),
        "start_distance": start_distance,
        "end_distance": end_distance,
        "distance_change": float(change),
        "min_distance_change_px": min_change,
        "trend": trend,
    }


def _relative_position_change_score(
    track: Sequence[Optional[Component]],
    reference_track: Sequence[Optional[Component]],
    temporal: Dict[str, Any],
) -> Tuple[float, bool, Dict[str, Any]]:
    pairs = _paired_track_points(track, reference_track)
    if len(pairs) < 2:
        return 0.0, False, {
            "paired_detection_rate": 0.0,
            "start_relation_score": 0.0,
            "end_relation_score": 0.0,
        }
    endpoint_fraction = float(temporal.get("endpoint_fraction", 0.25))
    window = max(1, int(math.ceil(len(pairs) * endpoint_fraction)))
    start_point = np.mean([point for _, point, _ in pairs[:window]], axis=0)
    start_ref = np.mean([reference_point for _, _, reference_point in pairs[:window]], axis=0)
    end_point = np.mean([point for _, point, _ in pairs[-window:]], axis=0)
    end_ref = np.mean([reference_point for _, _, reference_point in pairs[-window:]], axis=0)
    margin = float(temporal.get("margin_px", 24.0))
    start_relation = temporal.get("start_relation", "left_of")
    end_relation = temporal.get("end_relation", "right_of")
    start_score, start_strict = _point_relation_score(start_point, start_ref, start_relation, margin)
    end_score, end_strict = _point_relation_score(end_point, end_ref, end_relation, margin)
    score = float((start_score + end_score) / 2.0)
    return score, bool(start_strict and end_strict), {
        "paired_detection_rate": len(pairs) / max(len(track), 1),
        "start_relation": start_relation,
        "end_relation": end_relation,
        "start_relation_score": float(start_score),
        "end_relation_score": float(end_score),
        "margin_px": margin,
    }


def _point_relation_score(
    point: Sequence[float],
    reference_point: Sequence[float],
    relation: str,
    margin: float,
) -> Tuple[float, bool]:
    point_arr = np.asarray(point, dtype=np.float32)
    ref_arr = np.asarray(reference_point, dtype=np.float32)
    delta = point_arr - ref_arr
    if relation == "left_of":
        signed = -float(delta[0])
    elif relation == "right_of":
        signed = float(delta[0])
    elif relation == "above":
        signed = -float(delta[1])
    elif relation == "below":
        signed = float(delta[1])
    else:
        return 0.0, False
    score = min(1.0, max(0.0, signed / max(margin, 1.0)))
    return float(score), bool(signed >= margin)


def _temporal_type_score(results: Sequence[Dict[str, Any]], types: set) -> float:
    values = [float(result.get("score", 0.0)) for result in results if result.get("type") in types]
    return float(np.mean(values)) if values else -10.0


def _temporal_result_mean(results: Sequence[Dict[str, Any]], key: str, default: float = 0.0) -> float:
    values = [float(result[key]) for result in results if key in result]
    return float(np.mean(values)) if values else float(default)


def _score_forbidden(spec: Dict[str, Any], components_by_color: Dict[str, List[Component]]) -> Tuple[float, List[Dict[str, Any]]]:
    details = []
    scores = []
    for forbidden in spec.get("forbidden", []):
        if forbidden.get("type") == "text":
            details.append({"type": "text", "status": "not_implemented", "score": 1.0})
            scores.append(1.0)
            continue
        color = forbidden.get("color")
        if color in components_by_color:
            pred = len(components_by_color[color])
            score = 1.0 if pred == 0 else 0.0
            details.append({"color": color, "pred_count": float(pred), "score": score})
            scores.append(score)
    return (float(np.mean(scores)) if scores else 1.0), details


def _score_extra_components(components_by_color: Dict[str, List[Component]], matched_component_ids: set) -> Tuple[float, int]:
    all_components = [comp for comps in components_by_color.values() for comp in comps]
    extra = [comp for comp in all_components if id(comp) not in matched_component_ids]
    if not all_components:
        return 1.0, 0
    score = max(0.0, 1.0 - len(extra) / max(len(all_components), 1))
    return float(score), len(extra)


def _score_semantic_extra_components(
    components_by_color: Dict[str, List[Component]],
    matched_component_ids: set,
    selected_components: Sequence[Tuple[Component, str]],
) -> Tuple[float, int]:
    """Count substantive unmatched objects while suppressing segmentation fragments."""

    all_components = [component for components in components_by_color.values() for component in components]
    largest_selected = max((component.area for component, _ in selected_components), default=0)
    min_substantive_area = max(80, int(round(0.02 * largest_selected)))
    extras: List[Component] = []
    for component in all_components:
        if id(component) in matched_component_ids:
            continue
        if component.area < min_substantive_area:
            continue
        if component.color_confidence < 0.60 or max(component.shape_scores.values(), default=0.0) < 0.30:
            continue
        aliases_requested_object = any(
            max(component.area, selected.area)
            <= (2.0 if component.color == selected.color else 1.25) * min(component.area, selected.area)
            and component.shape_scores.get(expected_shape, 0.0) >= 0.10
            and _components_alias([component], [selected])
            for selected, expected_shape in selected_components
        )
        if aliases_requested_object:
            continue
        fragment_of_selected = any(
            selected.area >= 2.85 * component.area
            and _bbox_intersection_fraction(component, selected) >= 0.50
            and max(component.shape_scores.values(), default=0.0) < 0.55
            for selected, _ in selected_components
        )
        if fragment_of_selected:
            continue
        extras.append(component)
    if not all_components:
        return 1.0, 0
    score = max(0.0, 1.0 - len(extras) / max(len(selected_components) + len(extras), 1))
    return float(score), len(extras)


def _background_score(image: np.ndarray, background_color: Optional[str]) -> float:
    if background_color not in BACKGROUND_RGB:
        return 1.0
    arr = image.astype(np.float32)
    border = np.concatenate(
        [
            arr[:24, :, :].reshape(-1, 3),
            arr[-24:, :].reshape(-1, 3),
            arr[:, :24, :].reshape(-1, 3),
            arr[:, -24:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    observed = np.median(border, axis=0)
    target = np.array(BACKGROUND_RGB[background_color], dtype=np.float32)
    dist = float(np.linalg.norm(observed - target))
    return float(max(0.0, 1.0 - dist / 180.0))


def _linear_membership(value: float, full_low: float, full_high: float, zero_low: float, zero_high: float) -> float:
    if full_low <= value <= full_high:
        return 1.0
    if value < full_low:
        return float(np.clip((value - zero_low) / max(full_low - zero_low, 1e-6), 0.0, 1.0))
    return float(np.clip((zero_high - value) / max(zero_high - full_high, 1e-6), 0.0, 1.0))


def _hue_membership(hue: float, center: float, full_radius: float, zero_radius: float) -> float:
    distance = abs((hue - center + 180.0) % 360.0 - 180.0)
    if distance <= full_radius:
        return 1.0
    return float(np.clip((zero_radius - distance) / max(zero_radius - full_radius, 1e-6), 0.0, 1.0))


def _background_score_v3(image: np.ndarray, background_color: Optional[str]) -> float:
    """Score semantic background-color ranges using the image border."""

    if background_color not in BACKGROUND_RGB:
        return 1.0
    observed, _ = _estimated_background(image)
    hue, saturation, value, _ = _rgb_to_hsv_arrays(observed.reshape(1, 1, 3))
    h = float(hue[0, 0])
    s = float(saturation[0, 0])
    v = float(value[0, 0])

    if background_color == "white":
        return min(
            _linear_membership(s, 0.0, 0.08, 0.0, 0.24),
            _linear_membership(v, 0.92, 1.0, 0.76, 1.0),
        )
    if background_color == "black":
        return _linear_membership(v, 0.0, 0.16, 0.0, 0.34)
    if background_color == "light gray":
        return min(
            _linear_membership(s, 0.0, 0.10, 0.0, 0.25),
            _linear_membership(v, 0.62, 1.0, 0.45, 1.0),
        )
    if background_color == "dark gray":
        return min(
            _linear_membership(s, 0.0, 0.12, 0.0, 0.28),
            _linear_membership(v, 0.10, 0.38, 0.02, 0.55),
        )

    pastel_ranges = {
        "beige": (42.0, 18.0, 38.0),
        "pale pink": (345.0, 24.0, 48.0),
        "pale cyan": (180.0, 24.0, 48.0),
    }
    center, full_radius, zero_radius = pastel_ranges[background_color]
    return min(
        _hue_membership(h, center, full_radius, zero_radius),
        _linear_membership(s, 0.06, 0.40, 0.015, 0.62),
        _linear_membership(v, 0.68, 1.0, 0.50, 1.0),
    )


def merge_detail_lists(detail_rows: Iterable[Dict[str, Any]]) -> Dict[str, List[Any]]:
    merged: Dict[str, List[Any]] = {}
    for row in detail_rows:
        for key, value in row.items():
            values = value if isinstance(value, list) else [value]
            merged.setdefault(key, []).extend(values)
    return merged
