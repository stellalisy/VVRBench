"""Canonical VVR-Bench family taxonomy used for public reporting."""

from __future__ import annotations

from collections.abc import Iterable


VERIFIER_TAXONOMY = {
    "grounding": {
        "color_attribute",
        "shape_attribute",
        "color_shape_binding",
    },
    "cardinality": {
        "exact_count",
        "same_count",
        "more_than_count",
        "fewer_than_count",
        "times_as_many",
    },
    "spatial": {
        "absolute_region",
        "left_of",
        "right_of",
        "above",
        "below",
        "all_left_of",
        "all_right_of",
        "all_above",
        "all_below",
        "leftmost",
        "rightmost",
        "topmost",
        "bottommost",
        "between",
        "same_row",
        "same_column",
        "all_same_row",
        "all_same_column",
        "not_all_same_row",
        "not_all_same_column",
        "grid_occupancy",
        "closer_than",
        "farther_than",
    },
    "size": {
        "larger_than",
        "smaller_than",
        "same_size",
        "all_larger_than",
        "all_smaller_than",
        "all_same_size",
        "not_all_same_size",
        "largest",
        "smallest",
    },
    "topology": {
        "touching",
        "not_touching",
        "inside",
        "contains",
        "each_inside",
        "each_contains",
    },
}
ATOMIC_TO_FAMILY = {
    atomic: family
    for family, atomics in VERIFIER_TAXONOMY.items()
    for atomic in atomics
}
ATOMIC_TO_FAMILY["twice_as_many"] = "cardinality"


def canonical_families(criteria: Iterable[str]) -> list[str]:
    criteria = [str(value) for value in criteria]
    unknown = sorted(set(criteria) - ATOMIC_TO_FAMILY.keys())
    if unknown:
        raise ValueError(f"unknown atomic verifications: {unknown}")
    active = {ATOMIC_TO_FAMILY[criterion] for criterion in criteria}
    return [family for family in VERIFIER_TAXONOMY if family in active]


def family_count_tier(count: int) -> str:
    if count == 1:
        return "single_family"
    if count == 2:
        return "two_family"
    if count == 3:
        return "three_family"
    if count in {4, 5}:
        return "four_or_more_family"
    raise ValueError(f"invalid canonical family count: {count}")
