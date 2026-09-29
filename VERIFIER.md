# VVRBench verifier

`vvr_bench/verifier.py` is the complete frozen verifier library used for
VVRBench evaluation. There is one public implementation and one scoring API.

```text
SHA256: 2c3661be030c078e69c53284949524b4473150c67fd23bb294ffaafd7fa64f77
Image API: score_image_spec(image, spec, reward_mode="default")
Image/video API: score_sample_spec(sample, spec, reward_mode="default")
```

## Constraint families

The verifier implements 46 constraint types in five families (`vvr_bench/taxonomy.py`):

| Family | Constraint types |
|---|---|
| Grounding | `color_attribute`, `shape_attribute`, `color_shape_binding` |
| Cardinality | `exact_count`, `same_count`, `more_than_count`, `fewer_than_count`, `times_as_many` |
| Spatial | `absolute_region`, `grid_occupancy`, `left_of`, `right_of`, `above`, `below`, `all_left_of`, `all_right_of`, `all_above`, `all_below`, `leftmost`, `rightmost`, `topmost`, `bottommost`, `between`, `same_row`, `same_column`, `all_same_row`, `all_same_column`, `not_all_same_row`, `not_all_same_column`, `closer_than`, `farther_than` |
| Size | `larger_than`, `smaller_than`, `same_size`, `all_larger_than`, `all_smaller_than`, `all_same_size`, `not_all_same_size`, `largest`, `smallest` |
| Topology | `touching`, `not_touching`, `inside`, `contains`, `each_inside`, `each_contains` |

Every task also checks its background color and that no unrequested colored objects appear.

Repeated-group size uses worst-case, all-to-all semantics. "Every A is larger
than every B" passes only when the smallest detected A is larger than the
largest detected B. It never compares group averages. Relative size is measured
with robust visible extent: the square root of the product of the central 90%
horizontal and vertical extents.

The same per-object shape decisions drive per-constraint diagnostics and
strict accuracy. The dense score is partial credit and must not be reported as
accuracy.

## Validation

The frozen source passed:

- 290 historical edge cases and 360 direct checks;
- 14,788 metamorphic checks;
- 320 historical atomic-matrix cases;
- 30 adversarial group-size cases;
- all nine ordered circle/square/triangle group-size combinations and their
  inverse controls;
- 10,000 VVRBench Core oracle images and 10,000 targeted counterfactuals;
- 720 released Challenge oracle images and 720 targeted counterfactuals.

VVRBench Core data is unchanged by this verifier-library update. The released
Challenge uses the same verifier and differs only in prompt complexity.

## Integrity

Verify the installed source before evaluation:

```bash
sha256sum vvr_bench/verifier.py
```

Any change to predicate semantics, thresholds, or image processing requires a
new frozen source hash and complete revalidation. Do not patch the file in
place for benchmark comparisons.
