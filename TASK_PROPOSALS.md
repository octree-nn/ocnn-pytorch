# Codebase task proposals

## 1) Typo fix task
- **Task:** Fix typoed comments in `to()` methods: change "save device" -> "same device" and "retrun" -> "return".
- **Why:** The same typo appears in multiple places and degrades readability.
- **Evidence:** `ocnn/octree/octree.py` and `ocnn/octree/points.py` comments above early-return path in `to()`.

## 2) Bug fix task
- **Task:** Make `Transform.preprocess()` robust when `sample` has no `normals` key (fallback to `None` or computed normals), instead of unconditionally `pop('normals')`.
- **Why:** Current code assumes every sample includes normals; datasets without normals will raise `KeyError` before augmentation/octree conversion.
- **Evidence:** `ocnn/dataset.py` currently calls `sample.pop('normals')` directly.

## 3) Comment/documentation discrepancy task
- **Task:** Align docstrings with actual signatures in `Transform`:
  - Rename documented parameter `jittor` to `jitter`.
  - Clarify that `flip` is also an init parameter.
- **Why:** The class docstring does not match `__init__`, which increases onboarding friction and can cause config mistakes.
- **Evidence:** `ocnn/dataset.py` class docstring vs `Transform.__init__(..., jitter: float, flip: list, ...)`.

## 4) Test improvement task
- **Task:** Add a targeted regression test in `test/test_octree_conv_t.py` for the currently skipped/high-error configuration (`depth=7`, `out_ratio=2.0`) and mark it with explicit expected tolerance/xfail rationale.
- **Why:** There is an inline TODO documenting known numerical error for this setup; converting it into an explicit test will prevent silent regressions and document expected behavior.
- **Evidence:** Inline TODO in `test/test_octree_conv_t.py` near the parameterized cases.
