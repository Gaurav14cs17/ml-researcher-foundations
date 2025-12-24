# Mixed Integer LP

## Overview

LP with integer constraints. NP-hard but powerful for modeling.

## Key Formula

```
MILP:
min  cᵀx + dᵀy
s.t. Ax + By ≤ b
     x ∈ ℤⁿ (integer)
     y ∈ ℝᵐ (continuous)

LP Relaxation:
Replace x ∈ ℤⁿ with x ∈ ℝⁿ
(provides lower bound)
```

## Key Concepts

- **LP Relaxation** - Drop integrality for bound
- **Big-M** - Model logical constraints
- **Integrality Gap** - Relaxation vs integer optimum
- **Cutting Planes** - Tighten relaxation

---

---

⬅️ [Back: Branch Bound](./branch-bound.md)
