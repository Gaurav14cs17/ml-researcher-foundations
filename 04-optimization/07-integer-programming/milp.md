<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=Mixed%20Integer%20LP&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---

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

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>
