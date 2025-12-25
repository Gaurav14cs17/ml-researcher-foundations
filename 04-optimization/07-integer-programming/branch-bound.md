<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=Branch%20and%20Bound&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---

# Branch and Bound

## Overview

Divide and conquer for integer programming. Use bounds to prune.

## Key Formula

```
Algorithm:
1. Solve LP relaxation (bound)
2. If integer → incumbent
3. If worse than incumbent → prune
4. Branch: xᵢ ≤ ⌊x*ᵢ⌋ and xᵢ ≥ ⌈x*ᵢ⌉
5. Repeat

Gap = (UB - LB) / UB
```

## Key Concepts

- **Branching** - Split on fractional variable
- **Bounding** - LP relaxation gives bound
- **Pruning** - Discard suboptimal branches
- **Node Selection** - DFS, BFS, best-first

---

---

➡️ [Next: Milp](./milp.md)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>
