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
