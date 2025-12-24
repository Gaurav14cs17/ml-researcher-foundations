# Convex Optimization

## Overview

Special class where any local minimum is global. Efficient algorithms exist.

## Key Formula

```
Convex Function:
f(θx + (1-θ)y) ≤ θf(x) + (1-θ)f(y)
for θ ∈ [0,1]

First-Order Condition:
f(y) ≥ f(x) + ∇f(x)ᵀ(y-x)

Second-Order:
∇²f(x) ≽ 0 (positive semidefinite)
```

## Key Concepts

- **Convex Sets** - Line segment stays in set
- **Convex Functions** - Bowl shape, curves up
- **Local = Global** - No bad local optima
- **DCP** - Rules for combining convex functions

## Hierarchy

```
LP ⊂ QP ⊂ SOCP ⊂ SDP ⊂ Convex
```

---

🏠 [Home](../README.md)

