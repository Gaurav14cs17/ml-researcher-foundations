<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=Convex%20Optimization&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---

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

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>
