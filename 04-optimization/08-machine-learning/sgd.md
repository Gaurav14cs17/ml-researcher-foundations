<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=SGD%20%20Variants&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---

# SGD & Variants

## Overview

Stochastic gradient descent for large datasets. Use mini-batches.

## Key Formula

```
SGD Update:
θₜ₊₁ = θₜ - αₜ∇f_{iₜ}(θₜ)

Momentum:
vₜ = βvₜ₋₁ + ∇f_{iₜ}(θₜ)
θₜ₊₁ = θₜ - αvₜ

Nesterov:
vₜ = βvₜ₋₁ + ∇f(θₜ - αβvₜ₋₁)
```

## Key Concepts

- **Mini-batch** - Subset for gradient estimate
- **Learning Rate Schedule** - Decay over time
- **Momentum** - Average gradients, accelerate
- **Variance Reduction** - SVRG, SAGA

---

---

⬅️ [Back: Adam](./adam.md)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>
