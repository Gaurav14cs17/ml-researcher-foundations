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
