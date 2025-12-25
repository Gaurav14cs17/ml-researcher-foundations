<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=Adam%20Optimizer&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---

# Adam Optimizer

> **Adaptive Moment Estimation - the default for deep learning**

---

## 📐 Algorithm

```
Initialize m₀ = 0, v₀ = 0, t = 0

For each step:
    t = t + 1
    g = ∇f(θ)                           # Gradient
    m = β₁m + (1-β₁)g                   # First moment
    v = β₂v + (1-β₂)g²                  # Second moment
    m̂ = m / (1-β₁ᵗ)                     # Bias correction
    v̂ = v / (1-β₂ᵗ)                     # Bias correction
    θ = θ - α·m̂ / (√v̂ + ε)              # Update
```

---

## 🔑 Hyperparameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| α (lr) | 0.001 | Learning rate |
| β₁ | 0.9 | First moment decay |
| β₂ | 0.999 | Second moment decay |
| ε | 1e-8 | Numerical stability |

---

## 💻 Code

```python
import torch

# Standard Adam
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999)
)

# AdamW (for transformers)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=0.01
)
```

---

## 🌍 When to Use

| Model Type | Recommended |
|------------|-------------|
| Transformers | AdamW |
| CNNs | SGD + Momentum |
| General | Adam |

---

---

➡️ [Next: Sgd](./sgd.md)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>
