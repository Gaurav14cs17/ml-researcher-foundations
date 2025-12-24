<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=Stochastic&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---


# Stochastic Optimization

> **Learning from random samples**

---

## 🎯 Visual Overview

<img src="./images/stochastic-opt.svg" width="100%">

*Caption: SGD: θ ← θ - η∇f_i(θ) uses mini-batch gradients. Unbiased estimate of full gradient. Noise helps escape local minima. Variance reduction: momentum, Adam, batch size tuning.*

---

## 📂 Overview

SGD and its variants power all of modern deep learning. The noise from mini-batches provides implicit regularization and enables training on datasets too large to fit in memory.

---

## 📐 Mathematical Definitions

### Stochastic Gradient Descent (SGD)
```
θₜ₊₁ = θₜ - ηₜ ∇f_i(θₜ)

Where:
• i ~ Uniform{1,...,n} (random sample)
• ηₜ = learning rate (step size)
• E[∇f_i(θ)] = ∇f(θ) (unbiased!)
```

### Mini-batch SGD
```
θₜ₊₁ = θₜ - ηₜ (1/|B|) Σᵢ∈B ∇fᵢ(θₜ)

Variance scales as σ²/|B|
• Larger batch → lower variance → smoother
• Smaller batch → more noise → regularization
```

### Convergence Rate
```
Convex f:
E[f(θₜ) - f(θ*)] = O(1/√T)

Strongly convex f:
E[||θₜ - θ*||²] = O(1/T)

With diminishing learning rate: ηₜ = η₀/√t
```

### Momentum
```
vₜ = βvₜ₋₁ + ∇f(θₜ)
θₜ₊₁ = θₜ - ηvₜ

• β ≈ 0.9 typical
• Accelerates along consistent gradient directions
• Dampens oscillations
```

### Adam
```
mₜ = β₁mₜ₋₁ + (1-β₁)∇f(θₜ)     # First moment
vₜ = β₂vₜ₋₁ + (1-β₂)(∇f(θₜ))²  # Second moment

m̂ₜ = mₜ/(1-β₁ᵗ)  # Bias correction
v̂ₜ = vₜ/(1-β₂ᵗ)

θₜ₊₁ = θₜ - η m̂ₜ/(√v̂ₜ + ε)
```

---

## 💻 Code Examples

```python
import torch
import torch.optim as optim

# SGD with momentum
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Adam (most common for transformers)
optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))

# AdamW (Adam + proper weight decay)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

# Manual SGD implementation
def sgd_step(params, grads, lr):
    for p, g in zip(params, grads):
        p.data -= lr * g

# Training loop with mini-batches
for epoch in range(epochs):
    for batch in dataloader:  # Random mini-batches
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()
        optimizer.step()
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | Adam Paper | [arXiv](https://arxiv.org/abs/1412.6980) |
| 📄 | AdamW Paper | [arXiv](https://arxiv.org/abs/1711.05101) |
| 📖 | Bottou: SGD Tricks | [Paper](https://leon.bottou.org/publications/pdf/tricks-2012.pdf) |
| 🇨🇳 | SGD优化器详解 | [知乎](https://zhuanlan.zhihu.com/p/32230623) |
| 🇨🇳 | Adam原理 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88318597) |
| 🇨🇳 | 优化器对比 | [B站](https://www.bilibili.com/video/BV1J94y1f7u5) |

---

<- [Back](../)

---

## 🔗 Where This Topic Is Used

| Application | Usage |
|-------------|-------|
| **Machine Learning** | Core concept for ML systems |
| **Deep Learning** | Foundation for neural networks |
| **Research** | Important for understanding papers |

---

---

⬅️ [Back: stochastic](../)

---

⬅️ [Back: Second Order](../second-order/)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>
