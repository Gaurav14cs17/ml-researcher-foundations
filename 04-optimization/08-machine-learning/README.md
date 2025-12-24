# 🤖 Machine Learning Optimization

> **Stochastic methods for large-scale ML**

## 🎯 Visual Overview

<img src="./images/ml-optimization.svg" width="100%">

*Caption: ML optimization uses SGD and adaptive methods (Adam). Training dynamics: warmup → constant/decay → convergence. Key considerations: batch size, learning rate, gradient clipping.*

---

## 📐 Mathematical Foundations

### SGD Update Rule
```
θₜ₊₁ = θₜ - η ∇L_B(θₜ)

Where:
• B = mini-batch (random subset)
• η = learning rate
• E[∇L_B] = ∇L (unbiased)
```

### Adam Update Rule
```
mₜ = β₁mₜ₋₁ + (1-β₁)gₜ      (momentum)
vₜ = β₂vₜ₋₁ + (1-β₂)gₜ²     (variance)
m̂ₜ = mₜ/(1-β₁ᵗ)             (bias correction)
v̂ₜ = vₜ/(1-β₂ᵗ)
θₜ₊₁ = θₜ - η m̂ₜ/(√v̂ₜ + ε)
```

### Convergence Bounds
```
SGD on convex f with variance σ²:
E[f(θ̄ₜ)] - f* ≤ O(||θ₀ - θ*||²/ηT + ηLσ²)

Optimal η ∝ 1/√T gives O(1/√T) rate
```

---

## 📂 Topics

| Folder | Topic | Used In |
|--------|-------|---------|
| [sgd/](./sgd/) | SGD & Variants | ResNet, BERT |
| [adam/](./adam/) | Adam Optimizer | GPT, Stable Diffusion |

## 🔗 Graph

```
[constrained] + [convex]
        ↓
     sgd.md
        ↓
     adam.md
        ↓
  [metaheuristics]
```

---

---

## 🔗 Where This Topic Is Used

| Application | Usage |
|-------------|-------|
| **Machine Learning** | Core concept |
| **Deep Learning** | Foundation for neural networks |
| **Research** | Important for understanding papers |

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | Adam Paper | [arXiv](https://arxiv.org/abs/1412.6980) |
| 📄 | AdamW Paper | [arXiv](https://arxiv.org/abs/1711.05101) |
| 📖 | Deep Learning Book Ch 8 | [Book](https://www.deeplearningbook.org/) |
| 🇨🇳 | 优化器详解 | [知乎](https://zhuanlan.zhihu.com/p/32230623) |

---

⬅️ [Back: Integer Programming](../07-integer-programming/) | ➡️ [Next: Metaheuristics](../09-metaheuristics/)
