<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=FF6B6B&height=120&section=header&text=Machine%20Learning%20Optimization&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-04-FF6B6B?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

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

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=FF6B6B&height=80&section=footer" width="100%"/>
</p>
