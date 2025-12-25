<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=FF6B6B&height=120&section=header&text=Stochastic%20Gradient%20Descent%20SG&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-04-FF6B6B?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 📐 Mathematical Foundations

### Vanilla SGD
```
θₜ₊₁ = θₜ - η ∇L_B(θₜ)

Where B is a mini-batch (random subset)
E[∇L_B] = ∇L (unbiased estimator)
```

### SGD with Momentum
```
vₜ₊₁ = β vₜ + ∇L_B(θₜ)
θₜ₊₁ = θₜ - η vₜ₊₁

β typically 0.9 (exponential moving average of gradients)
```

### Nesterov Accelerated Gradient
```
θ_lookahead = θₜ - β vₜ
vₜ₊₁ = β vₜ + ∇L_B(θ_lookahead)
θₜ₊₁ = θₜ - η vₜ₊₁

Evaluates gradient at "lookahead" position
```

### Convergence Rate
```
For convex functions with σ² gradient variance:
E[f(θₜ) - f*] ≤ O(1/√t) + O(σ²/η)

Learning rate schedule:
ηₜ = η₀ / √t or step decay
```

---

## 📂 Subtopics

| File | Topic | Application |
|------|-------|-------------|
| [vanilla-sgd.md](./vanilla-sgd.md) | Basic SGD | Neural network training |
| [momentum.md](./momentum.md) | SGD with Momentum | Faster convergence |

---

## 🌍 Where SGD is Used

| Application | How | Paper/Reference |
|-------------|-----|-----------------|
| **GPT/LLM Training** | Mini-batch SGD on billions of tokens | [GPT-3 Paper](https://arxiv.org/abs/2005.14165) |
| **Diffusion Models** | Denoising score matching with SGD | [DDPM](https://arxiv.org/abs/2006.11239) |
| **ResNet/ImageNet** | SGD with momentum, batch norm | [ResNet](https://arxiv.org/abs/1512.03385) |
| **Recommendation Systems** | Matrix factorization with SGD | Netflix Prize |
| **Reinforcement Learning** | Policy gradient (a form of SGD) | [PPO](https://arxiv.org/abs/1707.06347) |

---

## 🔗 Dependency Graph

```
foundations/linear-algebra
         |
         v
    basic-methods/gradient-descent
         |
         v
+--------+--------+
|   SGD Variants  |
+-----------------+
| • vanilla-sgd   |
| • momentum      |
| • nesterov      |
| • learning-rates|
+--------+--------+
         |
         v
    machine-learning/adam
```

---

## 📚 References

| Type | Resource | Link |
|------|----------|------|
| 📄 | Robbins & Monro (1951) | Original stochastic approximation |
| 📄 | SGD Convergence | [Bottou et al., 2018](https://arxiv.org/abs/1606.04838) |
| 📖 | Goodfellow Ch. 8 | [Deep Learning Book](https://www.deeplearningbook.org/) |
| 🎥 | Stanford CS231n | [Optimization Lecture](http://cs231n.stanford.edu/) |
| 🇨🇳 | SGD优化详解 | [知乎](https://zhuanlan.zhihu.com/p/22252270) |
| 🇨🇳 | 动量法原理 | [机器之心](https://www.jiqizhixin.com/articles/2017-07-12-8) |

---

⬅️ [Back: Adam](../adam/)

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=FF6B6B&height=80&section=footer" width="100%"/>
</p>
