<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=Mlp&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---


# Multi-Layer Perceptron (MLP)

> **The foundation of all neural networks**

---

## 🎯 Visual Overview

<img src="./images/mlp.svg" width="100%">

*Caption: An MLP consists of input, hidden, and output layers. Each layer performs a linear transformation followed by a nonlinear activation: h = σ(Wx + b). MLPs are universal function approximators.*

---

## 📂 Overview

The Multi-Layer Perceptron is the simplest form of neural network. Despite being "basic," MLPs are powerful function approximators and the building block for more complex architectures.

---

## 📐 Mathematical Foundations

### Forward Pass
```
Layer l:
hₗ = σ(Wₗhₗ₋₁ + bₗ)

Full network:
f(x) = WₗσWₗ₋₁σ...σW₁x + biases

h₁ = σ(W₁x + b₁)
h₂ = σ(W₂h₁ + b₂)
y = W₃h₂ + b₃
```

### Backward Pass (Gradient)
```
δₗ = ∂L/∂hₗ

δₗ₋₁ = (Wₗᵀδₗ) ⊙ σ'(zₗ₋₁)  where zₗ = Wₗhₗ₋₁ + bₗ

∂L/∂Wₗ = δₗ hₗ₋₁ᵀ
∂L/∂bₗ = δₗ
```

### Universal Approximation
```
Theorem (Hornik, 1989):
MLP with one hidden layer can approximate any
continuous function on compact subsets of ℝⁿ
to arbitrary precision, given enough neurons.

f(x) ≈ Σᵢ₌₁ᴺ αᵢ σ(wᵢᵀx + bᵢ)
```

---

## 🔑 Key Topics

| Topic | Description |
|-------|-------------|
| Fundamentals | Core concepts |
| Applications | ML use cases |
| Implementation | Code examples |

---

## 💻 Example

```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.act = nn.GELU()
    
    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return self.fc3(x)
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | Universal Approximation | [Paper](https://cognitivemedium.com/magic_paper/assets/Hornik.pdf) |
| 🎥 | 3Blue1Brown: Neural Networks | [YouTube](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) |
| 📖 | PyTorch nn.Linear | [Docs](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) |
| 🇨🇳 | MLP详解 | [知乎](https://zhuanlan.zhihu.com/p/25110450) |
| 🇨🇳 | 多层感知机实现 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88619088) |
| 🇨🇳 | 神经网络入门 | [B站](https://www.bilibili.com/video/BV1J94y1f7u5) |

---

<- [Back](../)

---

## 🔗 Where This Topic Is Used

| Application | MLP Usage |
|-------------|----------|
| **Classification** | Output layer |
| **Regression** | Function approximation |
| **Transformer FFN** | Position-wise FFN |
| **Feature Extraction** | Hidden layers |

---

---

⬅️ [Back: mlp](../)

---

⬅️ [Back: Diffusion](../diffusion/) | ➡️ [Next: Moe](../moe/)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>
