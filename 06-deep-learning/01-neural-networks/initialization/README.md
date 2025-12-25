<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=120&section=header&text=Weight%20Initialization&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-06-45B7D1?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/weight-init.svg" width="100%">

*Caption: Proper initialization keeps gradients from vanishing or exploding through layers. Xavier/Glorot works for tanh/sigmoid, while Kaiming/He is designed for ReLU networks. Modern LLMs often use small constant initialization.*

---

## 📂 Overview

Weight initialization determines the starting point of optimization. Bad initialization can make training impossible; good initialization enables fast, stable convergence.

---

## 🔑 Initialization Methods

| Method | Formula | Best For |
|--------|---------|----------|
| **Xavier/Glorot** | N(0, 2/(n_in + n_out)) | tanh, sigmoid, GELU |
| **Kaiming/He** | N(0, 2/n_in) | ReLU, LeakyReLU |
| **Orthogonal** | QR decomposition | RNNs |
| **Small Constant** | N(0, 0.02) | LLMs (GPT-style) |

---

## 📐 The Problem

```
Forward: h_l = W_l * h_{l-1}
         Var(h_l) = n_{l-1} * Var(W_l) * Var(h_{l-1})

If Var(W) too small: Var(h) → 0 (vanishing)
If Var(W) too large: Var(h) → ∞ (exploding)

Solution: Set Var(W) = 1/n_in (or 2/n_in for ReLU)
```

---

## 💻 Code

```python
import torch.nn as nn

# Xavier (for tanh, sigmoid, GELU)
nn.init.xavier_uniform_(layer.weight)
nn.init.xavier_normal_(layer.weight)

# Kaiming (for ReLU)
nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')

# Orthogonal (for RNNs)
nn.init.orthogonal_(layer.weight)

# Small constant (for LLMs)
nn.init.normal_(layer.weight, mean=0.0, std=0.02)
nn.init.zeros_(layer.bias)
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | Xavier/Glorot Paper | [AISTATS 2010](http://proceedings.mlr.press/v9/glorot10a.html) |
| 📄 | Kaiming/He Paper | [arXiv](https://arxiv.org/abs/1502.01852) |
| 📖 | PyTorch init | [Docs](https://pytorch.org/docs/stable/nn.init.html) |
| 🇨🇳 | 权重初始化详解 | [知乎](https://zhuanlan.zhihu.com/p/25110450) |
| 🇨🇳 | Xavier与Kaiming初始化 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88619088) |
| 🇨🇳 | 深度学习初始化方法 | [B站](https://www.bilibili.com/video/BV1Y64y1Q7hi) |


## 🔗 Where This Topic Is Used

| Init Method | Best For |
|-------------|---------|
| **Xavier/Glorot** | tanh, sigmoid |
| **He/Kaiming** | ReLU family |
| **Orthogonal** | RNNs |
| **Pretrained** | Transfer learning |

---

⬅️ [Back: Neural Networks](../)

---

⬅️ [Back: Activations](../activations/) | ➡️ [Next: Layers](../layers/)

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=80&section=footer" width="100%"/>
</p>
