<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=Layers&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---


# Neural Network Layers

> **Building blocks of deep learning architectures**

---

## 🎯 Visual Overview

<img src="./images/layer-types.svg" width="100%">

*Caption: Different layer types serve different purposes: Linear layers for general transformations, Conv2D for local patterns, Attention for global relationships, Normalization for stability, Dropout for regularization, and Embedding for discrete inputs.*

---

## 📂 Overview

Neural network layers are modular building blocks. Understanding each layer's purpose helps you design effective architectures.

---

## 🔑 Common Layer Types

| Layer | Purpose | Common Use |
|-------|---------|------------|
| **Linear (Dense)** | General transformation | MLPs, output heads |
| **Conv2D** | Local patterns | Images, CNNs |
| **Attention** | Global relationships | Transformers |
| **LayerNorm** | Stabilize training | Transformers |
| **BatchNorm** | Stabilize training | CNNs |
| **Dropout** | Regularization | Training |
| **Embedding** | Discrete → dense | NLP, categorical |

---

## 📐 Layer Mathematics

```python
# Linear: y = Wx + b
# Conv2D: y[i,j] = Σ W[m,n] * x[i+m, j+n] + b
# Attention: softmax(QK^T / √d) V
# LayerNorm: (x - μ) / σ * γ + β
# Dropout: mask * x / (1-p)  (training)
```

---

## 💻 Code

```python
import torch.nn as nn

# Common layers in PyTorch
linear = nn.Linear(in_features=512, out_features=256)
conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
attention = nn.MultiheadAttention(embed_dim=512, num_heads=8)
norm = nn.LayerNorm(normalized_shape=512)
dropout = nn.Dropout(p=0.1)
embedding = nn.Embedding(num_embeddings=50000, embedding_dim=512)
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | PyTorch nn.Module | [Docs](https://pytorch.org/docs/stable/nn.html) |
| 🎥 | 3Blue1Brown: Neural Networks | [YouTube](https://www.youtube.com/watch?v=aircAruvnKk) |
| 📄 | Attention Is All You Need | [arXiv](https://arxiv.org/abs/1706.03762) |
| 🇨🇳 | 神经网络层详解 | [知乎](https://zhuanlan.zhihu.com/p/25110450) |
| 🇨🇳 | PyTorch层操作 | [B站](https://www.bilibili.com/video/BV1Y64y1Q7hi) |
| 🇨🇳 | 注意力机制详解 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88318597) |


## 🔗 Where This Topic Is Used

| Layer Type | Application |
|------------|------------|
| **Linear/Dense** | Classification, MLP |
| **Conv2D** | Image processing |
| **LSTM/GRU** | Sequence modeling |
| **Attention** | Transformers |
| **Embedding** | Token representations |

---

⬅️ [Back: Neural Networks](../)

---

⬅️ [Back: Initialization](../initialization/) | ➡️ [Next: Neurons](../neurons/)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>
