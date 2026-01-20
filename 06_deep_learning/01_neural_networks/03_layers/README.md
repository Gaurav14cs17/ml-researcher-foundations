<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=120&section=header&text=Neural%20Network%20Layers&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-06-45B7D1?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<p align="center">
  <a href="../">⬆️ Back to Neural Networks</a> &nbsp;|&nbsp;
  <a href="../02_initialization/">⬅️ Prev: Initialization</a> &nbsp;|&nbsp;
  <a href="../04_neurons/">Next: Neurons ➡️</a>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/layer-types.svg" width="100%">

*Caption: Different layer types serve different purposes: Linear layers for general transformations, Conv2D for local patterns, Attention for global relationships, Normalization for stability, Dropout for regularization, and Embedding for discrete inputs.*

---

## 📂 Overview

Neural network layers are modular building blocks. Understanding each layer's purpose and mathematics helps you design effective architectures.

---

## 📐 Linear (Fully Connected) Layer

### Forward Pass

```math
y = Wx + b

```

Where $W \in \mathbb{R}^{m \times n}$, $x \in \mathbb{R}^n$, $b \in \mathbb{R}^m$, $y \in \mathbb{R}^m$

### Backward Pass (Gradients)

Given $\frac{\partial \mathcal{L}}{\partial y}$:

```math
\frac{\partial \mathcal{L}}{\partial x} = W^\top \frac{\partial \mathcal{L}}{\partial y}
\frac{\partial \mathcal{L}}{\partial W} = \frac{\partial \mathcal{L}}{\partial y} \cdot x^\top
\frac{\partial \mathcal{L}}{\partial b} = \frac{\partial \mathcal{L}}{\partial y}

```

### Parameters Count

```math
\text{Parameters} = m \times n + m = m(n + 1)

```

---

## 📐 Convolutional Layer (Conv2D)

### Forward Pass

For input $X \in \mathbb{R}^{H \times W \times C_{in}}$ and kernel $K \in \mathbb{R}^{k \times k \times C_{in} \times C_{out}}$:

```math
Y[i, j, c_{out}] = \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} \sum_{c_{in}=0}^{C_{in}-1} X[i+m, j+n, c_{in}] \cdot K[m, n, c_{in}, c_{out}] + b[c_{out}]

```

### Output Size

```math
H_{out} = \left\lfloor \frac{H + 2p - k}{s} \right\rfloor + 1
W_{out} = \left\lfloor \frac{W + 2p - k}{s} \right\rfloor + 1

```

Where $p$ = padding, $s$ = stride, $k$ = kernel size.

### Parameters Count

```math
\text{Parameters} = C_{out} \times (C_{in} \times k^2 + 1)

```

### Backward Pass

```math
\frac{\partial \mathcal{L}}{\partial X} = \text{full_conv}\left(\frac{\partial \mathcal{L}}{\partial Y}, \text{flip}(K)\right)
\frac{\partial \mathcal{L}}{\partial K} = \text{conv}\left(X, \frac{\partial \mathcal{L}}{\partial Y}\right)

```

---

## 📐 Self-Attention Layer

### Forward Pass

```math
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V

```

Where:

- $Q = XW_Q$ (queries)

- $K = XW_K$ (keys)

- $V = XW_V$ (values)

### Complexity

- Time: $O(n^2 d)$ for sequence length $n$

- Memory: $O(n^2)$ for attention matrix

### Parameters

```math
\text{Parameters} = 3d_{model}^2 + d_{model}^2 = 4d_{model}^2

```

(for $W_Q$, $W_K$, $W_V$, and $W_O$)

---

## 📐 Layer Normalization

### Forward Pass

For input $x \in \mathbb{R}^d$:

```math
\mu = \frac{1}{d} \sum_{i=1}^{d} x_i
\sigma^2 = \frac{1}{d} \sum_{i=1}^{d} (x_i - \mu)^2
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}
y_i = \gamma \hat{x}_i + \beta

```

### Backward Pass

```math
\frac{\partial \mathcal{L}}{\partial \gamma} = \sum_i \frac{\partial \mathcal{L}}{\partial y_i} \hat{x}_i
\frac{\partial \mathcal{L}}{\partial \beta} = \sum_i \frac{\partial \mathcal{L}}{\partial y_i}
\frac{\partial \mathcal{L}}{\partial x_i} = \frac{\gamma}{\sigma} \left( \frac{\partial \mathcal{L}}{\partial y_i} - \frac{1}{d}\sum_j \frac{\partial \mathcal{L}}{\partial y_j} - \frac{\hat{x}_i}{d}\sum_j \frac{\partial \mathcal{L}}{\partial y_j} \hat{x}_j \right)

```

---

## 📐 Dropout Layer

### Forward Pass (Training)

```math
\tilde{x}_i = \begin{cases}
\frac{x_i}{1-p} & \text{with probability } 1-p \\
0 & \text{with probability } p
\end{cases}

```

Using mask $m \sim \text{Bernoulli}(1-p)$:

```math
y = \frac{x \odot m}{1-p}

```

### Forward Pass (Inference)

```math
y = x

```

(No dropout, no scaling needed due to training-time scaling)

### Backward Pass

```math
\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \odot \frac{m}{1-p}

```

---

## 📐 Embedding Layer

### Forward Pass

For vocabulary size $V$ and embedding dimension $d$:

```math
E \in \mathbb{R}^{V \times d}

```

Given token index $i$:

```math
y = E[i, :] \in \mathbb{R}^d

```

(Simple table lookup, no mathematical operations)

### Backward Pass

```math
\frac{\partial \mathcal{L}}{\partial E[i, :]} = \frac{\partial \mathcal{L}}{\partial y}

```

(Gradient accumulated for each occurrence of token $i$)

---

## 📐 Softmax Layer

### Forward Pass

```math
\text{softmax}(x)_i = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}

```

### Numerical Stability

```math
\text{softmax}(x)_i = \frac{e^{x_i - \max(x)}}{\sum_{j=1}^{n} e^{x_j - \max(x)}}

```

### Jacobian

```math
\frac{\partial s_i}{\partial x_j} = \begin{cases}
s_i(1 - s_i) & \text{if } i = j \\
-s_i s_j & \text{if } i \neq j
\end{cases}

```

Or in matrix form:

```math
\frac{\partial s}{\partial x} = \text{diag}(s) - s s^\top

```

---

## 📊 Layer Comparison Table

| Layer | Purpose | Parameters | Complexity |
|-------|---------|------------|------------|
| **Linear** | Transform | $m(n+1)$ | $O(mn)$ |
| **Conv2D** | Local patterns | $C_{out}(C_{in}k^2+1)$ | $O(HWk^2C_{in}C_{out})$ |
| **Attention** | Global relations | $4d^2$ | $O(n^2d)$ |
| **LayerNorm** | Normalize | $2d$ | $O(d)$ |
| **Dropout** | Regularize | $0$ | $O(n)$ |
| **Embedding** | Lookup | $Vd$ | $O(d)$ |

---

## 💻 Code Examples

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Linear (Fully Connected)
linear = nn.Linear(in_features=512, out_features=256)
# Parameters: 512 * 256 + 256 = 131,328

# Convolutional
conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
# Parameters: 64 * (3 * 3 * 3 + 1) = 1,792

# Multi-head Attention
attention = nn.MultiheadAttention(embed_dim=512, num_heads=8)
# Parameters: 4 * 512 * 512 = 1,048,576

# Layer Normalization
norm = nn.LayerNorm(normalized_shape=512)
# Parameters: 512 + 512 = 1,024 (gamma and beta)

# Dropout
dropout = nn.Dropout(p=0.1)
# Parameters: 0 (just masks during training)

# Embedding
embedding = nn.Embedding(num_embeddings=50000, embedding_dim=512)
# Parameters: 50000 * 512 = 25,600,000

# Custom Layer Implementation
class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) / in_features**0.5)
        self.bias = nn.Parameter(torch.zeros(out_features))
    
    def forward(self, x):
        return x @ self.weight.T + self.bias

class CustomLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta

class CustomDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    
    def forward(self, x):
        if self.training:
            mask = torch.bernoulli(torch.full_like(x, 1 - self.p))
            return x * mask / (1 - self.p)
        return x

```

---

## 🔗 Layer Composition Patterns

### Residual Block

```python
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, dim)
    
    def forward(self, x):
        return x + self.linear(F.gelu(self.norm(x)))

```

### Transformer Block

```python
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, 4*dim),
            nn.GELU(),
            nn.Linear(4*dim, dim)
        )
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ff(self.norm2(x))
        return x

```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | PyTorch nn.Module | [Docs](https://pytorch.org/docs/stable/nn.html) |
| 🎥 | 3Blue1Brown: Neural Networks | [YouTube](https://www.youtube.com/watch?v=aircAruvnKk) |
| 📄 | Attention Is All You Need | [arXiv](https://arxiv.org/abs/1706.03762) |
| 📄 | Layer Normalization | [arXiv](https://arxiv.org/abs/1607.06450) |
| 📄 | Dropout Paper | [JMLR](https://jmlr.org/papers/v15/srivastava14a.html) |
| 🇨🇳 | 神经网络层详解 | [知乎](https://zhuanlan.zhihu.com/p/25110450) |

---

## 🔗 Where This Topic Is Used

| Layer Type | Application |
|------------|------------|
| **Linear/Dense** | Classification, MLP, Transformer FFN |
| **Conv2D** | Image processing, CNNs |
| **LSTM/GRU** | Sequence modeling |
| **Attention** | Transformers, Vision Transformers |
| **Embedding** | Token representations, NLP |
| **LayerNorm** | Transformers (pre/post-norm) |
| **Dropout** | Regularization during training |

---

<p align="center">
  <a href="../">⬆️ Back to Neural Networks</a> &nbsp;|&nbsp;
  <a href="../02_initialization/">⬅️ Prev: Initialization</a> &nbsp;|&nbsp;
  <a href="../04_neurons/">Next: Neurons ➡️</a>
</p>

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=80&section=footer" width="100%"/>
</p>
