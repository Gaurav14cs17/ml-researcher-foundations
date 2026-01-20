<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=120&section=header&text=Weight%20Initialization&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-06-45B7D1?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<p align="center">
  <a href="../">⬆️ Back to Neural Networks</a> &nbsp;|&nbsp;
  <a href="../01_activations/">⬅️ Prev: Activations</a> &nbsp;|&nbsp;
  <a href="../03_layers/">Next: Layers ➡️</a>
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

## 📐 Mathematical Foundations

### The Variance Propagation Problem

**Forward Pass Analysis:**

Consider a linear layer: $y = Wx$ where $W \in \mathbb{R}^{n\_{out} \times n\_{in}}$

Assuming:
- $w\_{ij} \sim \mathcal{N}(0, \sigma\_w^2)$ independent
- $x\_i \sim \mathcal{N}(0, \sigma\_x^2)$ independent
- $w\_{ij}$ and $x\_i$ are independent

Then:

$$
y_j = \sum_{i=1}^{n_{in}} w_{ji} x_i
\mathbb{E}[y_j] = \sum_{i=1}^{n_{in}} \mathbb{E}[w_{ji}] \mathbb{E}[x_i] = 0
\text{Var}(y_j) = \sum_{i=1}^{n_{in}} \text{Var}(w_{ji} x_i) = n_{in} \cdot \sigma_w^2 \cdot \sigma_x^2
$$

**Key Insight:**

$$
\text{Var}(y) = n_{in} \cdot \sigma_w^2 \cdot \text{Var}(x)
$$

To maintain $\text{Var}(y) = \text{Var}(x)$:

$$
\sigma_w^2 = \frac{1}{n_{in}}
$$

---

## 🔬 Xavier/Glorot Initialization

### For tanh and sigmoid activations

**Goal:** Maintain variance in both forward and backward passes.

**Derivation:**

Forward: $\text{Var}(y) = n\_{in} \cdot \sigma\_w^2 \cdot \text{Var}(x)$

Backward: $\text{Var}(\delta\_x) = n\_{out} \cdot \sigma\_w^2 \cdot \text{Var}(\delta\_y)$

To satisfy both:

$$
n_{in} \cdot \sigma_w^2 = 1 \quad \text{and} \quad n_{out} \cdot \sigma_w^2 = 1
$$

**Compromise:**

$$
\sigma_w^2 = \frac{2}{n_{in} + n_{out}}
$$

**Uniform Distribution:**

$$
W \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{in} + n_{out}}}, \sqrt{\frac{6}{n_{in} + n_{out}}}\right)
$$

**Normal Distribution:**

$$
W \sim \mathcal{N}\left(0, \frac{2}{n_{in} + n_{out}}\right)
$$

---

## 🔬 Kaiming/He Initialization

### For ReLU activations

**Problem:** ReLU zeros out half the distribution!

$$
\text{ReLU}(x) = \max(0, x)
$$

For $x \sim \mathcal{N}(0, \sigma^2)$:

$$
\text{Var}(\text{ReLU}(x)) = \frac{\sigma^2}{2}
$$

**Proof:**

$$
\mathbb{E}[\text{ReLU}(x)^2] = \int_0^\infty x^2 \cdot \frac{1}{\sqrt{2\pi}\sigma} e^{-x^2/2\sigma^2} dx = \frac{\sigma^2}{2}
\mathbb{E}[\text{ReLU}(x)] = \int_0^\infty x \cdot \frac{1}{\sqrt{2\pi}\sigma} e^{-x^2/2\sigma^2} dx = \frac{\sigma}{\sqrt{2\pi}}
\text{Var}(\text{ReLU}(x)) = \mathbb{E}[X^2] - \mathbb{E}[X]^2 = \frac{\sigma^2}{2} - \frac{\sigma^2}{2\pi} \approx \frac{\sigma^2}{2}
$$

**Solution:** Double the variance to compensate:

$$
\sigma_w^2 = \frac{2}{n_{in}}
$$

**He Normal:**

$$
W \sim \mathcal{N}\left(0, \frac{2}{n_{in}}\right)
$$

**He Uniform:**

$$
W \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{in}}}, \sqrt{\frac{6}{n_{in}}}\right)
$$

---

## 🔬 For Leaky ReLU

Leaky ReLU: $f(x) = \max(\alpha x, x)$ where $\alpha \in (0, 1)$

**Variance through Leaky ReLU:**

$$
\text{Var}(f(x)) = \frac{1 + \alpha^2}{2} \cdot \text{Var}(x)
$$

**Initialization:**

$$
\sigma_w^2 = \frac{2}{(1 + \alpha^2) \cdot n_{in}}
$$

---

## 📊 Initialization Methods Summary

| Method | Formula | Best For |
|--------|---------|----------|
| **Xavier/Glorot** | $\mathcal{N}(0, 2/(n\_{in} + n\_{out}))$ | tanh, sigmoid, GELU |
| **Kaiming/He** | $\mathcal{N}(0, 2/n\_{in})$ | ReLU, LeakyReLU |
| **Orthogonal** | QR decomposition | RNNs, preserves gradient norms |
| **Small Constant** | $\mathcal{N}(0, 0.02)$ | LLMs (GPT-style) |
| **Zero** | $W = 0$ | Biases, certain skip connections |

---

## 🔬 Orthogonal Initialization (RNNs)

**Problem in RNNs:** 

$$
h_t = Wh_{t-1} + Ux_t
$$

After $T$ steps: $h\_T = W^T h\_0 + ...$

If $\|W\| > 1$: exploding gradients  
If $\|W\| < 1$: vanishing gradients

**Solution:** Initialize $W$ as orthogonal matrix.

For orthogonal $W$: $W^\top W = I$

**Properties:**
- All singular values = 1
- Preserves vector norms: $\|Wx\| = \|x\|$
- Gradients neither vanish nor explode

**Algorithm:**
1. Generate random matrix $A$
2. Compute QR decomposition: $A = QR$
3. Use $Q$ as weight matrix

---

## 💻 Implementation

```python
import torch
import torch.nn as nn
import math

# Xavier/Glorot
nn.init.xavier_uniform_(layer.weight)  # Uniform
nn.init.xavier_normal_(layer.weight)   # Normal

# Kaiming/He (for ReLU)
nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')

# For Leaky ReLU
nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='leaky_relu', a=0.01)

# Orthogonal (for RNNs)
nn.init.orthogonal_(layer.weight)

# Small constant (for LLMs like GPT)
nn.init.normal_(layer.weight, mean=0.0, std=0.02)
nn.init.zeros_(layer.bias)

# Custom initialization function
def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=0.02)

model.apply(init_weights)

# GPT-style initialization with scaled residuals
class GPTBlock(nn.Module):
    def __init__(self, d_model, n_layers):
        super().__init__()
        self.attn = nn.Linear(d_model, d_model)
        self.ffn = nn.Linear(d_model, d_model)
        
        # Scale residual outputs by 1/√(2*n_layers)
        # to prevent output variance from growing with depth
        self.scale = 1 / math.sqrt(2 * n_layers)
        
        nn.init.normal_(self.attn.weight, std=0.02)
        nn.init.normal_(self.ffn.weight, std=0.02 * self.scale)
```

---

## 📐 Variance Verification

```python
def check_variance_propagation(model, input_shape):
    """Verify variance is preserved through layers"""
    x = torch.randn(input_shape)
    print(f"Input variance: {x.var().item():.4f}")
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            x = module(x)
            print(f"{name}: variance = {x.var().item():.4f}")
            x = torch.relu(x)  # or other activation
            print(f"  after ReLU: variance = {x.var().item():.4f}")
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | Xavier/Glorot Paper | [AISTATS 2010](http://proceedings.mlr.press/v9/glorot10a.html) |
| 📄 | Kaiming/He Paper | [arXiv](https://arxiv.org/abs/1502.01852) |
| 📄 | Orthogonal Init | [arXiv](https://arxiv.org/abs/1312.6120) |
| 📖 | PyTorch init | [Docs](https://pytorch.org/docs/stable/nn.init.html) |
| 🇨🇳 | 权重初始化详解 | [知乎](https://zhuanlan.zhihu.com/p/25110450) |
| 🇨🇳 | Xavier与Kaiming初始化 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88619088) |

---

## 🔗 Where This Topic Is Used

| Init Method | Best For |
|-------------|---------|
| **Xavier/Glorot** | tanh, sigmoid, GELU |
| **He/Kaiming** | ReLU family |
| **Orthogonal** | RNNs, LSTMs |
| **Small Constant (0.02)** | Transformers, LLMs |
| **Pretrained** | Transfer learning |

---

<p align="center">
  <a href="../">⬆️ Back to Neural Networks</a> &nbsp;|&nbsp;
  <a href="../01_activations/">⬅️ Prev: Activations</a> &nbsp;|&nbsp;
  <a href="../03_layers/">Next: Layers ➡️</a>
</p>

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=80&section=footer" width="100%"/>
</p>
