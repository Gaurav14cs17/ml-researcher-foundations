<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=120&section=header&text=Normalization&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-06-45B7D1?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/normalization.svg" width="100%">

*Caption: Different normalization techniques normalize over different dimensions. BatchNorm normalizes across the batch (good for CNNs), LayerNorm across features (good for Transformers), and RMSNorm is a simpler variant used in modern LLMs.*

---

## 📂 Overview

Normalization layers are critical for stable and fast training. They address internal covariate shift and allow higher learning rates.

---

## 📐 Mathematical Foundations

### General Normalization Form

```math
y = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta

```

Where:

- $\mu$: Mean over normalization dimension

- $\sigma^2$: Variance over normalization dimension

- $\gamma, \beta$: Learned scale and shift parameters

- $\epsilon$: Small constant for numerical stability (e.g., $10^{-5}$)

---

## 🔬 Batch Normalization

### Algorithm

**Training:**
Given mini-batch $B = \{x\_1, ..., x\_m\}$:

```math
\mu_B = \frac{1}{m} \sum_{i=1}^{m} x_i
\sigma_B^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_B)^2
\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
y_i = \gamma \hat{x}_i + \beta

```

**Inference (Running Statistics):**

```math
\mu_{running} \leftarrow (1 - \alpha) \mu_{running} + \alpha \mu_B
\sigma_{running}^2 \leftarrow (1 - \alpha) \sigma_{running}^2 + \alpha \sigma_B^2

```

Where $\alpha$ is the momentum (typically 0.1).

### For Convolutional Layers

Input: $x \in \mathbb{R}^{B \times C \times H \times W}$

Normalize over $(B, H, W)$ for each channel $c$:

```math
\mu_c = \frac{1}{B \cdot H \cdot W} \sum_{b,h,w} x_{b,c,h,w}
\sigma_c^2 = \frac{1}{B \cdot H \cdot W} \sum_{b,h,w} (x_{b,c,h,w} - \mu_c)^2

```

**Parameters:** $\gamma, \beta \in \mathbb{R}^C$ (one per channel)

### Gradient Derivation

**Forward:**

```math
\hat{x} = \frac{x - \mu}{\sigma}, \quad y = \gamma \hat{x} + \beta

```

**Backward:**

```math
\frac{\partial L}{\partial \gamma} = \sum_i \frac{\partial L}{\partial y_i} \cdot \hat{x}_i
\frac{\partial L}{\partial \beta} = \sum_i \frac{\partial L}{\partial y_i}
\frac{\partial L}{\partial \hat{x}_i} = \gamma \frac{\partial L}{\partial y_i}

```

**Key insight:** Gradient of $\hat{x}$ w.r.t. $x$ is complex because $\mu$ and $\sigma$ depend on all $x\_i$:

```math
\frac{\partial L}{\partial x_i} = \frac{1}{m\sigma}\left[m\frac{\partial L}{\partial \hat{x}_i} - \sum_j \frac{\partial L}{\partial \hat{x}_j} - \hat{x}_i \sum_j \frac{\partial L}{\partial \hat{x}_j} \hat{x}_j\right]

```

### Why BatchNorm Works

1. **Reduces Internal Covariate Shift:**
   - Distribution of layer inputs stabilized
   - Easier optimization landscape

2. **Allows Higher Learning Rates:**
   - Normalized activations → bounded gradients
   - Can use 10-100x larger LR

3. **Regularization Effect:**
   - Batch statistics add noise
   - Acts like mild dropout

4. **Smooths Loss Landscape:**
   - Recent theory: main benefit is Lipschitz-smoothing the loss

---

## 🔬 Layer Normalization

### Algorithm

For each sample $x \in \mathbb{R}^d$:

```math
\mu = \frac{1}{d} \sum_{i=1}^{d} x_i
\sigma^2 = \frac{1}{d} \sum_{i=1}^{d} (x_i - \mu)^2
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}
y_i = \gamma_i \hat{x}_i + \beta_i

```

**Key difference from BatchNorm:**
- Normalize over features, not batch

- Same statistics at train and test time

- Works with batch size = 1

### For Transformers

Input: $x \in \mathbb{R}^{B \times L \times D}$

Normalize over $D$ for each token $(b, l)$:

```math
\mu_{b,l} = \frac{1}{D} \sum_{d=1}^{D} x_{b,l,d}
\sigma_{b,l}^2 = \frac{1}{D} \sum_{d=1}^{D} (x_{b,l,d} - \mu_{b,l})^2

```

**Parameters:** $\gamma, \beta \in \mathbb{R}^D$

### BatchNorm vs LayerNorm Comparison

| Aspect | BatchNorm | LayerNorm |
|--------|-----------|-----------|
| Normalizes over | Batch $(N)$ | Features $(D)$ |
| Batch size dependent | Yes | No |
| Same at train/test | No (running stats) | Yes |
| Works with batch=1 | No | Yes |
| Used in | CNNs | Transformers, RNNs |
| Running statistics | Required | Not needed |

### Why LayerNorm for Transformers?

1. **Variable sequence lengths:** Can't compute batch statistics when sequences differ
2. **Batch size = 1:** Common in inference
3. **Consistent behavior:** Same normalization at train and test
4. **Works with attention:** No coupling between sequences

---

## 🔬 RMSNorm (Root Mean Square Normalization)

### Algorithm

Simpler variant without mean subtraction:

```math
\text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2}
y = \frac{x}{\text{RMS}(x)} \cdot \gamma

```

**Note:** No $\beta$ parameter, no mean centering.

### Why RMSNorm?

**Computational Comparison:**

```
LayerNorm operations:
  1. Compute mean: O(d)
  2. Compute variance: O(d)
  3. Subtract mean: O(d)
  4. Divide by std: O(d)
  5. Scale and shift: O(d)
  Total: 5d operations

RMSNorm operations:
  1. Compute squared sum: O(d)
  2. Take sqrt: O(1)
  3. Divide: O(d)
  4. Scale: O(d)
  Total: ~3d operations

Speedup: ~10-15% faster

```

**Theoretical Justification:**

The re-centering (subtracting mean) in LayerNorm can be redundant because:
1. The subsequent linear layer can learn a bias
2. Many activations are already approximately zero-centered

**Used in:** LLaMA, Mistral, Gemma, and most modern LLMs

---

## 🔬 Other Normalization Variants

### Group Normalization

Divides channels into groups, normalizes within each group:

```math
\mu_g = \frac{1}{|G|} \sum_{i \in G} x_i

```

**Use case:** Small batch sizes where BatchNorm fails

### Instance Normalization

Normalizes over $(H, W)$ for each sample and channel:

```math
\mu_{n,c} = \frac{1}{HW} \sum_{h,w} x_{n,c,h,w}

```

**Use case:** Style transfer (removes style information)

### Weight Normalization

Normalizes weight vectors instead of activations:

```math
w = g \cdot \frac{v}{\|v\|}

```

**Use case:** RNNs, when BatchNorm is problematic

---

## 📊 Comparison Table

| Type | Normalizes Over | Statistics | Best For |
|------|-----------------|------------|----------|
| **BatchNorm** | Batch $(B)$ | Running | CNNs, large batches |
| **LayerNorm** | Features $(D)$ | Per-sample | Transformers, RNNs |
| **GroupNorm** | Channel groups | Per-sample | Small batch, detection |
| **InstanceNorm** | $H \times W$ per channel | Per-sample | Style transfer |
| **RMSNorm** | Features $(D)$ | Per-sample | LLMs (faster) |

### Visual Comparison

```
Input tensor: (B, C, H, W)

BatchNorm:    +---------------+
              | ▓▓▓▓▓▓▓▓▓▓▓▓▓ | ← Normalize across batch
              | ▓▓▓▓▓▓▓▓▓▓▓▓▓ |   for each channel
              | ▓▓▓▓▓▓▓▓▓▓▓▓▓ |
              +---------------+
               B samples

LayerNorm:    +---------------+
              | ▓ ▓ ▓ ▓ ▓ ▓ ▓ | ← Normalize across features
              | ▓ ▓ ▓ ▓ ▓ ▓ ▓ |   for each sample
              | ▓ ▓ ▓ ▓ ▓ ▓ ▓ |
              +---------------+
                  D features

```

---

## 💻 Implementation

```python
import torch
import torch.nn as nn
import math

class BatchNorm1d(nn.Module):
    """Manual BatchNorm implementation for understanding"""
    
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        
        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        
        # Running statistics (not parameters)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
    
    def forward(self, x):
        if self.training:
            # Compute batch statistics
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
            
            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var
        
        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta

class LayerNorm(nn.Module):
    """Manual LayerNorm implementation"""
    
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
    
    def forward(self, x):
        # Normalize over last dimension(s)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta

class RMSNorm(nn.Module):
    """RMSNorm - used in LLaMA, Mistral"""
    
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        # RMS = sqrt(mean(x^2))
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight

# Pre-LN vs Post-LN Transformer Block
class PreLNTransformerBlock(nn.Module):
    """Pre-LayerNorm (more stable for deep models, used in GPT-2+)"""
    
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
    
    def forward(self, x):
        # Pre-LN: LayerNorm BEFORE attention/FFN
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        x = x + self.ffn(self.ln2(x))
        return x

class PostLNTransformerBlock(nn.Module):
    """Post-LayerNorm (original Transformer)"""
    
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
    
    def forward(self, x):
        # Post-LN: LayerNorm AFTER residual
        x = self.ln1(x + self.attn(x, x, x)[0])
        x = self.ln2(x + self.ffn(x))
        return x

# PyTorch built-in usage
bn = nn.BatchNorm2d(64)           # For CNNs
ln = nn.LayerNorm(512)            # For Transformers
gn = nn.GroupNorm(8, 64)          # 8 groups of 8 channels
instance_norm = nn.InstanceNorm2d(64)  # For style transfer

```

---

## 📊 Why Normalization Helps: Theoretical Analysis

### 1. Internal Covariate Shift

**Without Normalization:**
Layer inputs have distributions that change during training:

```math
x^{(l)} \sim p_t(x^{(l)})

```

where $p\_t$ depends on all previous layer parameters at step $t$.

**With Normalization:**

```math
\hat{x}^{(l)} \sim \mathcal{N}(\beta, \gamma^2) \quad \text{(approximately)}

```

Distribution controlled by learnable $\beta, \gamma$.

### 2. Gradient Flow

**Theorem (Gradient Magnitude):**
For normalized activations:

```math
\left\|\frac{\partial L}{\partial W^{(l)}}\right\| \approx O\left(\frac{1}{\sqrt{d}}\right) \left\|\frac{\partial L}{\partial a^{(l)}}\right\|

```

Gradients don't explode or vanish as quickly.

### 3. Loss Landscape Smoothing

**Theorem (Santurkar et al., 2018):**
BatchNorm makes the loss landscape more Lipschitz-smooth:

```math
\|\nabla L(w_1) - \nabla L(w_2)\| \leq \beta \|w_1 - w_2\|

```

with smaller $\beta$ than unnormalized networks.

**Implication:** Larger learning rates are stable.

---

## 🔗 Where This Topic Is Used

| Norm Type | Application |
|-----------|-------------|
| **BatchNorm** | CNNs (ResNet, VGG), large batch training |
| **LayerNorm** | Transformers (BERT, GPT), RNNs |
| **RMSNorm** | LLaMA, Mistral, modern LLMs (faster than LayerNorm) |
| **GroupNorm** | Object detection (DETR), small batch training |
| **InstanceNorm** | Style transfer, GANs |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | BatchNorm Paper | [arXiv](https://arxiv.org/abs/1502.03167) |
| 📄 | LayerNorm Paper | [arXiv](https://arxiv.org/abs/1607.06450) |
| 📄 | GroupNorm Paper | [arXiv](https://arxiv.org/abs/1803.08494) |
| 📄 | RMSNorm Paper | [arXiv](https://arxiv.org/abs/1910.07467) |
| 📄 | How Does BN Help? | [NeurIPS 2018](https://arxiv.org/abs/1805.11604) |
| 🇨🇳 | 归一化层详解 | [知乎](https://zhuanlan.zhihu.com/p/33173246) |
| 🇨🇳 | BatchNorm原理与实现 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88401258) |
| 🇨🇳 | 深度学习归一化 | [B站](https://www.bilibili.com/video/BV1Lq4y1k7j6) |

---

➡️ [Next: Optimizers](../02_optimizers/README.md)

---

⬅️ [Back: Training](../../README.md)

---

➡️ [Next: Optimizers](../02_optimizers/README.md)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=80&section=footer" width="100%"/>
</p>
