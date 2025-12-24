# Normalization

> **Stabilizing training through feature normalization**

---

## 🎯 Visual Overview

<img src="./images/normalization.svg" width="100%">

*Caption: Different normalization techniques normalize over different dimensions. BatchNorm normalizes across the batch (good for CNNs), LayerNorm across features (good for Transformers), and RMSNorm is a simpler variant used in modern LLMs.*

---

## 📂 Overview

Normalization layers are critical for stable and fast training. They address internal covariate shift and allow higher learning rates.

---

## 📐 Mathematical Definitions

### General Form

```
y = γ · (x - μ) / √(σ² + ε) + β

Where:
    μ: Mean over normalization dimension
    σ²: Variance over normalization dimension
    γ, β: Learned scale and shift parameters
    ε: Small constant for numerical stability (e.g., 1e-5)
```

### BatchNorm

```
Input: x ∈ ℝ^(B×C×H×W)  for images

Normalize over: batch dimension (B) for each channel

μ_c = (1/B·H·W) Σ_{b,h,w} x_{b,c,h,w}
σ²_c = (1/B·H·W) Σ_{b,h,w} (x_{b,c,h,w} - μ_c)²

Output: y_{b,c,h,w} = γ_c · (x - μ_c)/σ_c + β_c

Training: Use batch statistics
Inference: Use running mean/variance
```

### LayerNorm

```
Input: x ∈ ℝ^(B×L×D)  for sequences

Normalize over: feature dimension (D) for each token

μ_{b,l} = (1/D) Σ_d x_{b,l,d}
σ²_{b,l} = (1/D) Σ_d (x_{b,l,d} - μ_{b,l})²

Output: y_{b,l,d} = γ_d · (x - μ)/σ + β_d

Same at training and inference (no running stats!)
```

### RMSNorm

```
Simpler variant (no mean subtraction):

y = x / RMS(x) · γ

Where RMS(x) = √((1/D) Σ_d x_d²)

Benefits:
- 10-15% faster than LayerNorm
- Similar performance
- Used in LLaMA, Mistral, etc.
```

---

## 📊 Comparison

| Type | Normalizes Over | Statistics | Best For |
|------|-----------------|------------|----------|
| **BatchNorm** | Batch (B) | Running | CNNs, large batches |
| **LayerNorm** | Features (D) | Per-sample | Transformers, RNNs |
| **GroupNorm** | Channel groups | Per-sample | Small batch, detection |
| **InstanceNorm** | H×W per channel | Per-sample | Style transfer |
| **RMSNorm** | Features (D) | Per-sample | LLMs (faster) |

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

## 🔑 When to Use What

| Architecture | Recommended | Reason |
|--------------|-------------|--------|
| **CNNs** | BatchNorm | Batch stats work well |
| **Transformers** | LayerNorm | Sequence length varies |
| **RNNs/LSTMs** | LayerNorm | Variable length sequences |
| **GANs** | InstanceNorm or LayerNorm | BatchNorm causes artifacts |
| **LLMs** | RMSNorm | Faster, works well |
| **Small batch** | GroupNorm | BatchNorm unstable |

---

## 💻 Code Examples

### PyTorch Implementations

```python
import torch
import torch.nn as nn

# BatchNorm (for CNNs)
# Input: (batch, channels, height, width)
bn = nn.BatchNorm2d(num_features=64)
x = torch.randn(32, 64, 28, 28)
y = bn(x)  # Same shape

# LayerNorm (for Transformers)
# Input: (batch, sequence, features)
ln = nn.LayerNorm(normalized_shape=512)
x = torch.randn(32, 100, 512)
y = ln(x)  # Same shape

# GroupNorm
gn = nn.GroupNorm(num_groups=8, num_channels=64)
x = torch.randn(32, 64, 28, 28)
y = gn(x)

# InstanceNorm
instance_norm = nn.InstanceNorm2d(num_features=64)
x = torch.randn(32, 64, 28, 28)
y = instance_norm(x)
```

### RMSNorm Implementation

```python
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        # RMS = sqrt(mean(x^2))
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


# Usage
rms_norm = RMSNorm(dim=512)
x = torch.randn(32, 100, 512)
y = rms_norm(x)
```

### Pre-LN vs Post-LN Transformer

```python
class PreLNTransformerBlock(nn.Module):
    """Pre-LayerNorm (more stable for deep models)"""
    
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )
    
    def forward(self, x):
        # Pre-LN: LayerNorm BEFORE attention/FFN
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        x = x + self.ffn(self.ln2(x))
        return x


class PostLNTransformerBlock(nn.Module):
    """Post-LayerNorm (original Transformer)"""
    
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )
    
    def forward(self, x):
        # Post-LN: LayerNorm AFTER residual
        x = self.ln1(x + self.attn(x, x, x)[0])
        x = self.ln2(x + self.ffn(x))
        return x
```

---

## 📊 Why Normalization Helps

```
Without Normalization:
    Layer outputs have widely varying scales
    Gradients can explode/vanish
    Learning rate must be very small
    
With Normalization:
    Outputs have controlled mean/variance
    Gradients are well-behaved
    Can use higher learning rates
    Smoother loss landscape
```

### Internal Covariate Shift

```
During training, distribution of layer inputs changes
as parameters of previous layers update.

Normalization stabilizes input distribution:
    Before: x ~ Distribution(θₜ)  (changes every step!)
    After:  normalized(x) ~ N(β, γ²)  (controlled)
```

---

## 🔗 Connection to Other Topics

```
Normalization
    |
    +-- BatchNorm → CNNs, ResNet
    +-- LayerNorm → Transformers, BERT, GPT
    +-- RMSNorm → LLaMA, Modern LLMs
    +-- GroupNorm → Object Detection (DETR)
```

---

## 🔗 Where This Topic Is Used

| Norm Type | Application |
|-----------|-------------|
| **BatchNorm** | CNNs (ResNet, VGG), large batch training |
| **LayerNorm** | Transformers (BERT, GPT), RNNs |
| **RMSNorm** | LLaMA, modern LLMs (faster than LayerNorm) |
| **GroupNorm** | Object detection (DETR), small batch training |
| **InstanceNorm** | Style transfer, GANs |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | Regularization | [../regularization/](../regularization/) |
| 📖 | Optimizers | [../optimizers/](../optimizers/) |
| 📄 | BatchNorm Paper | [arXiv](https://arxiv.org/abs/1502.03167) |
| 📄 | LayerNorm Paper | [arXiv](https://arxiv.org/abs/1607.06450) |
| 📄 | RMSNorm Paper | [arXiv](https://arxiv.org/abs/1910.07467) |
| 🇨🇳 | 归一化层详解 | [知乎](https://zhuanlan.zhihu.com/p/33173246) |
| 🇨🇳 | BatchNorm原理与实现 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88401258) |
| 🇨🇳 | 深度学习归一化 | [B站](https://www.bilibili.com/video/BV1Lq4y1k7j6) |
| 🇨🇳 | Pre-LN vs Post-LN | [机器之心](https://www.jiqizhixin.com/articles/2020-06-18-6) |
| 🇨🇳 | RMSNorm解读 | [PaperWeekly](https://www.paperweekly.site/)

---

⬅️ [Back: Training](../)

---

➡️ [Next: Optimizers](../optimizers/)
