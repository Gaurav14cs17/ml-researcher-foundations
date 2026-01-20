<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=120&section=header&text=Batch%20Normalization&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-06-45B7D1?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 📂 Overview

Batch Normalization (BatchNorm) is one of the most important innovations in deep learning, enabling training of very deep networks. It normalizes activations within each mini-batch, reducing internal covariate shift and allowing higher learning rates.

---

## 📐 Mathematical Formulation

### Forward Pass

For a mini-batch $\mathcal{B} = \{x_1, \ldots, x_m\}$:

**Step 1: Compute batch statistics**

$$
\mu_{\mathcal{B}} = \frac{1}{m} \sum_{i=1}^{m} x_i
\sigma^2_{\mathcal{B}} = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_{\mathcal{B}})^2
$$

**Step 2: Normalize**

$$
\hat{x}_i = \frac{x_i - \mu_{\mathcal{B}}}{\sqrt{\sigma^2_{\mathcal{B}} + \epsilon}}
$$

**Step 3: Scale and shift (learnable)**

$$
y_i = \gamma \hat{x}_i + \beta
$$

where:
- $\gamma$: learnable scale parameter (initialized to 1)
- $\beta$: learnable shift parameter (initialized to 0)
- $\epsilon$: small constant for numerical stability (typically $10^{-5}$)

---

## 📊 Why BatchNorm Works

### 1. Internal Covariate Shift (Original Hypothesis)

**Problem:** During training, the distribution of each layer's inputs changes as parameters in previous layers update.

```
Without BatchNorm:
Layer 2 sees inputs with changing statistics
→ Must constantly adapt to new distributions
→ Slower training

With BatchNorm:
Layer 2 always sees normalized inputs (μ≈0, σ≈1)
→ Stable learning dynamics
→ Faster convergence
```

### 2. Smoothing the Loss Landscape (Modern Understanding)

**Recent research suggests BatchNorm works by:**
- Making the loss landscape smoother (smaller Lipschitz constant)
- Reducing gradient variance
- Allowing larger learning rates without divergence

```
Mathematical insight:
Without BatchNorm: ∇L has high variance across batches
With BatchNorm: ∇L is more consistent
→ Optimization becomes easier
```

### 3. Regularization Effect

BatchNorm adds noise during training (batch statistics vary):
```
Training: Use batch statistics (noisy)
- μ_B varies between batches
- Acts like data augmentation

Inference: Use running averages (stable)
- μ_running = α · μ_running + (1-α) · μ_B
- σ²_running = α · σ²_running + (1-α) · σ²_B
```

---

## 🔬 Backward Pass Derivation

### Gradient Computation (Complete Proof)

Given $\frac{\partial L}{\partial y_i}$, compute gradients with respect to $\gamma$, $\beta$, and $x_i$.

**Gradient with respect to $\gamma$:**

$$
\frac{\partial L}{\partial \gamma} = \sum_{i=1}^{m} \frac{\partial L}{\partial y_i} \cdot \hat{x}_i
$$

**Gradient with respect to $\beta$:**

$$
\frac{\partial L}{\partial \beta} = \sum_{i=1}^{m} \frac{\partial L}{\partial y_i}
$$

**Gradient with respect to $\hat{x}_i$:**

$$
\frac{\partial L}{\partial \hat{x}_i} = \frac{\partial L}{\partial y_i} \cdot \gamma
$$

**Gradient with respect to $\sigma^2_{\mathcal{B}}$:**
```
∂L/∂σ²_B = Σᵢ ∂L/∂x̂ᵢ · (xᵢ - μ_B) · (-1/2)(σ²_B + ε)^(-3/2)
         = -1/2 · (σ²_B + ε)^(-3/2) · Σᵢ ∂L/∂x̂ᵢ · (xᵢ - μ_B)
```

**Gradient with respect to $\mu_{\mathcal{B}}$:**
```
∂L/∂μ_B = Σᵢ ∂L/∂x̂ᵢ · (-1/√(σ²_B + ε))
        + ∂L/∂σ²_B · (-2/m) Σᵢ (xᵢ - μ_B)
        = -1/√(σ²_B + ε) · Σᵢ ∂L/∂x̂ᵢ + 0
        (second term is 0 because Σᵢ(xᵢ - μ_B) = 0)
```

**Gradient with respect to $x_i$:**
```
∂L/∂xᵢ = ∂L/∂x̂ᵢ · 1/√(σ²_B + ε)
       + ∂L/∂σ²_B · 2(xᵢ - μ_B)/m
       + ∂L/∂μ_B · 1/m

Simplified form:
∂L/∂xᵢ = (1/m) · (1/√(σ²_B + ε)) · [m · ∂L/∂x̂ᵢ - Σⱼ ∂L/∂x̂ⱼ - x̂ᵢ · Σⱼ ∂L/∂x̂ⱼ · x̂ⱼ]
```

---

## 📊 Variants of Normalization

### Comparison

| Method | Normalizes Over | Use Case |
|--------|-----------------|----------|
| **Batch Norm** | Batch dimension | CNNs (large batches) |
| **Layer Norm** | Feature dimension | RNNs, Transformers |
| **Instance Norm** | H×W per channel | Style transfer |
| **Group Norm** | Groups of channels | Small batches |

### Mathematical Comparison

For input tensor $x \in \mathbb{R}^{N \times C \times H \times W}$:

```
Batch Norm: Normalize over (N, H, W) for each C
→ Statistics: μ, σ² ∈ ℝᶜ

Layer Norm: Normalize over (C, H, W) for each N
→ Statistics: μ, σ² ∈ ℝᴺ

Instance Norm: Normalize over (H, W) for each (N, C)
→ Statistics: μ, σ² ∈ ℝᴺˣᶜ

Group Norm: Normalize over (C/G, H, W) for G groups
→ Statistics: μ, σ² ∈ ℝᴺˣᴳ
```

---

## 💻 Complete Implementation

### NumPy Implementation

```python
import numpy as np

class BatchNorm:
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Learnable parameters
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        
        # Running statistics (for inference)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        
        # Cache for backward pass
        self.cache = None
        
        self.training = True
    
    def forward(self, x):
        """
        Args:
            x: (N, C) or (N, C, H, W)
        """
        if x.ndim == 4:

            # For CNNs: (N, C, H, W) -> (N*H*W, C)
            N, C, H, W = x.shape
            x_flat = x.transpose(0, 2, 3, 1).reshape(-1, C)
            out_flat = self._forward_1d(x_flat)
            return out_flat.reshape(N, H, W, C).transpose(0, 3, 1, 2)
        else:
            return self._forward_1d(x)
    
    def _forward_1d(self, x):
        """
        Forward pass for 2D input (N, C)
        """
        if self.training:

            # Compute batch statistics
            mu = x.mean(axis=0)
            var = x.var(axis=0)
            
            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:

            # Use running statistics
            mu = self.running_mean
            var = self.running_var
        
        # Normalize
        x_centered = x - mu
        std = np.sqrt(var + self.eps)
        x_norm = x_centered / std
        
        # Scale and shift
        out = self.gamma * x_norm + self.beta
        
        # Cache for backward
        if self.training:
            self.cache = (x, x_norm, x_centered, std, self.gamma)
        
        return out
    
    def backward(self, dout):
        """
        Backward pass
        
        Args:
            dout: gradient of loss w.r.t. output
        
        Returns:
            dx: gradient w.r.t. input
            dgamma: gradient w.r.t. gamma
            dbeta: gradient w.r.t. beta
        """
        x, x_norm, x_centered, std, gamma = self.cache
        N = x.shape[0]
        
        # Gradients for gamma and beta
        dbeta = dout.sum(axis=0)
        dgamma = (dout * x_norm).sum(axis=0)
        
        # Gradient for normalized x
        dx_norm = dout * gamma
        
        # Gradient for variance
        dvar = np.sum(dx_norm * x_centered * -0.5 * std**(-3), axis=0)
        
        # Gradient for mean
        dmean = np.sum(dx_norm * -1/std, axis=0) + dvar * np.mean(-2 * x_centered, axis=0)
        
        # Gradient for input
        dx = dx_norm / std + dvar * 2 * x_centered / N + dmean / N
        
        return dx, dgamma, dbeta

# Test
bn = BatchNorm(64)
x = np.random.randn(32, 64)  # (batch, features)

# Training mode
out = bn.forward(x)
print(f"Output mean: {out.mean(axis=0)[:5]}")  # Should be close to beta (0)
print(f"Output var: {out.var(axis=0)[:5]}")    # Should be close to gamma² (1)

# Backward
dout = np.random.randn(*out.shape)
dx, dgamma, dbeta = bn.backward(dout)
print(f"Gradient shapes: dx={dx.shape}, dgamma={dgamma.shape}, dbeta={dbeta.shape}")
```

### PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BatchNormCustom(nn.Module):
    """
    Custom BatchNorm implementation
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        
        # Running statistics (not parameters)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
    
    def forward(self, x):
        """
        Args:
            x: (N, C, ...) - can be 2D (N, C) or 4D (N, C, H, W)
        """
        if self.training:

            # Compute batch statistics
            if x.dim() == 2:
                mean = x.mean(dim=0)
                var = x.var(dim=0, unbiased=False)
            else:

                # For (N, C, H, W): compute over N, H, W
                mean = x.mean(dim=(0, 2, 3))
                var = x.var(dim=(0, 2, 3), unbiased=False)
            
            # Update running statistics
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var
        
        # Reshape for broadcasting
        if x.dim() == 4:
            mean = mean.view(1, -1, 1, 1)
            var = var.view(1, -1, 1, 1)
            gamma = self.gamma.view(1, -1, 1, 1)
            beta = self.beta.view(1, -1, 1, 1)
        else:
            gamma = self.gamma
            beta = self.beta
        
        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # Scale and shift
        return gamma * x_norm + beta

class ConvBNReLU(nn.Module):
    """
    Common pattern: Conv -> BatchNorm -> ReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class ResidualBlockWithBN(nn.Module):
    """
    ResNet-style block with BatchNorm
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity  # Skip connection
        return F.relu(out)

# Example: Compare with/without BatchNorm
def test_batchnorm_effect():
    torch.manual_seed(42)
    
    # Network without BatchNorm
    net_no_bn = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    
    # Network with BatchNorm
    net_with_bn = nn.Sequential(
        nn.Linear(784, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    
    # Random input
    x = torch.randn(32, 784)
    
    # Check activation statistics
    print("Without BatchNorm:")
    out = x
    for i, layer in enumerate(net_no_bn):
        out = layer(out)
        if isinstance(layer, nn.ReLU):
            print(f"  After layer {i}: mean={out.mean():.4f}, std={out.std():.4f}")
    
    print("\nWith BatchNorm:")
    out = x
    for i, layer in enumerate(net_with_bn):
        out = layer(out)
        if isinstance(layer, nn.ReLU):
            print(f"  After layer {i}: mean={out.mean():.4f}, std={out.std():.4f}")

test_batchnorm_effect()
```

### Fused BatchNorm for Inference

```python
def fuse_conv_bn(conv, bn):
    """
    Fuse Conv2d and BatchNorm2d for faster inference
    
    Conv: y = W*x + b
    BN: z = γ((y - μ)/σ) + β = γ/σ * y + (β - γμ/σ)
    
    Fused: z = W'*x + b'
    where W' = γ/σ * W, b' = γ/σ * (b - μ) + β
    """
    assert conv.bias is None, "Conv should have no bias before fusion"
    
    # Get BN parameters
    gamma = bn.weight
    beta = bn.bias
    mean = bn.running_mean
    var = bn.running_var
    eps = bn.eps
    
    std = torch.sqrt(var + eps)
    
    # Fused weights
    fused_conv = nn.Conv2d(
        conv.in_channels, conv.out_channels,
        conv.kernel_size, conv.stride, conv.padding,
        bias=True
    )
    
    # W' = γ/σ * W
    fused_conv.weight.data = conv.weight * (gamma / std).view(-1, 1, 1, 1)
    
    # b' = β - γμ/σ
    fused_conv.bias.data = beta - gamma * mean / std
    
    return fused_conv

# Example usage
conv = nn.Conv2d(64, 128, 3, padding=1, bias=False)
bn = nn.BatchNorm2d(128)

# Run some data to populate running stats
x = torch.randn(4, 64, 32, 32)
_ = bn(conv(x))

# Fuse
fused = fuse_conv_bn(conv, bn)

# Compare outputs
bn.eval()
x_test = torch.randn(1, 64, 32, 32)
out_original = bn(conv(x_test))
out_fused = fused(x_test)
print(f"Max difference: {(out_original - out_fused).abs().max():.2e}")
```

---

## 🎯 Best Practices

### When to Use

| Scenario | Recommendation |
|----------|----------------|
| CNNs with large batches | ✅ BatchNorm |
| Small batches (< 8) | ❌ Use GroupNorm |
| RNNs/Transformers | ❌ Use LayerNorm |
| Style Transfer | ❌ Use InstanceNorm |
| GANs Discriminator | ❌ Often avoided |

### Common Pitfalls

```python

# ❌ Wrong: Using batch stats at inference
model.train()  # Always in training mode
output = model(x)  # Batch stats vary!

# ✅ Correct: Switch to eval mode
model.eval()  # Use running statistics
with torch.no_grad():
    output = model(x)

# ❌ Wrong: BatchNorm before activation in some cases
x = relu(bn(conv(x)))  # Standard (usually fine)

# ✅ Consider: For residual connections
x = bn(relu(conv(x)))  # Sometimes better for gradients
```

---

## 📚 Key Insights

| Insight | Details |
|---------|---------|
| **No bias in Conv** | BN absorbs bias; set `bias=False` |
| **Initialize γ=1, β=0** | Network starts with identity transform |
| **Momentum ≈ 0.1** | Running stats update: `0.9 * old + 0.1 * new` |
| **Inference fusion** | Fold BN into Conv for 10-30% speedup |

---

## 📚 References

| Type | Resource | Link |
|------|----------|------|
| 📄 | Original Paper | Ioffe & Szegedy, 2015 |
| 📄 | How Does BN Help? | Santurkar et al., 2018 |
| 📄 | Group Normalization | Wu & He, 2018 |
| 🎥 | Andrew Ng Explanation | [YouTube](https://www.youtube.com/watch?v=tNIpEZLv_eg) |

---

## 🗺️ Navigation

| ⬅️ Previous | 🏠 Home | ➡️ Next |
|:-----------:|:-------:|:-------:|
| [Normalization](../README.md) | [Training Techniques](../../README.md) | [LayerNorm](../02_layer_norm/README.md) |

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=80&section=footer" width="100%"/>
</p>
