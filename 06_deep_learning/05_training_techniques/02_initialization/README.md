<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=120&section=header&text=Weight%20Initialization&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-06-45B7D1?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 📂 Overview

Proper weight initialization is crucial for training deep networks. Bad initialization leads to vanishing or exploding gradients, making training impossible. The goal is to maintain activation and gradient variance throughout the network.

---

## 📐 The Core Problem

### Variance Propagation

For a fully connected layer \(y = Wx\) where \(W \in \mathbb{R}^{m \times n}\):

$$y_i = \sum_{j=1}^{n} w_{ij} x_j$$

**Variance of output:**
```
Var[yᵢ] = Var[Σⱼ wᵢⱼ xⱼ]
        = Σⱼ Var[wᵢⱼ xⱼ]         (assuming independence)
        = Σⱼ E[wᵢⱼ²] E[xⱼ²]      (if E[w]=E[x]=0)
        = n · Var[w] · E[x²]
        = n · Var[w] · Var[x]     (if E[x]=0)
```

**Key insight:** If \(\text{Var}[w] = 1/n\), then \(\text{Var}[y] = \text{Var}[x]\)

---

## 🔬 Xavier/Glorot Initialization

### Derivation (Complete Proof)

**Goal:** Keep variance constant in both forward and backward passes.

**Forward pass analysis:**
```
For layer l: y⁽ˡ⁾ = W⁽ˡ⁾ a⁽ˡ⁻¹⁾

Var[y⁽ˡ⁾] = n_in · Var[W] · Var[a⁽ˡ⁻¹⁾]

For Var[y] = Var[a], we need:
Var[W] = 1/n_in
```

**Backward pass analysis:**
```
Gradient: ∂L/∂a⁽ˡ⁻¹⁾ = W⁽ˡ⁾ᵀ · ∂L/∂y⁽ˡ⁾

Var[∂L/∂a⁽ˡ⁻¹⁾] = n_out · Var[W] · Var[∂L/∂y⁽ˡ⁾]

For gradient variance preservation:
Var[W] = 1/n_out
```

**Xavier compromise:**
$$\text{Var}[W] = \frac{2}{n_{in} + n_{out}}$$

**Xavier Distributions:**
```
Uniform: W ~ U[-√(6/(n_in + n_out)), √(6/(n_in + n_out))]
Normal:  W ~ N(0, 2/(n_in + n_out))
```

**Proof for uniform bounds:**
```
For U[-a, a]: Var = (2a)²/12 = a²/3

Setting a²/3 = 2/(n_in + n_out):
a = √(6/(n_in + n_out)) ✓
```

---

## 🔬 He/Kaiming Initialization

### Why ReLU Changes Everything

For ReLU: \(a = \max(0, y)\)

**Effect on variance:**
```
E[a] = E[max(0, y)]

If y ~ N(0, σ²):
E[a] = ∫₀^∞ y · (1/√(2πσ²)) exp(-y²/2σ²) dy
     = σ/√(2π)

E[a²] = ∫₀^∞ y² · (1/√(2πσ²)) exp(-y²/2σ²) dy
      = σ²/2

Var[a] = E[a²] - E[a]²
       = σ²/2 - σ²/(2π)
       ≈ σ²/2  (approximately, since π > 2)
```

**Key insight:** ReLU cuts variance roughly in half!

### Derivation

**For ReLU, forward pass:**
```
Var[a⁽ˡ⁾] ≈ (1/2) · n_in · Var[W] · Var[a⁽ˡ⁻¹⁾]

To preserve variance:
(1/2) · n_in · Var[W] = 1
Var[W] = 2/n_in
```

**He Initialization:**
$$W \sim \mathcal{N}\left(0, \frac{2}{n_{in}}\right)$$

Or for uniform:
$$W \sim U\left[-\sqrt{\frac{6}{n_{in}}}, \sqrt{\frac{6}{n_{in}}}\right]$$

---

## 📊 Comparison of Methods

| Method | Variance | Best For | Formula |
|--------|----------|----------|---------|
| **Xavier Uniform** | \(\frac{2}{n_{in}+n_{out}}\) | tanh, sigmoid | \(U[-\sqrt{6/(n_{in}+n_{out})}, +\sqrt{6/(n_{in}+n_{out})}]\) |
| **Xavier Normal** | \(\frac{2}{n_{in}+n_{out}}\) | tanh, sigmoid | \(N(0, 2/(n_{in}+n_{out}))\) |
| **He Uniform** | \(\frac{2}{n_{in}}\) | ReLU, LeakyReLU | \(U[-\sqrt{6/n_{in}}, +\sqrt{6/n_{in}}]\) |
| **He Normal** | \(\frac{2}{n_{in}}\) | ReLU, LeakyReLU | \(N(0, 2/n_{in})\) |
| **LeCun** | \(\frac{1}{n_{in}}\) | SELU | \(N(0, 1/n_{in})\) |
| **Orthogonal** | - | RNNs | \(W = QR\) decomposition |

---

## 🎯 LeakyReLU Adjustment

For LeakyReLU with slope \(\alpha\):
$$\text{LeakyReLU}(x) = \begin{cases} x & x > 0 \\ \alpha x & x \leq 0 \end{cases}$$

**Variance factor:**
```
E[a²] = (1/2)·E[y²|y>0] + (1/2)·α²·E[y²|y<0]
      = (1/2)·σ² + (1/2)·α²·σ²
      = σ²(1 + α²)/2
```

**Adjusted initialization:**
$$\text{Var}[W] = \frac{2}{(1 + \alpha^2) \cdot n_{in}}$$

---

## 📐 Orthogonal Initialization

### Why Orthogonal Matrices?

**Key property:** Orthogonal matrices preserve norms.

For \(Q^TQ = I\):
```
||Qx||² = xᵀQᵀQx = xᵀx = ||x||²

This means:
- Forward pass: Activations don't explode/vanish
- Backward pass: Gradients don't explode/vanish
```

### Gram-Schmidt Orthogonalization

```python
def orthogonal_init(shape):
    """
    Generate orthogonal matrix via QR decomposition
    """
    # Start with random Gaussian matrix
    a = np.random.randn(*shape)
    
    # QR decomposition
    q, r = np.linalg.qr(a)
    
    # Ensure consistent sign
    d = np.diag(r)
    q *= np.sign(d)
    
    return q
```

---

## 💻 Complete Implementation

### NumPy Implementation

```python
import numpy as np

def xavier_uniform(shape):
    """
    Xavier/Glorot uniform initialization
    U[-sqrt(6/(fan_in+fan_out)), sqrt(6/(fan_in+fan_out))]
    """
    fan_in, fan_out = shape[0], shape[1]
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, shape)

def xavier_normal(shape):
    """
    Xavier/Glorot normal initialization
    N(0, 2/(fan_in+fan_out))
    """
    fan_in, fan_out = shape[0], shape[1]
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return np.random.normal(0, std, shape)

def he_uniform(shape, mode='fan_in'):
    """
    He/Kaiming uniform initialization for ReLU
    U[-sqrt(6/fan), sqrt(6/fan)]
    """
    fan_in, fan_out = shape[0], shape[1]
    fan = fan_in if mode == 'fan_in' else fan_out
    limit = np.sqrt(6.0 / fan)
    return np.random.uniform(-limit, limit, shape)

def he_normal(shape, mode='fan_in'):
    """
    He/Kaiming normal initialization for ReLU
    N(0, 2/fan)
    """
    fan_in, fan_out = shape[0], shape[1]
    fan = fan_in if mode == 'fan_in' else fan_out
    std = np.sqrt(2.0 / fan)
    return np.random.normal(0, std, shape)

def lecun_normal(shape):
    """
    LeCun normal initialization for SELU
    N(0, 1/fan_in)
    """
    fan_in = shape[0]
    std = np.sqrt(1.0 / fan_in)
    return np.random.normal(0, std, shape)

def orthogonal(shape, gain=1.0):
    """
    Orthogonal initialization
    """
    rows, cols = shape
    if rows < cols:
        # Generate more rows, then truncate
        flat_shape = (cols, rows)
    else:
        flat_shape = (rows, cols)
    
    a = np.random.normal(0, 1, flat_shape)
    q, r = np.linalg.qr(a)
    d = np.diag(r)
    q *= np.sign(d)
    
    if rows < cols:
        q = q.T
    
    return gain * q[:rows, :cols]


# Demonstration: Effect of initialization on deep networks
def test_initialization(init_fn, depth=10, width=256, n_samples=100):
    """
    Test how initialization affects activation variance through depth
    """
    x = np.random.randn(n_samples, width)
    
    variances = [x.var()]
    
    for _ in range(depth):
        W = init_fn((width, width))
        x = np.maximum(0, x @ W)  # ReLU
        variances.append(x.var())
    
    return variances

# Compare methods
print("Variance propagation through 10 ReLU layers:")
print(f"{'Layer':<8} {'Xavier':<12} {'He':<12} {'Random':<12}")
print("-" * 44)

np.random.seed(42)
xavier_vars = test_initialization(lambda s: xavier_normal(s))
he_vars = test_initialization(lambda s: he_normal(s))
random_vars = test_initialization(lambda s: np.random.randn(*s) * 0.01)

for i in range(11):
    print(f"{i:<8} {xavier_vars[i]:<12.4f} {he_vars[i]:<12.4f} {random_vars[i]:<12.6f}")
```

### PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.init as init

class ProperlyInitializedMLP(nn.Module):
    """
    MLP with proper initialization for ReLU
    """
    def __init__(self, layer_sizes, activation='relu'):
        super().__init__()
        
        layers = []
        for i in range(len(layer_sizes) - 1):
            linear = nn.Linear(layer_sizes[i], layer_sizes[i+1])
            
            # Proper initialization based on activation
            if activation == 'relu':
                init.kaiming_normal_(linear.weight, mode='fan_in', nonlinearity='relu')
            elif activation == 'tanh':
                init.xavier_normal_(linear.weight)
            elif activation == 'selu':
                init.normal_(linear.weight, std=1/layer_sizes[i]**0.5)
            
            # Bias to zero
            init.zeros_(linear.bias)
            
            layers.append(linear)
            
            # Add activation (except last layer)
            if i < len(layer_sizes) - 2:
                if activation == 'relu':
                    layers.append(nn.ReLU())
                elif activation == 'tanh':
                    layers.append(nn.Tanh())
                elif activation == 'selu':
                    layers.append(nn.SELU())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class InitializedConvNet(nn.Module):
    """
    ConvNet with proper initialization
    """
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        
        self.classifier = nn.Linear(256, num_classes)
        
        # Apply proper initialization
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # He initialization for ReLU
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                init.ones_(m.weight)
                init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, 0, 0.01)
                init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# LSTM initialization
class ProperlyInitializedLSTM(nn.Module):
    """
    LSTM with orthogonal recurrent weights and proper forget gate bias
    """
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                # Input-hidden weights: Xavier
                init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                # Hidden-hidden weights: Orthogonal
                init.orthogonal_(param)
            elif 'bias' in name:
                # Biases: zero, except forget gate
                init.zeros_(param)
                
                # Set forget gate bias to 1
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)
    
    def forward(self, x):
        return self.lstm(x)


# Visualization of initialization effects
def visualize_activations():
    """
    Compare activation distributions with different initializations
    """
    import matplotlib.pyplot as plt
    
    depth = 50
    width = 256
    x = torch.randn(100, width)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for ax, (name, init_fn) in zip(axes, [
        ('He (ReLU)', lambda w: init.kaiming_normal_(w, nonlinearity='relu')),
        ('Xavier', lambda w: init.xavier_normal_(w)),
        ('Small Random', lambda w: init.normal_(w, std=0.01)),
    ]):
        activation_stds = []
        h = x.clone()
        
        for _ in range(depth):
            W = torch.empty(width, width)
            init_fn(W)
            h = torch.relu(h @ W)
            activation_stds.append(h.std().item())
        
        ax.plot(activation_stds)
        ax.set_title(f'{name}')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Activation Std')
        ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('initialization_comparison.png', dpi=150)
    print("Saved initialization_comparison.png")

# Run demonstration
model = ProperlyInitializedMLP([784, 256, 128, 64, 10], activation='relu')
x = torch.randn(32, 784)
y = model(x)
print(f"Output shape: {y.shape}")
print(f"Output mean: {y.mean():.4f}, std: {y.std():.4f}")
```

---

## 🔗 Guidelines Summary

| Activation | Initialization | Variance |
|------------|----------------|----------|
| **tanh, sigmoid** | Xavier/Glorot | \(2/(n_{in}+n_{out})\) |
| **ReLU** | He/Kaiming | \(2/n_{in}\) |
| **LeakyReLU(α)** | Modified He | \(2/((1+α^2)n_{in})\) |
| **SELU** | LeCun | \(1/n_{in}\) |
| **Linear** | Xavier | \(2/(n_{in}+n_{out})\) |
| **LSTM/GRU** | Orthogonal + bias | See above |

---

## 📚 Key Insights

| Insight | Details |
|---------|---------|
| **Bias initialization** | Almost always zero (except LSTM forget gate = 1) |
| **BatchNorm reduces sensitivity** | With BN, initialization matters less |
| **Residual connections** | Scale by \(1/\sqrt{N}\) for N blocks |
| **Transformers** | Often use scaled initialization: \(1/\sqrt{d_{model}}\) |

---

## 📚 References

| Type | Resource | Link |
|------|----------|------|
| 📄 | Xavier Paper | Glorot & Bengio, AISTATS 2010 |
| 📄 | He Paper | He et al., ICCV 2015 |
| 📄 | Orthogonal Init | Saxe et al., ICLR 2014 |
| 📄 | Fixup Initialization | Zhang et al., ICLR 2019 |

---

## 🗺️ Navigation

| ⬅️ Previous | 🏠 Home | ➡️ Next |
|:-----------:|:-------:|:-------:|
| [Gradient Clipping](../01_gradient_clipping/README.md) | [Training Techniques](../README.md) | [Normalization](../03_normalization/README.md) |

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=80&section=footer" width="100%"/>
</p>
