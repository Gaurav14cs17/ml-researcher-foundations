<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=120&section=header&text=Dropout&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-06-45B7D1?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 📂 Overview

Dropout is a powerful regularization technique that prevents overfitting by randomly "dropping" (zeroing out) neurons during training. It can be interpreted as training an ensemble of $2^n$ subnetworks and approximating Bayesian inference.

---

## 📐 Mathematical Formulation

### Training Phase

For a layer with activations $h \in \mathbb{R}^d$:

**Step 1: Sample binary mask**

$$
m_i \sim \text{Bernoulli}(1-p) \quad \text{for } i = 1, \ldots, d
$$

where $p$ is the dropout probability (typically 0.5 for hidden layers, 0.2 for input).

**Step 2: Apply mask with scaling**

$$
\tilde{h} = \frac{1}{1-p} \cdot (h \odot m)
$$

The scaling factor $\frac{1}{1-p}$ ensures expected value is preserved:
```
E[h̃ᵢ] = E[(1/(1-p)) · hᵢ · mᵢ]
       = (1/(1-p)) · hᵢ · E[mᵢ]
       = (1/(1-p)) · hᵢ · (1-p)
       = hᵢ ✓
```

### Inference Phase

At inference, use all neurons without dropout:

$$
\tilde{h} = h
$$

This is equivalent to computing the expected output over all possible masks.

---

## 🔬 Why Dropout Works

### 1. Ensemble Interpretation

**Key insight:** A network with $n$ droppable units represents $2^n$ different subnetworks.

```
With n=1000 hidden units:
Number of subnetworks = 2^1000 ≈ 10^301

Each training batch uses a different subnetwork.
At test time, we approximate the ensemble average.
```

**Ensemble average approximation:**
```
True ensemble: y = (1/2ⁿ) Σ_{m} f_m(x)
              where f_m is network with mask m

Dropout approx: y ≈ f(x) with all weights scaled by (1-p)
              (or equivalently, scale activations during training)

This is called "weight scaling inference rule"
```

### 2. Bayesian Interpretation

Dropout approximates Bayesian inference over network weights:

**Variational distribution:**

$$
q(W) = \prod_{ij} q(w_{ij}) = \prod_{ij} [(1-p)\delta(w_{ij} - \hat{w}_{ij}) + p\delta(w_{ij})]
$$

**Predictive distribution:**

$$
p(y|x, D) \approx \int p(y|x, W) q(W) dW \approx \frac{1}{T} \sum_{t=1}^{T} f(x; W_t)
$$

where $W_t$ are sampled using dropout masks.

### 3. Noise Injection Perspective

Dropout adds multiplicative noise to activations:
```
h̃ = h · ε  where ε ~ (1/(1-p)) · Bernoulli(1-p)

E[ε] = 1
Var[ε] = p/(1-p)

This noise acts as regularization, similar to:
- Data augmentation
- L2 regularization (for linear models)
- Adds stochasticity to gradients
```

---

## 📊 Mathematical Analysis

### Gradient with Dropout

For a simple linear layer $y = Wx$ with dropout:

**Forward:**

$$
\tilde{x} = \frac{1}{1-p} (x \odot m)
y = W\tilde{x}
$$

**Backward:**
```
∂L/∂W = ∂L/∂y · x̃ᵀ

∂L/∂x = Wᵀ · ∂L/∂y · (m/(1-p))

Key: Gradients only flow through non-dropped units!
```

### Equivalence to L2 Regularization

For linear regression with dropout, the expected loss:

$$
\mathbb{E}[\|y - W\tilde{x}\|^2] = \|y - Wx\|^2 + \frac{p}{1-p}\|W\|_F^2 \cdot \mathbb{E}[\|x\|^2]
$$

**Proof:**
```
E[||y - Wx̃||²] = E[||y - W·(1/(1-p))·(x⊙m)||²]

Let x̃ = x·m/(1-p) where E[m] = (1-p)

E[x̃] = x
Var[x̃] = x²·Var[m/(1-p)] = x²·p/(1-p)

E[||Wx̃||²] = ||Wx||² + ||W||²·Σᵢxᵢ²·p/(1-p)

This shows dropout ≈ L2 regularization with λ ∝ p/(1-p)
```

---

## 📐 Dropout Variants

### 1. Inverted Dropout (Standard)

Scale during training, no scaling at inference:
```python

# Training
mask = (torch.rand(h.shape) > p).float()
h_dropped = h * mask / (1 - p)

# Inference  
h_out = h  # No change
```

### 2. Standard Dropout (Original)

No scaling during training, scale at inference:
```python

# Training
mask = (torch.rand(h.shape) > p).float()
h_dropped = h * mask

# Inference
h_out = h * (1 - p)  # Scale down
```

### 3. DropConnect

Drop individual weights instead of activations:

$$
\tilde{W} = W \odot M \quad \text{where } M_{ij} \sim \text{Bernoulli}(1-p)
$$

### 4. Spatial Dropout (Dropout2D)

For CNNs, drop entire feature maps:
```python

# Shape: (batch, channels, height, width)
mask = (torch.rand(batch, channels, 1, 1) > p).float()
h_dropped = h * mask / (1 - p)
```

### 5. DropPath (Stochastic Depth)

For residual networks, drop entire residual branches:

$$
y = x + \text{drop}(f(x))
$$

### 6. Alpha Dropout

For SELU activations, maintains self-normalizing property:
```python
alpha = 1.6732632423543772848170429916717
scale = 1.0507009873554804934193349852946
alpha_p = -alpha * scale

# Affine transformation to maintain mean/variance
a = (1 - p) ** (-0.5)
b = -a * alpha_p * p
```

---

## 💻 Complete Implementation

### NumPy Implementation

```python
import numpy as np

class Dropout:
    def __init__(self, p=0.5):
        """
        Inverted dropout layer
        
        Args:
            p: probability of dropping a unit (not keeping)
        """
        self.p = p
        self.mask = None
        self.training = True
    
    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        
        # Generate mask: 1 with probability (1-p), 0 with probability p
        self.mask = (np.random.rand(*x.shape) > self.p).astype(np.float32)
        
        # Apply mask and scale
        return x * self.mask / (1 - self.p)
    
    def backward(self, dout):
        """
        Gradient flows only through non-dropped units
        """
        if not self.training or self.p == 0:
            return dout
        
        return dout * self.mask / (1 - self.p)

class Dropout2D:
    """
    Spatial dropout for CNNs - drops entire channels
    """
    def __init__(self, p=0.5):
        self.p = p
        self.mask = None
        self.training = True
    
    def forward(self, x):
        """
        Args:
            x: (N, C, H, W)
        """
        if not self.training or self.p == 0:
            return x
        
        N, C, H, W = x.shape
        
        # Mask shape: (N, C, 1, 1) - same for all spatial positions
        self.mask = (np.random.rand(N, C, 1, 1) > self.p).astype(np.float32)
        
        return x * self.mask / (1 - self.p)
    
    def backward(self, dout):
        if not self.training or self.p == 0:
            return dout
        return dout * self.mask / (1 - self.p)

# Monte Carlo Dropout for uncertainty estimation
class MCDropoutPredictor:
    """
    Use dropout at test time for uncertainty estimation
    """
    def __init__(self, model, n_samples=100):
        self.model = model
        self.n_samples = n_samples
    
    def predict_with_uncertainty(self, x):
        """
        Returns mean prediction and uncertainty (std)
        """
        self.model.train()  # Keep dropout active
        
        predictions = []
        for _ in range(self.n_samples):
            with torch.no_grad():
                pred = self.model(x)
            predictions.append(pred)
        
        predictions = torch.stack(predictions)
        
        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)  # Epistemic uncertainty
        
        return mean, std

# Demonstration
def test_dropout_statistics():
    """Verify dropout preserves expected value"""
    np.random.seed(42)
    
    dropout = Dropout(p=0.5)
    x = np.random.randn(1000, 256)
    
    # Multiple forward passes
    outputs = []
    for _ in range(100):
        out = dropout.forward(x.copy())
        outputs.append(out)
    
    outputs = np.stack(outputs)
    
    print(f"Input mean: {x.mean():.4f}")
    print(f"Output mean (avg over 100 samples): {outputs.mean():.4f}")
    print(f"Input std: {x.std():.4f}")
    print(f"Output std: {outputs.mean(axis=0).std():.4f}")

test_dropout_statistics()
```

### PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DropoutCustom(nn.Module):
    """
    Custom dropout implementation
    """
    def __init__(self, p=0.5, inplace=False):
        super().__init__()
        self.p = p
        self.inplace = inplace
    
    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        
        # Bernoulli mask
        mask = torch.bernoulli(torch.full_like(x, 1 - self.p))
        
        # Scale by 1/(1-p)
        if self.inplace:
            x.mul_(mask).div_(1 - self.p)
            return x
        else:
            return x * mask / (1 - self.p)

class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) for residual blocks
    Used in Vision Transformers
    """
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob

        # Shape: (batch_size, 1, 1, ..., 1)
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        
        return x.div(keep_prob) * random_tensor

class MLPWithDropout(nn.Module):
    """
    MLP with dropout between layers
    """
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_p=0.5):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_p)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class ConvNetWithDropout(nn.Module):
    """
    CNN with spatial dropout
    """
    def __init__(self, in_channels, num_classes, dropout_p=0.2):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(dropout_p),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(dropout_p),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(dropout_p),
            nn.AdaptiveAvgPool2d(1),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# MC Dropout for Uncertainty
class MCDropoutModel(nn.Module):
    """
    Model with MC Dropout for uncertainty estimation
    """
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
    
    def predict_with_uncertainty(self, x, n_samples=50):
        """
        Perform multiple forward passes with dropout enabled
        
        Returns:
            mean: Mean prediction
            var: Predictive variance (epistemic uncertainty)
        """
        self.train()  # Enable dropout
        
        predictions = []
        for _ in range(n_samples):
            with torch.no_grad():
                pred = self.base_model(x)
                predictions.append(F.softmax(pred, dim=-1))
        
        predictions = torch.stack(predictions)  # (n_samples, batch, classes)
        
        mean = predictions.mean(dim=0)
        var = predictions.var(dim=0)  # Epistemic uncertainty
        
        return mean, var
    
    def entropy_uncertainty(self, x, n_samples=50):
        """
        Compute predictive entropy as uncertainty measure
        """
        mean, var = self.predict_with_uncertainty(x, n_samples)
        
        # Predictive entropy
        entropy = -torch.sum(mean * torch.log(mean + 1e-10), dim=-1)
        
        return mean, entropy

# Example usage
model = MLPWithDropout(784, [512, 256], 10, dropout_p=0.5)
x = torch.randn(32, 784)

# Training mode
model.train()
y_train = model(x)
print(f"Training output shape: {y_train.shape}")

# Eval mode (no dropout)
model.eval()
y_eval = model(x)
print(f"Eval output shape: {y_eval.shape}")

# MC Dropout
mc_model = MCDropoutModel(model)
mean, uncertainty = mc_model.predict_with_uncertainty(x, n_samples=30)
print(f"Mean prediction shape: {mean.shape}")
print(f"Uncertainty shape: {uncertainty.shape}")
```

---

## 🎯 Best Practices

| Scenario | Recommendation |
|----------|----------------|
| **Fully connected layers** | p = 0.5 |
| **Input layer** | p = 0.2 (preserve input info) |
| **Convolutional layers** | p = 0.1-0.3 or Dropout2D |
| **After BatchNorm** | Often not needed |
| **Transformers** | Use DropPath (0.1-0.3) |
| **Small datasets** | Higher p (more regularization) |
| **Large models** | Essential for preventing overfitting |

---

## 📚 Key Insights

| Insight | Details |
|---------|---------|
| **Inverted is standard** | Scale during training, not inference |
| **Not with BatchNorm** | BN provides regularization; dropout may hurt |
| **Spatial for CNNs** | Drop entire channels, not individual pixels |
| **MC Dropout** | Keep dropout at test time for uncertainty |
| **Scheduled dropout** | Start with p=0, increase during training |

---

## 📚 References

| Type | Resource | Link |
|------|----------|------|
| 📄 | Original Paper | Srivastava et al., JMLR 2014 |
| 📄 | Dropout as Bayesian | Gal & Ghahramani, 2016 |
| 📄 | DropConnect | Wan et al., ICML 2013 |
| 📄 | Stochastic Depth | Huang et al., ECCV 2016 |

---

## 🗺️ Navigation

| ⬅️ Previous | 🏠 Home | ➡️ Next |
|:-----------:|:-------:|:-------:|
| [Regularization](../README.md) | [Training Techniques](../../README.md) | [Weight Decay](../02_weight_decay/README.md) |

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=80&section=footer" width="100%"/>
</p>
