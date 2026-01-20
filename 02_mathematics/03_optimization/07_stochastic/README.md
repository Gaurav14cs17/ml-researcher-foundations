<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=180&section=header&text=Stochastic%20Optimization&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=32&desc=SGD%20·%20Momentum%20·%20Adam%20·%20AdamW&descAlignY=52&descSize=16" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/📚_Section-03.07_Stochastic-00C853?style=for-the-badge" alt="Section"/>
  <img src="https://img.shields.io/badge/📊_Topics-SGD_Adam_Momentum-blue?style=for-the-badge" alt="Topics"/>
  <img src="https://img.shields.io/badge/✍️_Author-Gaurav_Goswami-purple?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/📅_Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## ⚡ TL;DR

> **SGD and its variants power ALL of deep learning.** The noise from mini-batches provides implicit regularization and enables training on massive datasets.

- 🎲 **SGD**: Use random mini-batches for unbiased gradient estimates
- 🚀 **Momentum**: Accelerate along consistent gradient directions
- 🧠 **Adam**: Adaptive learning rates per parameter
- 🏆 **AdamW**: Adam + proper weight decay (best for transformers)

---

## 📑 Table of Contents

1. [SGD Fundamentals](#1-sgd-fundamentals)
2. [Momentum](#2-momentum)
3. [Adam Optimizer](#3-adam-optimizer)
4. [AdamW](#4-adamw)
5. [Convergence Theory](#5-convergence-theory)
6. [Learning Rate Schedules](#6-learning-rate-schedules)
7. [Code Implementation](#7-code-implementation)
8. [Which Optimizer to Use](#8-which-optimizer-to-use)
9. [Resources](#-resources)

---

## 🎨 Visual Overview

<img src="./images/stochastic-opt.svg" width="100%">

```
+-----------------------------------------------------------------------------+
|                    OPTIMIZER COMPARISON                                      |
+-----------------------------------------------------------------------------+
|                                                                              |
|   SGD              SGD+Momentum           Adam                               |
|   ---              ------------           ----                               |
|                                                                              |
|   θ ← θ - η∇L     v ← βv + ∇L           m ← β₁m + (1-β₁)∇L                 |
|                    θ ← θ - ηv            v ← β₂v + (1-β₂)(∇L)²              |
|                                          θ ← θ - η·m̂/√v̂                     |
|                                                                              |
|   [Noisy path]    [Smoother path]       [Adaptive per-param]                |
|                                                                              |
|   +===================================================================+     |
|   |  KEY INSIGHT: SGD noise = implicit regularization                  |     |
|   |  Larger batch = less noise = often worse generalization           |     |
|   +===================================================================+     |
|                                                                              |
|   WHEN TO USE WHAT:                                                          |
|   -----------------                                                          |
|   • CNNs (ResNet, etc): SGD + Momentum (better generalization)              |
|   • Transformers (BERT, GPT): AdamW                                         |
|   • Quick experiments: Adam                                                  |
|   • Fine-tuning: Lower lr + AdamW                                           |
|                                                                              |
+-----------------------------------------------------------------------------+
```

---

## 1. SGD Fundamentals

### 📌 Algorithm

**Full-Batch Gradient Descent**:

```math
\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t) = \theta_t - \eta \frac{1}{n}\sum_{i=1}^{n} \nabla \ell_i(\theta_t)
```

**Stochastic Gradient Descent**:

```math
\theta_{t+1} = \theta_t - \eta \nabla \ell_{i_t}(\theta_t)
```

where $i\_t$ is randomly sampled.

**Mini-Batch SGD**:

```math
\theta_{t+1} = \theta_t - \eta \frac{1}{|B|}\sum_{i \in B} \nabla \ell_i(\theta_t)
```

### 📐 Key Properties

| Property | Formula/Value | Significance |
|----------|---------------|--------------|
| Unbiased | $\mathbb{E}[\nabla \ell\_i] = \nabla L$ | Correct on average |
| Variance | $\text{Var} \propto 1/|B|$ | Larger batch = less noise |
| Per-step cost | $O(|B| \cdot d)$ | Linear in batch size |

### 🔍 Why SGD Works

```
1. COMPUTATIONAL EFFICIENCY:
   Full GD: O(n) per step
   SGD: O(batch_size) per step
   For ImageNet (n=1.2M), batch=256 → 4700× faster per step!

2. IMPLICIT REGULARIZATION:
   SGD noise acts like regularization
   Larger batch → less noise → often worse test accuracy
   "Sharp minima" are unstable under SGD → finds flatter minima

3. ESCAPING SADDLES:
   Noise helps escape saddle points and bad local minima
   Pure GD can get stuck in saddles (gradient = 0)
```

### 💡 Example: Variance Reduction

```
Consider L(θ) = (1/n)Σᵢ ℓᵢ(θ)

Full gradient variance: 0 (deterministic)

Single-sample variance: σ² = E[‖∇ℓᵢ - ∇L‖²]

Mini-batch variance: σ²/|B|

Example:
  σ² = 100, batch_size = 64
  Variance = 100/64 ≈ 1.56

  Increasing batch to 256:
  Variance = 100/256 ≈ 0.39 (4× reduction)
```

---

## 2. Momentum

### 📌 Algorithm

```math
\begin{align}
v_{t+1} &= \beta v_t + \nabla L(\theta_t) \\
\theta_{t+1} &= \theta_t - \eta v_{t+1}
\end{align}
```

Or equivalently (with damping factor):

```math
\begin{align}
v_{t+1} &= \beta v_t + (1-\beta) \nabla L(\theta_t) \\
\theta_{t+1} &= \theta_t - \eta v_{t+1}
\end{align}
```

### 📐 Intuition

```
PHYSICAL ANALOGY:
  θ = position of a ball
  v = velocity
  ∇L = force (gravity pulling toward minimum)
  β = friction coefficient

The ball:
1. Accelerates along consistent gradient directions
2. Dampens oscillations in inconsistent directions
3. Builds up speed going downhill

WHY IT HELPS:
  +--------------------------------------------+
  |                                            |
  |   Without Momentum:     With Momentum:     |
  |   ~~~~~~~~~~~~~~~~      ~~~~~~~~~~~~~~     |
  |   ↓ ↗ ↓ ↗ ↓ ↗ ↓        ---------→        |
  |   (oscillates)          (smooth path)      |
  |                                            |
  +--------------------------------------------+
```

### 📐 Nesterov Momentum

```
"Look ahead" before computing gradient:

v_{t+1} = β v_t + ∇L(θ_t - η·β·v_t)  # Gradient at "lookahead" position
θ_{t+1} = θ_t - η v_{t+1}

Intuition: Evaluate gradient where we're going, not where we are
Often slightly better than standard momentum
```

---

## 3. Adam Optimizer

### 📌 Algorithm

```math
\begin{align}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) \nabla L(\theta_t) \quad &\text{(First moment estimate)} \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2) (\nabla L(\theta_t))^2 \quad &\text{(Second moment estimate)} \\
\hat{m}_t &= m_t / (1 - \beta_1^t) \quad &\text{(Bias correction)} \\
\hat{v}_t &= v_t / (1 - \beta_2^t) \quad &\text{(Bias correction)} \\
\theta_{t+1} &= \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
\end{align}
```

### 📐 Default Hyperparameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| $\eta$ (lr) | 0.001 | Learning rate |
| $\beta\_1$ | 0.9 | First moment decay (momentum) |
| $\beta\_2$ | 0.999 | Second moment decay (RMSprop-like) |
| $\epsilon$ | 1e-8 | Numerical stability |

### 🔍 Why Bias Correction?

```
Problem: m₀ = v₀ = 0 biases early estimates toward 0

Without correction at t=1:
  m₁ = β₁·0 + (1-β₁)g₁ = (1-β₁)g₁ = 0.1·g₁  (too small!)

With correction:
  m̂₁ = m₁/(1-β₁¹) = 0.1g₁/0.1 = g₁  ✓

As t → ∞: 1 - β^t → 1, so correction vanishes
```

### 📐 Understanding Adam

```
Adam ≈ Momentum + RMSprop + Bias Correction

MOMENTUM COMPONENT (m):
  Smooths gradients, reduces noise
  Like SGD momentum

RMSPROP COMPONENT (v):
  Adapts learning rate per-parameter
  Parameters with large gradients → smaller effective lr
  Parameters with small gradients → larger effective lr

ADAPTIVE LEARNING RATE:
  Effective lr for parameter i ≈ η / √(avg squared gradient)
  
  Large avg gradient → small step (careful in "steep" directions)
  Small avg gradient → large step (faster in "flat" directions)
```

---

## 4. AdamW

### 📌 The Problem with Adam + L2

```
Standard L2 regularization:
  L_reg(θ) = L(θ) + (λ/2)‖θ‖²

With Adam:
  ∇L_reg = ∇L + λθ

This gets divided by √v̂, effectively making regularization
adaptive too - NOT what we want!

Weight decay should be constant, not adaptive.
```

### 📌 AdamW Algorithm

```math
\theta_{t+1} = \theta_t - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t \right)
```

Decoupled weight decay: subtract $\eta \lambda \theta$ directly, bypassing Adam's adaptive scaling.

### 📐 Adam vs AdamW

```
Adam + L2:
  g = ∇L + λθ          # Regularization in gradient
  update = m̂ / √v̂      # Both L and λθ get scaled by √v̂

AdamW:
  g = ∇L               # Only task loss
  update = m̂ / √v̂ + λθ  # Weight decay is NOT scaled

AdamW is what you actually want for regularization!
```

---

## 5. Convergence Theory

### 📐 Convergence Rates

**Convex Functions**:

```math
\mathbb{E}[L(\theta_T) - L(\theta^*)] = O\left(\frac{1}{\sqrt{T}}\right)
```

**Strongly Convex Functions**:

```math
\mathbb{E}[\|\theta_T - \theta^*\|^2] = O\left(\frac{1}{T}\right)
```

**Non-Convex Functions** (finding stationary point):

```math
\min_{t \leq T} \mathbb{E}[\|\nabla L(\theta_t)\|^2] = O\left(\frac{1}{\sqrt{T}}\right)
```

### 📐 Learning Rate Decay

For convergence guarantees, need:

```math
\sum_{t=1}^{\infty} \eta_t = \infty \quad \text{and} \quad \sum_{t=1}^{\infty} \eta_t^2 < \infty
```

Examples:
- $\eta\_t = \eta\_0 / \sqrt{t}$ ✓
- $\eta\_t = \eta\_0 / t$ ✓
- $\eta\_t = \eta\_0$ (constant) ✗ (doesn't converge exactly)

---

## 6. Learning Rate Schedules

### 📊 Common Schedules

| Schedule | Formula | When to Use |
|----------|---------|-------------|
| Constant | $\eta\_t = \eta\_0$ | Quick experiments |
| Step decay | $\eta\_t = \eta\_0 \cdot \gamma^{\lfloor t/s \rfloor}$ | CNNs (ResNet) |
| Cosine | $\eta\_t = \eta\_{\min} + \frac{1}{2}(\eta\_0 - \eta\_{\min})(1 + \cos(\frac{t\pi}{T}))$ | Transformers |
| Warmup + decay | Linear warmup then cosine | Large models |
| One-cycle | Increase then decrease | Fast training |

### 💻 Implementation

```python
import torch.optim.lr_scheduler as lr_scheduler

# Step decay (divide lr by 10 every 30 epochs)
scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# Cosine annealing
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# Warmup + cosine
def warmup_cosine(step, warmup_steps, total_steps):
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))

scheduler = lr_scheduler.LambdaLR(optimizer, warmup_cosine)
```

---

## 7. Code Implementation

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ============================================================
# OPTIMIZER IMPLEMENTATIONS FROM SCRATCH
# ============================================================

class SGD:
    def __init__(self, params, lr=0.01):
        self.params = list(params)
        self.lr = lr
    
    def step(self):
        for p in self.params:
            if p.grad is not None:
                p.data -= self.lr * p.grad
    
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

class SGDMomentum:
    def __init__(self, params, lr=0.01, momentum=0.9):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.v = [torch.zeros_like(p) for p in self.params]
    
    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is not None:
                self.v[i] = self.momentum * self.v[i] + p.grad
                p.data -= self.lr * self.v[i]
    
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

class Adam:
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0
    
    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            
            g = p.grad
            
            # Update biased first and second moment estimates
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g**2
            
            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            
            # Update parameters
            p.data -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
    
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

class AdamW:
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0
    
    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            
            g = p.grad
            
            # Update moments
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g**2
            
            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            
            # Decoupled weight decay + Adam update
            p.data -= self.lr * (m_hat / (torch.sqrt(v_hat) + self.eps) + self.weight_decay * p.data)
    
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

# ============================================================
# USING PYTORCH OPTIMIZERS
# ============================================================

def training_loop(model, train_loader, optimizer, epochs=10):
    """Standard training loop."""
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in train_loader:

            # Forward
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            # Update
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

# Example usage
model = nn.Linear(10, 2)

# SGD with momentum (good for CNNs)
opt_sgd = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

# Adam (quick experiments)
opt_adam = optim.Adam(model.parameters(), lr=0.001)

# AdamW (transformers)
opt_adamw = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
```

---

## 8. Which Optimizer to Use

### 📊 Recommendations

| Model Type | Optimizer | Learning Rate | Notes |
|------------|-----------|---------------|-------|
| **ResNet, CNNs** | SGD + Momentum | 0.1 with step decay | Best generalization |
| **BERT, GPT** | AdamW | 1e-4 to 5e-5 | With warmup |
| **ViT** | AdamW | 1e-3 to 1e-4 | High weight decay |
| **Fine-tuning** | AdamW | 1e-5 to 5e-5 | Lower than pre-training |
| **Quick prototype** | Adam | 1e-3 | Fast convergence |
| **GAN** | Adam | 1e-4 to 2e-4 | $\beta\_1 = 0.5$ often used |

### 📐 General Guidelines

```
1. START WITH: AdamW (works well almost everywhere)

2. FOR BEST RESULTS ON CNNs:
   SGD + Momentum + step decay
   (May need more tuning but often generalizes better)

3. LEARNING RATE:
   Too high → diverge or oscillate
   Too low → slow convergence
   Start with default, then tune

4. BATCH SIZE:
   Larger batch = more stable but may need larger lr
   Linear scaling: double batch → double lr (up to a point)
   
5. WEIGHT DECAY:
   Transformers: 0.01 - 0.1
   CNNs: 1e-4 - 1e-3
```

---

## 📚 Resources

| Type | Resource | Description |
|------|----------|-------------|
| 📄 | [Adam Paper](https://arxiv.org/abs/1412.6980) | Original Adam |
| 📄 | [AdamW Paper](https://arxiv.org/abs/1711.05101) | Decoupled weight decay |
| 📖 | [SGD Tricks](https://leon.bottou.org/publications/pdf/tricks-2012.pdf) | Bottou's guide |
| 🎥 | [Stanford CS231n](https://www.youtube.com/watch?v=_JB0AO7QxSA) | Optimization lecture |

---

## 🗺️ Navigation

| ⬅️ Previous | 🏠 Home | ➡️ Next |
|:-----------:|:-------:|:-------:|
| [Second Order](../06_second_order/README.md) | [Optimization](../README.md) | [Distance Metrics](../../04_distance_metrics/README.md) |

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer&animation=twinkling" width="100%"/>
</p>
