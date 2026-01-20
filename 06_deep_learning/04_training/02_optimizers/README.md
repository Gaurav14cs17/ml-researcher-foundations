<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=120&section=header&text=Optimizers&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-06-45B7D1?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/optimizers.svg" width="100%">

*Caption: Evolution of optimizers from SGD to Adam. SGD oscillates in ravines, momentum smooths the path, RMSprop adapts learning rates per parameter, and Adam combines both. AdamW is the go-to for modern LLMs.*

---

## 📐 Mathematical Foundations

### 1. Stochastic Gradient Descent (SGD)

**Update Rule:**

```math
\theta_{t+1} = \theta_t - \alpha \cdot \nabla L(\theta_t; B_t)

```

Where:

- $\alpha$: learning rate

- $B_t$: mini-batch at step $t$

- $\nabla L$: gradient of loss

**Properties:**
- Simple, well-understood

- Noisy gradients help escape local minima

- Requires careful learning rate tuning

**Convergence (Convex Case):**

For $L$-smooth, convex $f$ with optimal $f^*$:

```math
\mathbb{E}[f(\theta_T)] - f^* \leq O\left(\frac{1}{\sqrt{T}}\right)

```

---

### 2. Momentum

**Classical Momentum:**

```math
v_{t+1} = \beta v_t + \nabla L(\theta_t)
\theta_{t+1} = \theta_t - \alpha v_{t+1}

```

Where $\beta \in [0.9, 0.99]$ is the momentum coefficient.

**Intuition: Ball Rolling Down Hill**
- Accumulates velocity in consistent gradient direction

- Dampens oscillations in inconsistent directions

- Escapes shallow local minima

**Nesterov Momentum (Look-ahead):**

```math
v_{t+1} = \beta v_t + \nabla L(\theta_t - \alpha \beta v_t)
\theta_{t+1} = \theta_t - \alpha v_{t+1}

```

**Why look-ahead?** Evaluate gradient at "future" position → more accurate update.

**Convergence Rate:**

| Method | Convex | Strongly Convex |
|--------|--------|-----------------|
| **SGD** | $O(1/\sqrt{k})$ | $O((1-\mu/L)^k)$ |
| **Momentum** | $O(1/k^2)$ | $O((1-\sqrt{\mu/L})^k)$ |

Momentum achieves the **optimal rate** for gradient descent!

### Mathematical Proof: Why Momentum Helps in Ravines

Consider loss: $L(\theta) = \frac{1}{2}(a\theta_1^2 + b\theta_2^2)$ with $a \gg b$ (ill-conditioned).

**SGD:** 

- Oscillates in $\theta_1$ direction (high curvature)

- Slow progress in $\theta_2$ direction (low curvature)

**Momentum:**
- Velocity in $\theta_1$ cancels (oscillating gradients)

- Velocity in $\theta_2$ accumulates (consistent gradients)

- Result: faster convergence along ravine

---

### 3. RMSprop

**Update Rules:**

```math
s_{t+1} = \beta s_t + (1-\beta) (\nabla L)^2
\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{s_{t+1} + \epsilon}} \nabla L

```

Where:

- $s$: exponential moving average of squared gradients

- $\beta \approx 0.9$: decay rate

- $\epsilon \approx 10^{-8}$: numerical stability

**Intuition: Adaptive Learning Rates**
- Parameters with large gradients → smaller effective LR

- Parameters with small gradients → larger effective LR

- Normalizes gradient magnitude per parameter

---

### 4. Adam (Adaptive Moment Estimation)

**The Algorithm:**

```math
m_t = \beta_1 m_{t-1} + (1-\beta_1) \nabla L \quad \text{(first moment = momentum)}
v_t = \beta_2 v_{t-1} + (1-\beta_2) (\nabla L)^2 \quad \text{(second moment = RMSprop)}

```

**Bias Correction:**

```math
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}
\hat{v}_t = \frac{v_t}{1 - \beta_2^t}

```

**Update:**

```math
\theta_{t+1} = \theta_t - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}

```

**Default hyperparameters:** $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$, $\alpha=10^{-3}$

### Why Bias Correction?

**Problem:** At $t=0$: $m_0=0$, $v_0=0$. Early estimates biased toward zero.

**Proof:**

For first moment at step $t$:

```math
m_t = (1-\beta_1) \sum_{i=1}^{t} \beta_1^{t-i} g_i

```

Taking expectation (assuming $g_i$ has mean $\bar{g}$):

```math
\mathbb{E}[m_t] = (1-\beta_1) \sum_{i=1}^{t} \beta_1^{t-i} \bar{g} = (1-\beta_1^t) \bar{g}

```

To get unbiased estimate:

```math
\hat{m}_t = \frac{m_t}{1-\beta_1^t} \implies \mathbb{E}[\hat{m}_t] = \bar{g}

```

---

### 5. AdamW (Decoupled Weight Decay)

**The Problem with Adam + L2:**

Standard L2 regularization in Adam:

```math
\theta_{t+1} = \theta_t - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \alpha \lambda \theta_t

```

**Issue:** Weight decay gets scaled by $1/\sqrt{\hat{v}_t}$ → inconsistent regularization!

**AdamW Solution:**

```math
\theta_{t+1} = \theta_t - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
\theta_{t+1} = \theta_{t+1} - \alpha \lambda \theta_t \quad \text{(separate step)}

```

**Why better?**
- Weight decay independent of gradient magnitude

- More consistent regularization across parameters

- Better generalization, especially for Transformers

---

### 6. LAMB (Layer-wise Adaptive Moments for Batch training)

For large batch training (batch size 32K+):

```math
\phi_l = \frac{\|\theta_l\|}{\|\text{update}_l\|} \quad \text{(trust ratio per layer)}
\theta_l^{t+1} = \theta_l^t - \alpha \cdot \phi_l \cdot \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda\theta_l^t\right)

```

**Key Insight:** Scale updates per layer based on parameter and gradient norms.

---

## 📊 Comparison Table

| Optimizer | Memory | Best For | Key Hyperparams |
|-----------|--------|----------|-----------------|
| **SGD** | $O(n)$ | CNNs, generalization | lr |
| **Momentum** | $O(2n)$ | Faster convergence | lr, $\beta$ |
| **RMSprop** | $O(2n)$ | RNNs, non-stationary | lr, $\beta$, $\epsilon$ |
| **Adam** | $O(3n)$ | Default choice | lr, $\beta_1$, $\beta_2$, $\epsilon$ |
| **AdamW** | $O(3n)$ | Transformers | lr, $\beta_1$, $\beta_2$, $\epsilon$, $\lambda$ |
| **LAMB** | $O(3n)$ | Large batch | lr, $\beta_1$, $\beta_2$, $\epsilon$, $\lambda$ |

---

## 💻 Implementation

```python
import torch
import torch.optim as optim
import math

class SGDMomentum:
    """Manual SGD with Momentum implementation"""
    
    def __init__(self, params, lr, momentum=0.9, nesterov=False):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.velocities = [torch.zeros_like(p) for p in self.params]
    
    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            
            # Update velocity
            self.velocities[i] = self.momentum * self.velocities[i] + p.grad
            
            if self.nesterov:
                # Nesterov: use gradient at look-ahead position
                update = self.momentum * self.velocities[i] + p.grad
            else:
                update = self.velocities[i]
            
            p.data -= self.lr * update
    
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

class Adam:
    """Manual Adam implementation"""
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0
        
        self.m = [torch.zeros_like(p) for p in self.params]  # First moment
        self.v = [torch.zeros_like(p) for p in self.params]  # Second moment
    
    def step(self):
        self.t += 1
        
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            
            g = p.grad
            
            # Update biased moments
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g * g
            
            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            p.data -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
    
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

class AdamW:
    """Manual AdamW implementation"""
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
    
    def step(self):
        self.t += 1
        
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            
            g = p.grad
            
            # Update moments
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g * g
            
            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Adam update
            p.data -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
            
            # Decoupled weight decay
            p.data -= self.lr * self.weight_decay * p.data

# PyTorch usage
model = torch.nn.Linear(100, 10)

# SGD with momentum (for CNNs)
optimizer = optim.SGD(
    model.parameters(), 
    lr=0.1, 
    momentum=0.9,
    weight_decay=1e-4,
    nesterov=True
)

# Adam (general purpose)
optimizer = optim.Adam(
    model.parameters(), 
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8
)

# AdamW (for Transformers - recommended!)
optimizer = optim.AdamW(
    model.parameters(), 
    lr=1e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01
)

# Different LR for different parameter groups
optimizer = optim.AdamW([
    {'params': model.weight, 'lr': 1e-5},
    {'params': model.bias, 'lr': 1e-4},
], weight_decay=0.01)

```

### Learning Rate Scheduling

```python
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# Warmup + Cosine decay (standard for LLMs)
def get_scheduler(optimizer, warmup_steps, total_steps):
    warmup = LinearLR(
        optimizer, 
        start_factor=0.01, 
        total_iters=warmup_steps
    )
    cosine = CosineAnnealingLR(
        optimizer, 
        T_max=total_steps - warmup_steps
    )
    return SequentialLR(optimizer, [warmup, cosine], [warmup_steps])

# Usage
scheduler = get_scheduler(optimizer, warmup_steps=1000, total_steps=100000)

for step in range(100000):
    train_step()
    scheduler.step()

```

---

## 🌍 Real-World Usage

| Model | Optimizer | LR Schedule | Notes |
|-------|-----------|-------------|-------|
| **GPT-3** | Adam | Cosine 6e-5 → 0 | 175B params |
| **LLaMA** | AdamW | Warmup + Cosine 3e-4 → 3e-5 | $\beta=(0.9, 0.95)$ |
| **BERT** | Adam | Linear warmup 1e-4 → 0 | Standard for NLU |
| **ResNet** | SGD+Momentum | Step decay 0.1 → 0.001 | Best for ImageNet |
| **ViT** | AdamW | Cosine 1e-3 | Heavy augmentation |
| **Stable Diffusion** | AdamW | Constant 1e-4 | + EMA |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | Adam Paper | [arXiv](https://arxiv.org/abs/1412.6980) |
| 📄 | AdamW Paper | [arXiv](https://arxiv.org/abs/1711.05101) |
| 📄 | LAMB Paper | [arXiv](https://arxiv.org/abs/1904.00962) |
| 🎥 | 3Blue1Brown: Gradient Descent | [YouTube](https://www.youtube.com/watch?v=IHZwWFHWa-w) |
| 📖 | Ruder's Optimizer Overview | [Blog](https://ruder.io/optimizing-gradient-descent/) |
| 📖 | PyTorch Optimizers | [Docs](https://pytorch.org/docs/stable/optim.html) |
| 🇨🇳 | Adam优化器详解 | [知乎](https://zhuanlan.zhihu.com/p/32230623) |
| 🇨🇳 | 优化器对比分析 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88318597) |
| 🇨🇳 | 深度学习优化算法 | [B站](https://www.bilibili.com/video/BV1Y64y1Q7hi) |

---

⬅️ [Back: Normalization](../01_normalization/README.md) | ➡️ [Next: Regularization](../03_regularization/README.md)

---

⬅️ [Back: Training](../../README.md)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=80&section=footer" width="100%"/>
</p>
