<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=120&section=header&text=Learning%20Rate%20Scheduling&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-06-45B7D1?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/lr-schedules.svg" width="100%">

*Caption: Different LR schedules work better for different tasks. Warmup + Cosine decay is the go-to for Transformers/LLMs. Step decay works well for CNNs. The schedule affects both convergence speed and final performance.*

---

## 📂 Overview

Learning rate scheduling adjusts the learning rate during training. Starting high enables exploration, while decaying enables fine-grained convergence. Warmup prevents early instability.

---

## 📐 Mathematical Foundations

### SGD Update Rule

```math
\theta_{t+1} = \theta_t - \eta_t \nabla_\theta \mathcal{L}(\theta_t)

```

Where $\eta_t$ is the learning rate at step $t$.

### Why Schedule the Learning Rate?

**Convergence Theory (Convex Case):**

For strongly convex functions, SGD converges with rate:

```math
\mathbb{E}[\|\theta_T - \theta^*\|^2] = O\left(\frac{1}{T}\right) \text{ when } \eta_t = O\left(\frac{1}{t}\right)

```

**Intuition:**
- **Large LR early:** Explore broadly, escape saddle points

- **Small LR late:** Fine-tune, converge precisely

---

## 🔬 Common Schedules

### 1. Constant Learning Rate

```math
\eta_t = \eta_0

```

**Use:** Quick experiments, baselines.

### 2. Step Decay

```math
\eta_t = \eta_0 \cdot \gamma^{\lfloor t/s \rfloor}

```

Where $\gamma \in (0, 1)$ is the decay factor and $s$ is the step size.

**Example:** $\eta_0 = 0.1$, $\gamma = 0.1$, $s = 30$ epochs

- Epochs 0-29: $\eta = 0.1$

- Epochs 30-59: $\eta = 0.01$

- Epochs 60-89: $\eta = 0.001$

### 3. Exponential Decay

```math
\eta_t = \eta_0 \cdot \gamma^t

```

**Continuous version:**

```math
\eta_t = \eta_0 \cdot e^{-\lambda t}

```

### 4. Polynomial Decay

```math
\eta_t = \eta_0 \cdot \left(1 - \frac{t}{T}\right)^p

```

Where $p$ controls decay speed (typically $p = 1$ for linear, $p = 2$ for quadratic).

### 5. Cosine Annealing

```math
\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{\pi t}{T}\right)\right)

```

**Properties:**
- Smooth decay from $\eta_{max}$ to $\eta_{min}$

- Slower decay at start and end

- Faster decay in the middle

### 6. Cosine Annealing with Warm Restarts

```math
\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{\pi T_{cur}}{T_i}\right)\right)

```

Where $T_{cur}$ is steps since last restart and $T_i$ is the $i$-th restart period.

---

## 🔬 Warmup

### The Problem

**Why warmup?**

1. **Adam's running averages:** Not calibrated at start

```math
m_0 = 0, \quad v_0 = 0

```math
Bias correction helps, but initial steps are still noisy.

2. **Large gradients:** Randomly initialized networks have large gradients

3. **Batch normalization:** Running statistics not calibrated

### Linear Warmup

```

\eta_t = \begin{cases}
\eta_{max} \cdot \frac{t}{T_{warmup}} & t < T_{warmup} \\
\text{schedule}(t) & t \geq T_{warmup}
\end{cases}

```math
### Warmup + Cosine Decay (LLM Standard)

```

\eta_t = \begin{cases}
\eta_{max} \cdot \frac{t}{T_w} & t < T_w \\
\eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{\pi(t - T_w)}{T - T_w}\right)\right) & t \geq T_w
\end{cases}

```math
**Typical values for LLMs:**
- $T_w = 2000$ steps (warmup)

- $\eta_{max} = 3 \times 10^{-4}$

- $\eta_{min} = 3 \times 10^{-5}$ (10% of max)

---

## 📊 OneCycle Policy

**Three phases:**

1. **Warmup:** LR increases from $\eta_{low}$ to $\eta_{max}$

2. **Annealing:** LR decreases from $\eta_{max}$ to $\eta_{low}$

3. **Fine-tune:** LR drops further to $\eta_{low}/10$

```

\eta_t = \begin{cases}
\eta_{low} + (\eta_{max} - \eta_{low}) \cdot \frac{t}{T_1} & t < T_1 \\
\eta_{max} - (\eta_{max} - \eta_{low}) \cdot \frac{t - T_1}{T_2 - T_1} & T_1 \leq t < T_2 \\
\eta_{low} \cdot \left(1 - \frac{t - T_2}{T - T_2}\right) & t \geq T_2
\end{cases}

```

**Momentum also changes:** High momentum when LR is low, low momentum when LR is high.

---

## 📊 Schedule Comparison

| Schedule | Formula | Best For |
|----------|---------|----------|
| **Constant** | $\eta_0$ | Quick experiments |
| **Step Decay** | $\eta_0 \gamma^{\lfloor t/s \rfloor}$ | CNNs (ResNet) |
| **Exponential** | $\eta_0 e^{-\lambda t}$ | Simple tasks |
| **Cosine** | $\frac{1}{2}(1 + \cos(\pi t/T))$ | General, Transformers |
| **Warmup+Cosine** | Linear → Cosine | LLMs 🔥 |
| **OneCycle** | Warmup → Peak → Decay | Fast training |
| **Polynomial** | $(1 - t/T)^p$ | Fine-tuning |

---

## 💻 Code Examples

```python
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import (
    StepLR, ExponentialLR, CosineAnnealingLR, 
    OneCycleLR, LambdaLR, CosineAnnealingWarmRestarts
)
import math

# Basic optimizer
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

# Step decay
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

# Exponential decay
scheduler = ExponentialLR(optimizer, gamma=0.95)

# Cosine annealing
scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)

# Cosine with warm restarts
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

# OneCycle
scheduler = OneCycleLR(
    optimizer, 
    max_lr=1e-3,
    total_steps=total_steps,
    pct_start=0.3,  # 30% warmup
    anneal_strategy='cos'
)

# Custom: Warmup + Cosine (LLM style)
def warmup_cosine_schedule(step, warmup_steps=2000, total_steps=100000, min_lr_ratio=0.1):
    """
    Warmup + Cosine decay schedule
    """
    if step < warmup_steps:
        # Linear warmup
        return step / warmup_steps
    else:
        # Cosine decay
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

scheduler = LambdaLR(optimizer, lr_lambda=lambda step: warmup_cosine_schedule(step))

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()
        optimizer.step()
        scheduler.step()  # Update LR (for OneCycle, step per batch)
    
    # For epoch-based schedulers:
    # scheduler.step()
    
    print(f"Epoch {epoch}, LR: {scheduler.get_last_lr()[0]:.6f}")

# Visualize schedule
def plot_schedule(scheduler, total_steps):
    import matplotlib.pyplot as plt
    lrs = []
    for step in range(total_steps):
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()
    
    plt.figure(figsize=(10, 4))
    plt.plot(lrs)
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.show()

```

### Complete LLM Training Schedule

```python
def get_lr(step, warmup_steps, total_steps, max_lr, min_lr):
    """
    Complete LR schedule for LLM training
    """
    # 1) Linear warmup
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    
    # 2) Cosine decay to min_lr
    if step < total_steps:
        decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)
    
    # 3) After total_steps, stay at min_lr
    return min_lr

# Usage in training
for step in range(total_steps):
    lr = get_lr(step, warmup_steps=2000, total_steps=100000, max_lr=3e-4, min_lr=3e-5)
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    # Training step
    loss.backward()
    optimizer.step()

```

---

## 🔬 Finding the Right Learning Rate

### Learning Rate Range Test

```python
def lr_range_test(model, dataloader, optimizer, start_lr=1e-7, end_lr=10, num_steps=100):
    """
    Find optimal LR by gradually increasing and monitoring loss
    """
    lrs, losses = [], []
    lr = start_lr
    mult = (end_lr / start_lr) ** (1 / num_steps)
    
    for step, batch in enumerate(dataloader):
        if step >= num_steps:
            break
        
        # Set LR
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Training step
        loss = model(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        lrs.append(lr)
        losses.append(loss.item())
        lr *= mult
    
    # Plot: Look for steepest descent region
    plt.plot(lrs, losses)
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.show()
    
    return lrs, losses

```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | Cosine Annealing Paper | [arXiv](https://arxiv.org/abs/1608.03983) |
| 📄 | OneCycleLR Paper | [arXiv](https://arxiv.org/abs/1708.07120) |
| 📄 | Warmup Analysis | [arXiv](https://arxiv.org/abs/1910.04209) |
| 📖 | PyTorch LR Schedulers | [Docs](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate) |
| 🎥 | Fast.ai LR Finder | [YouTube](https://www.youtube.com/watch?v=dxpyg3mP_rU) |
| 🇨🇳 | 学习率调度详解 | [知乎](https://zhuanlan.zhihu.com/p/32923584) |

---

## 🔗 Where This Topic Is Used

| Schedule | Application |
|----------|------------|
| **Warmup + Cosine** | LLMs (GPT, LLaMA, etc.) |
| **Step Decay** | CNNs (ResNet, VGG) |
| **Cosine** | Vision Transformers |
| **OneCycle** | Fast training, transfer learning |
| **Constant** | Fine-tuning, simple tasks |

---

⬅️ [Back: Regularization](../03_regularization/README.md)

---

⬅️ [Back: Training](../../README.md)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=80&section=footer" width="100%"/>
</p>
