<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=FF6B6B&height=120&section=header&text=Learning%20Rate%20Schedules&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-04-FF6B6B?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Why Learning Rate Schedules Matter

```
+---------------------------------------------------------+
|                                                         |
|   Constant LR:                With Schedule:            |
|                                                         |
|     Loss                        Loss                    |
|       |                           |                     |
|       |╲                          |╲                    |
|       | ╲                         | ╲                   |
|       |  ╲------- oscillates      |  ╲                  |
|       |                           |   ╲_____ converges  |
|       +--------------            +--------------        |
|                                                         |
|   Too high → oscillate          Start high → explore   |
|   Too low → slow                End low → converge     |
|                                                         |
+---------------------------------------------------------+

```

---

## 📐 Mathematical Foundation

### Robbins-Monro Conditions

```
For stochastic approximation to converge:

1. Σₜ αₜ = ∞        (step sizes sum to infinity)
2. Σₜ αₜ² < ∞       (step sizes are square-summable)

Interpretation:
• Condition 1: Take enough total steps to reach optimum
• Condition 2: Reduce noise effect over time

Examples that satisfy both:
• αₜ = 1/t        ✓
• αₜ = 1/√t       ✓
• αₜ = constant   ✗ (violates condition 2)

```

### Convergence Rate Analysis

```
For convex f with variance σ² in gradients:

Constant LR α:
  E[f(xₜ)] - f* ≤ O(1/t) + O(α σ²)
                   ↓         ↓
              optimization  variance floor

Decaying LR αₜ = α₀/√t:
  E[f(xₜ)] - f* ≤ O(1/√t)
  
  No variance floor - converges to optimum!

```

---

## 📐 Common Schedules

### 1. Step Decay

```
η(t) = η₀ × γ^⌊t/s⌋

Parameters:
• η₀ = initial learning rate
• γ = decay factor (typically 0.1)
• s = step size (epochs between decays)

Example (ResNet on ImageNet):
  η₀ = 0.1
  γ = 0.1 (divide by 10)
  s = 30 epochs
  
  Epochs 0-30:  η = 0.1
  Epochs 30-60: η = 0.01
  Epochs 60-90: η = 0.001

```

### 2. Exponential Decay

```
η(t) = η₀ × e^(-λt)

or equivalently:
η(t) = η₀ × γ^t  where γ = e^(-λ)

Properties:
• Smooth decay
• γ close to 1 → slow decay
• γ close to 0 → fast decay

Common: γ = 0.95 to 0.99 per epoch

```

### 3. Polynomial Decay

```
η(t) = η₀ × (1 - t/T)^p

Parameters:
• T = total training steps
• p = power (p=1 for linear, p=2 for quadratic)

Special case (linear decay):
η(t) = η_max × (1 - t/T)

Starts at η_max, linearly decreases to 0

```

### 4. Cosine Annealing

```
η(t) = η_min + (η_max - η_min) × (1 + cos(πt/T))/2

Properties:
• Smooth S-curve decay
• Starts at η_max
• Ends at η_min
• Gentle at start and end, faster in middle

Variant with warm restarts (SGDR):
η(t) = η_min + (η_max - η_min) × (1 + cos(π × t_i/T_i))/2

where t_i is time since last restart

```

### 5. Warmup + Decay (Transformers)

```
Original "Attention is All You Need" schedule:

η(t) = d^(-0.5) × min(t^(-0.5), t × warmup^(-1.5))

where:
• d = model dimension
• warmup = warmup steps (typically 4000)

Behavior:
• t < warmup: Linear increase (η ∝ t)
• t > warmup: Inverse square root decay (η ∝ 1/√t)

```

### 6. Linear Warmup + Linear Decay (BERT/GPT)

```
        η_max --------.
                      |
                      ╲
                       ╲
                        ╲
                         ╲
    η = 0 ---------------╲---
           warmup    total_steps

For t ≤ warmup:
  η(t) = η_max × t / warmup

For t > warmup:
  η(t) = η_max × (total - t) / (total - warmup)

```

### 7. One Cycle (fast.ai)

```
Phase 1 (Warmup):
  η: η_min → η_max
  momentum: β_max → β_min

Phase 2 (Annealing):
  η: η_max → η_min (cosine)
  momentum: β_min → β_max

Benefits:
• Explore early (high LR)
• Converge late (low LR)
• Counter-cyclic momentum helps

```

---

## 📐 Mathematical Analysis

### Why Warmup Helps

```
Problem at initialization:
• Gradients can be very large (random weights)
• Adam's m, v estimates are zero initially
• Large initial updates can destabilize training

Warmup solution:
• Start with tiny LR → small updates
• Gradually increase → let optimizer adapt
• Adam's m, v accumulate meaningful statistics

Theoretical justification:
At step t, Adam uses:
  m̂_t = m_t / (1 - β₁^t)  ← bias correction

Early steps (small t):
  (1 - β₁^t) is small → m̂_t is amplified
  Combined with large gradients → instability

Warmup counteracts this amplification

```

### Why Decay Helps

```
SGD update with noise:
  x_{t+1} = x_t - α_t(∇f(x_t) + ε_t)

where ε_t is gradient noise with variance σ²

Stationary distribution analysis:
  Near optimum x*, the iterates fluctuate around x*
  with variance proportional to α × σ²

Constant LR:
  Variance stays constant → never converges exactly

Decaying LR (α_t → 0):
  Variance → 0 → converges to x*

```

---

## 💻 Code Examples

### PyTorch Schedulers

```python
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import (
    StepLR, 
    ExponentialLR,
    CosineAnnealingLR,
    OneCycleLR,
    LambdaLR
)

optimizer = optim.Adam(model.parameters(), lr=0.001)

# 1. Step decay
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

# 2. Exponential decay
scheduler = ExponentialLR(optimizer, gamma=0.95)

# 3. Cosine annealing
scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

# 4. One cycle
scheduler = OneCycleLR(
    optimizer, 
    max_lr=0.01, 
    epochs=100, 
    steps_per_epoch=len(dataloader)
)

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()
        optimizer.step()
        
        # For OneCycleLR: step after each batch
        if isinstance(scheduler, OneCycleLR):
            scheduler.step()
    
    # For other schedulers: step after each epoch
    if not isinstance(scheduler, OneCycleLR):
        scheduler.step()

```

### Custom Warmup + Cosine Schedule

```python
import math

def get_lr(step, total_steps, warmup_steps, max_lr, min_lr=0):
    """
    Linear warmup + cosine decay schedule
    """
    if step < warmup_steps:
        # Linear warmup
        return max_lr * step / warmup_steps
    else:
        # Cosine decay
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

# Create scheduler
scheduler = LambdaLR(
    optimizer,
    lr_lambda=lambda step: get_lr(step, total_steps, warmup_steps, max_lr) / max_lr
)

```

### Transformer Schedule (Original)

```python
def transformer_lr(step, d_model=512, warmup_steps=4000):
    """
    From "Attention is All You Need" (Vaswani et al., 2017)
    """
    step = max(step, 1)  # Avoid division by zero
    return (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))

# Usage with LambdaLR
scheduler = LambdaLR(
    optimizer,
    lr_lambda=lambda step: transformer_lr(step, d_model=512, warmup_steps=4000)
)

```

---

## 📊 Best Practices by Task

### Computer Vision (ResNet, ViT)

```
Classic (ResNet):
  • Initial LR: 0.1
  • Schedule: Step decay
  • Decay: ÷10 at epochs 30, 60, 90
  • Total: 90-120 epochs

Modern (ViT):
  • Initial LR: 3e-4
  • Schedule: Warmup + cosine
  • Warmup: 5-10 epochs
  • Total: 300-1000 epochs

```

### NLP (BERT, GPT)

```
BERT:
  • Peak LR: 1e-4 to 5e-4
  • Schedule: Linear warmup + linear decay
  • Warmup: ~10% of training

GPT:
  • Peak LR: 2.5e-4 to 6e-4
  • Schedule: Warmup + cosine
  • Warmup: 375M tokens (GPT-2)
  • Min LR: 10% of peak

```

### Fine-tuning

```
Transfer learning:
  • LR: 10-100× smaller than pre-training
  • Schedule: Linear decay (no warmup or short warmup)
  • Freeze early layers initially
  
BERT fine-tuning:
  • LR: 2e-5 to 5e-5
  • Epochs: 2-4
  • Linear decay

```

### Reinforcement Learning

```
PPO/A2C:
  • Linear decay often works well
  • Anneal from 3e-4 to 0

SAC/TD3:
  • Constant LR often sufficient
  • LR: 3e-4

```

---

## 📐 Advanced Topics

### Learning Rate Range Test

```python
def lr_range_test(model, train_loader, start_lr=1e-7, end_lr=10, num_iters=100):
    """
    Find good learning rate by gradually increasing LR
    and plotting loss vs LR
    """
    optimizer = optim.SGD(model.parameters(), lr=start_lr)
    
    lr_mult = (end_lr / start_lr) ** (1 / num_iters)
    lrs, losses = [], []
    
    for i, (x, y) in enumerate(train_loader):
        if i >= num_iters:
            break
            
        optimizer.zero_grad()
        loss = model(x, y)
        loss.backward()
        optimizer.step()
        
        lrs.append(optimizer.param_groups[0]['lr'])
        losses.append(loss.item())
        
        # Increase LR
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_mult
    
    # Plot: Good LR is where loss decreases fastest
    # (steepest negative slope)
    return lrs, losses

```

### Layer-wise Learning Rates

```python
# Different LR for different parts of model
optimizer = optim.Adam([
    {'params': model.backbone.parameters(), 'lr': 1e-5},  # Pre-trained
    {'params': model.head.parameters(), 'lr': 1e-3}       # New layers
])

```

### Cyclic Learning Rates

```python
from torch.optim.lr_scheduler import CyclicLR

scheduler = CyclicLR(
    optimizer,
    base_lr=1e-5,
    max_lr=1e-3,
    step_size_up=2000,  # Steps to increase
    mode='triangular2'   # Halve amplitude each cycle
)

```

---

## 📊 Summary Comparison

| Schedule | Best For | Pros | Cons |
|----------|----------|------|------|
| **Constant** | Debugging | Simple | Suboptimal |
| **Step decay** | CNNs | Predictable | Abrupt changes |
| **Exponential** | General | Smooth | Hard to tune λ |
| **Cosine** | Most tasks | Smooth, proven | Fixed duration |
| **Warmup+decay** | Transformers | Stable training | Extra params |
| **One cycle** | Fast training | Fast convergence | Single epoch needed |

---

## 📚 Resources

| Type | Title | Link |
|------|-------|------|
| 📄 | Cyclical LR | [arXiv](https://arxiv.org/abs/1506.01186) |
| 📄 | SGDR (Warm Restarts) | [arXiv](https://arxiv.org/abs/1608.03983) |
| 📄 | 1cycle Policy | [arXiv](https://arxiv.org/abs/1803.09820) |
| 📖 | fast.ai Course | [fast.ai](https://www.fast.ai/) |
| 🎥 | Learning Rate Finder | [YouTube](https://www.youtube.com/watch?v=WW8TrbM3vLY) |
| 🇨🇳 | 学习率调度策略 | [知乎](https://zhuanlan.zhihu.com/p/32923584) |

---

⬅️ [Back: Constrained Optimization](../05_constrained_optimization/) | ➡️ [Next: Linear Programming](../06_linear_programming/)

> **Note:** This folder covers learning rate schedules used in optimization algorithms.

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=FF6B6B&height=80&section=footer" width="100%"/>
</p>
