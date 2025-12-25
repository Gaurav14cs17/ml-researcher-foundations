<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=120&section=header&text=Optimizers&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-06-45B7D1?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/optimizers.svg" width="100%">

*Caption: Evolution of optimizers from SGD to Adam. SGD oscillates in ravines, momentum smooths the path, RMSprop adapts learning rates per parameter, and Adam combines both. AdamW is the go-to for modern LLMs.*

---

## 📐 Mathematical Formulations

### Stochastic Gradient Descent (SGD)

```
θₜ₊₁ = θₜ - α · ∇L(θₜ; Bₜ)

Where:
• α: learning rate
• Bₜ: mini-batch at step t
• ∇L: gradient of loss

Properties:
• Simple, well-understood
• Noisy gradients help escape local minima
• Requires careful learning rate tuning
```

### SGD with Momentum

```
vₜ₊₁ = β · vₜ + ∇L(θₜ)
θₜ₊₁ = θₜ - α · vₜ₊₁

Where:
• β ∈ [0.9, 0.99]: momentum coefficient
• v: velocity (exponential moving average of gradients)

Nesterov Momentum:
vₜ₊₁ = β · vₜ + ∇L(θₜ - α · β · vₜ)  # Look-ahead gradient
θₜ₊₁ = θₜ - α · vₜ₊₁

Benefits:
• Accelerates in consistent gradient direction
• Dampens oscillation in high-curvature directions
• Faster convergence than vanilla SGD
```

### RMSprop

```
sₜ₊₁ = β · sₜ + (1-β) · (∇L)²    # EMA of squared gradients
θₜ₊₁ = θₜ - α · ∇L / (√sₜ₊₁ + ε)

Where:
• β ≈ 0.9: decay rate
• ε ≈ 1e-8: numerical stability

Benefits:
• Adapts learning rate per parameter
• Larger updates for sparse gradients
• Smaller updates for frequent gradients
```

### Adam (Adaptive Moment Estimation)

```
mₜ = β₁ · mₜ₋₁ + (1-β₁) · ∇L        # First moment (momentum)
vₜ = β₂ · vₜ₋₁ + (1-β₂) · (∇L)²    # Second moment (RMSprop)

# Bias correction (important early in training!)
m̂ₜ = mₜ / (1 - β₁ᵗ)
v̂ₜ = vₜ / (1 - β₂ᵗ)

θₜ₊₁ = θₜ - α · m̂ₜ / (√v̂ₜ + ε)

Defaults: β₁=0.9, β₂=0.999, ε=1e-8, α=1e-3

Why bias correction?
At t=0: m₀=0, v₀=0
Without correction: estimates biased toward 0
With correction: unbiased estimates from start
```

### AdamW (Decoupled Weight Decay)

```
# Standard Adam with L2:
θₜ₊₁ = θₜ - α · m̂ₜ / (√v̂ₜ + ε) - α·λ·θₜ  ← WRONG!

# AdamW (correct):
θₜ₊₁ = θₜ - α · m̂ₜ / (√v̂ₜ + ε)
θₜ₊₁ = θₜ₊₁ - α·λ·θₜ                      ← Decoupled!

Why decoupled is better:
• L2 regularization gets scaled by Adam's adaptive LR
• Weight decay should be independent of gradient magnitude
• Better generalization, especially for Transformers
```

### LAMB (Layer-wise Adaptive Moments for Batch training)

```
# For large batch training
φ = ‖θ‖ / ‖update‖  # Trust ratio
θₜ₊₁ = θₜ - α · φ · (m̂ₜ / (√v̂ₜ + ε) + λθₜ)

Benefits:
• Scales to batch sizes of 32K+
• Layer-wise learning rate scaling
• Used for BERT pretraining
```

---

## 📊 Comparison Table

| Optimizer | Memory | Computation | Best For | Hyperparams |
|-----------|--------|-------------|----------|-------------|
| **SGD** | O(n) | O(n) | CNNs, generalization | lr |
| **Momentum** | O(2n) | O(n) | Faster convergence | lr, β |
| **RMSprop** | O(2n) | O(n) | RNNs, non-stationary | lr, β, ε |
| **Adam** | O(3n) | O(n) | Default choice | lr, β₁, β₂, ε |
| **AdamW** | O(3n) | O(n) | Transformers | lr, β₁, β₂, ε, λ |
| **LAMB** | O(3n) | O(n) | Large batch | lr, β₁, β₂, ε, λ |

---

## 💻 Code Examples

```python
import torch
import torch.optim as optim

model = YourModel()

# SGD with momentum (for CNNs)
optimizer = optim.SGD(
    model.parameters(), 
    lr=0.1, 
    momentum=0.9,
    weight_decay=1e-4,  # L2 regularization
    nesterov=True       # Nesterov momentum
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
    weight_decay=0.01  # Decoupled weight decay
)

# Different LR for different parameter groups
optimizer = optim.AdamW([
    {'params': model.encoder.parameters(), 'lr': 1e-5},
    {'params': model.decoder.parameters(), 'lr': 1e-4},
], weight_decay=0.01)

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()           # Clear gradients
        loss = model(batch)             # Forward pass
        loss.backward()                 # Compute gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
        optimizer.step()                # Update weights
```

### Learning Rate Scheduling

```python
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, LinearLR

# Cosine annealing (very common)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

# Warmup + cosine (for Transformers)
warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
main_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_steps)
scheduler = SequentialLR(optimizer, [warmup_scheduler, main_scheduler], [warmup_steps])

# In training loop
for epoch in range(num_epochs):
    train_one_epoch()
    scheduler.step()
```

---

## 🌍 ML Applications

| Model | Optimizer | LR | Notes |
|-------|-----------|-------|-------|
| **GPT-3** | Adam | 6e-5 → 0 | Cosine decay |
| **LLaMA** | AdamW | 3e-4 → 3e-5 | Warmup + cosine |
| **BERT** | Adam | 1e-4 → 0 | Linear warmup |
| **ResNet** | SGD+Momentum | 0.1 → 0 | Step decay |
| **ViT** | AdamW | 1e-3 | Heavy augmentation |
| **Stable Diffusion** | AdamW | 1e-4 | + EMA |

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
| 🇨🇳 | AdamW与LAMB详解 | [机器之心](https://www.jiqizhixin.com/articles/2019-10-25-8)

---

## 🔗 Where Optimizers Are Used

| Application | Optimizer & Settings |
|-------------|---------------------|
| **GPT / LLaMA Training** | AdamW, lr=3e-4, β=(0.9, 0.95), wd=0.1 |
| **BERT Pre-training** | Adam with linear warmup |
| **ResNet/ImageNet** | SGD + Momentum 0.9, step LR decay |
| **Stable Diffusion** | AdamW + Exponential Moving Average |
| **Fine-tuning LLMs** | AdamW, lower lr (1e-5 to 1e-4) |
| **RLHF Training** | Adam for reward model, PPO for policy |
| **LoRA Adapters** | AdamW for adapter weights only |
| **Large Batch Training** | LAMB for scaling to 32K+ batch size |

---


⬅️ [Back: Training](../)

---

⬅️ [Back: Normalization](../normalization/) | ➡️ [Next: Regularization](../regularization/)

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=80&section=footer" width="100%"/>
</p>
