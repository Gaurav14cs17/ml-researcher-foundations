# Learning Rate Scheduling

> **Control learning rate over training for better convergence**

---

## 🎯 Visual Overview

<img src="./images/lr-schedules.svg" width="100%">

*Caption: Different LR schedules work better for different tasks. Warmup + Cosine decay is the go-to for Transformers/LLMs. Step decay works well for CNNs. The schedule affects both convergence speed and final performance.*

---

## 📂 Overview

Learning rate scheduling adjusts the learning rate during training. Starting high enables exploration, while decaying enables fine-grained convergence. Warmup prevents early instability.

---

## 🔑 Common Schedules

| Schedule | Formula | Best For |
|----------|---------|----------|
| **Constant** | lr₀ | Quick experiments |
| **Step Decay** | lr₀ × γ^floor(t/s) | CNNs |
| **Cosine** | lr_min + ½(lr₀-lr_min)(1+cos(πt/T)) | General |
| **Warmup+Cosine** | Linear warmup → Cosine | LLMs 🔥 |
| **OneCycle** | Warmup → peak → decay | Fast training |

---

## 📐 Warmup: Why It Matters

```
Problem: Large gradients early in training cause instability
         (especially with Adam's running averages not yet calibrated)

Solution: Start with small LR, linearly increase to target

Warmup schedule:
lr(t) = lr₀ × (t / T_warmup)    for t < T_warmup
lr(t) = schedule(t)              for t ≥ T_warmup

Typical: T_warmup = 1000-10000 steps for LLMs
```

---

## 💻 Code

```python
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

# Cosine annealing
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

# Warmup + Cosine (common for LLMs)
def warmup_cosine(step, warmup_steps=1000, total_steps=100000):
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))

scheduler = LambdaLR(optimizer, warmup_cosine)

# Training loop
for step in range(total_steps):
    loss = train_step()
    optimizer.step()
    scheduler.step()  # Update LR
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | Cosine Annealing Paper | [arXiv](https://arxiv.org/abs/1608.03983) |
| 📄 | OneCycleLR Paper | [arXiv](https://arxiv.org/abs/1708.07120) |
| 📖 | PyTorch LR Schedulers | [Docs](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate) |
| 🇨🇳 | 学习率调度详解 | [知乎](https://zhuanlan.zhihu.com/p/32923584) |
| 🇨🇳 | Warmup原理 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88318597) |
| 🇨🇳 | LR Scheduler使用 | [B站](https://www.bilibili.com/video/BV1Y64y1Q7hi) |


## 🔗 Where This Topic Is Used

| Schedule | Application |
|----------|------------|
| **Cosine Decay** | Transformers, LLMs |
| **Step Decay** | CNNs |
| **Warmup** | Large models |
| **One Cycle** | Fast training |

---

⬅️ [Back: Training](../)

---

⬅️ [Back: Regularization](../regularization/)
