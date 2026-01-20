<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=120&section=header&text=Efficient%20Training%20Techniques&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-06-45B7D1?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/efficient-training.svg" width="100%">

*Caption: Efficient training combines memory optimization (checkpointing, gradient accumulation), compute optimization (Flash Attention, fused kernels), parameter-efficient fine-tuning (LoRA, QLoRA), and data efficiency.*

---

## 📂 Overview

Modern deep learning requires efficient training techniques to handle large models and datasets. These methods reduce memory, speed up training, and enable training models that wouldn't otherwise fit on available hardware.

---

## 📐 Mathematical Foundations

### Gradient Checkpointing

```
Standard: Store all activations O(L × d)
Checkpointing: Store O(√L) activations
• Recompute during backward
• Memory: O(√L × d) vs O(L × d)
• Time: ~1.3x slower

```

### Gradient Accumulation

```
Effective batch size: B_eff = B × K
Where K = accumulation steps

Update: θ ← θ - η (1/K) Σₖ ∇L(Bₖ)

Mathematically equivalent to larger batch!

```

### Flash Attention Memory

```
Standard attention: O(N²) memory
Flash Attention: O(N) memory

Key idea: Tiling + online softmax
QKᵀV computed in blocks
Never materialize full N×N matrix

```

### LoRA Efficiency

```
Standard fine-tuning: ΔW ∈ ℝᵈˣᵈ → d² params
LoRA: ΔW = BA where B ∈ ℝᵈˣʳ, A ∈ ℝʳˣᵈ

Params: 2dr << d² when r << d
Typical: r = 8-64, d = 4096
Savings: 99.5%+ parameter reduction

```

---

## 🔑 Key Topics

| Topic | Description |
|-------|-------------|
| Fundamentals | Core concepts |
| Applications | ML use cases |
| Implementation | Code examples |

---

## 💻 Example

```python
# Gradient Checkpointing
from torch.utils.checkpoint import checkpoint
output = checkpoint(model.layer, input)

# Gradient Accumulation
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | Flash Attention | [arXiv](https://arxiv.org/abs/2205.14135) |
| 📄 | Gradient Checkpointing | [arXiv](https://arxiv.org/abs/1604.06174) |
| 📖 | PyTorch Checkpointing | [Docs](https://pytorch.org/docs/stable/checkpoint.html) |
| 🇨🇳 | 高效训练技巧 | [知乎](https://zhuanlan.zhihu.com/p/548036530) |
| 🇨🇳 | Flash Attention详解 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/118461491) |
| 🇨🇳 | 内存优化方法 | [B站](https://www.bilibili.com/video/BV1J94y1f7u5) |

---

## 🔗 Where This Topic Is Used

| Technique | Application |
|-----------|------------|
| **Gradient Checkpointing** | Memory saving |
| **Flash Attention** | Long sequences |
| **Fused Kernels** | GPU efficiency |
| **Tensor Cores** | Matrix multiplication |

---

⬅️ [Back: Distributed](../01_distributed/README.md) | ➡️ [Next: Mixed Precision](../03_mixed_precision/README.md)

---

⬅️ [Back: Scaling](../../README.md)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=80&section=footer" width="100%"/>
</p>
