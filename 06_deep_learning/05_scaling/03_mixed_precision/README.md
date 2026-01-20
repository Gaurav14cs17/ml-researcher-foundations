<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=120&section=header&text=Mixed%20Precision%20Training&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-06-45B7D1?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/mixed-precision.svg" width="100%">

*Caption: Mixed precision uses FP16/BF16 for forward/backward (fast, memory efficient) while keeping master weights in FP32 (accurate updates). BF16 has wider range than FP16, making it better for training.*

---

## 📐 Mathematical Foundations

### Floating Point Representation

```
FP32 (IEEE 754):
sign (1) | exponent (8) | mantissa (23)
Value = (-1)^s × 2^(e-127) × (1 + m/2²³)

FP16:
sign (1) | exponent (5) | mantissa (10)
Value = (-1)^s × 2^(e-15) × (1 + m/2¹⁰)

BF16:
sign (1) | exponent (8) | mantissa (7)
Same range as FP32, less precision

```

### Loss Scaling

```
Problem: Small gradients underflow in FP16

Solution:
1. Scale loss: L_scaled = L × scale_factor (e.g., 1024)
2. Backward: gradients are scaled
3. Unscale: gradient = scaled_gradient / scale_factor
4. Update: θ ← θ - α × gradient

Dynamic scaling: Adjust scale_factor based on overflow detection

```

### Numerical Error Analysis

```
FP16 precision: ~3.3 decimal digits
FP32 precision: ~7.2 decimal digits

Catastrophic cancellation:
If a ≈ b, then (a - b) loses precision

Accumulation in FP32:
sum = Σᵢ xᵢ  (computed in FP32 to avoid error accumulation)

```

---

## 📊 Formats

| Format | Bits | Range | Precision |
|--------|------|-------|-----------|
| FP32 | 32 | ±3.4×10³⁸ | High |
| FP16 | 16 | ±65,504 | Medium |
| BF16 | 16 | ±3.4×10³⁸ | Low |

---

## 🔑 Key Ideas

```
1. Forward/Backward in FP16 (fast!)
2. Master weights in FP32 (accurate updates)
3. Loss scaling (prevent underflow)

```

---

## 💻 Code

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()
    
    with autocast():  # FP16 forward
        loss = model(batch)
    
    scaler.scale(loss).backward()  # Scaled backward
    scaler.step(optimizer)
    scaler.update()

```

---

## 🌍 Benefits

| Benefit | Amount |
|---------|--------|
| Memory | 2x less |
| Speed | 2-3x faster |
| Accuracy | Same (usually) |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | Mixed Precision Paper | [arXiv](https://arxiv.org/abs/1710.03740) |
| 📖 | PyTorch AMP | [Docs](https://pytorch.org/docs/stable/amp.html) |
| 📖 | NVIDIA Mixed Precision | [Guide](https://developer.nvidia.com/automatic-mixed-precision) |
| 🇨🇳 | 混合精度训练详解 | [知乎](https://zhuanlan.zhihu.com/p/103685310) |
| 🇨🇳 | FP16/BF16对比 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88776412) |
| 🇨🇳 | AMP使用教程 | [B站](https://www.bilibili.com/video/BV1Y64y1Q7hi) |

## 🔗 Where This Topic Is Used

| Format | Application |
|--------|------------|
| **FP16** | Training speedup |
| **BF16** | TPU, LLM training |
| **INT8** | Inference |
| **FP8** | H100 training |

---

⬅️ [Back: Efficient](../02_efficient/README.md)

---

⬅️ [Back: Scaling](../../README.md)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=80&section=footer" width="100%"/>
</p>
