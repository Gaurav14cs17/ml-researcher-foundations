<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=1ABC9C&height=120&section=header&text=Lecture%2012%20Efficient%20Training&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-09-1ABC9C?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

# Lecture 12: Efficient Training

[← Back to Course](../) | [← Previous](../11_efficient_transformers/) | [Next: On-Device Training →](../13_on_device_training/)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/ml-researcher-foundations/blob/main/09_efficient_ml/12_efficient_training/demo.ipynb) ← **Try the code!**

---

## 📺 Video Lecture

| Resource | Link |
|----------|------|
| 🎥 **Lecture Video** | [Watch on YouTube](https://www.youtube.com/watch?v=2d9ZfRVaKFU&list=PL80kAHvQbh-pT4lCkDT53zT8DKmhE0idB&index=12) |
| 📊 **Slides** | [MIT Course Page](https://hanlab.mit.edu/courses/2023-fall-65940) |
| ⏱️ **Duration** | ~90 minutes |

### 📝 Video Summary

This lecture covers **efficient training techniques**:

- **Memory breakdown**: Where GPU memory goes during training
- **Mixed precision**: Using FP16/BF16 for 2× speedup
- **Gradient checkpointing**: Trading compute for memory
- **8-bit optimizers**: Reducing optimizer state memory
- **LoRA**: Parameter-efficient fine-tuning
- **torch.compile()**: JIT compilation for faster training

> 💡 *"Training uses 12-16× more memory than inference due to gradients and optimizer states—understanding this is key to efficiency."* — Prof. Song Han

---

![Overview](overview.png)

## Training Memory Breakdown

| Component | Size | Notes |
|-----------|------|-------|
| Model weights | W | Parameters |
| Gradients | W | Same size as weights |
| Optimizer states | 2W-8W | Adam: momentum + variance |
| Activations | Huge | Grows with batch × seq |

**Total for Adam: ~12-16x model size!**

---

## 📐 Mathematical Foundations & Proofs

### Mixed Precision Training

**Standard FP32 training:**

```math
w \leftarrow w - \eta \nabla_w \mathcal{L}
```

**Mixed precision:**
1. Weights stored in FP32 (master copy)
2. Forward/backward in FP16
3. Update in FP32

**Loss scaling (prevent underflow):**

```math
\mathcal{L}_{scaled} = s \cdot \mathcal{L}
g_{unscaled} = g_{scaled} / s
```

**Why it works:**

FP16 has limited range: \( 2^{-24} \) to \( 2^{15} \).

Gradients can underflow (become 0) if too small. Scaling by \( s \) (e.g., 1024) shifts values into representable range.

---

### BF16 vs FP16

**FP16:** 5 exponent bits, 10 mantissa bits
- Range: \( \pm 65504 \)
- Precision: ~3.3 digits

**BF16:** 8 exponent bits, 7 mantissa bits
- Range: \( \pm 3.4 \times 10^{38} \) (same as FP32!)
- Precision: ~2.4 digits

**BF16 advantage:** No overflow issues, simpler training (no loss scaling needed).

---

### Gradient Checkpointing

**Standard backprop memory:**

```math
M_{act} = \sum_{l=1}^L |a_l|
```

For L layers, store all L activations.

**Checkpointing strategy:**

Divide network into \( K \) segments. Only store activations at segment boundaries.

```math
M_{checkpoint} = K \cdot |a_{segment}| + \max_l |a_l|
```

**Optimal K:**

```math
K^* = \sqrt{L}
M_{optimal} = O(\sqrt{L})
```

**Proof:**
Total memory = \( K \cdot |a| + L/K \cdot |a| \)

Taking derivative w.r.t. K and setting to 0:

```math
\frac{d}{dK}(K + L/K) = 1 - L/K^2 = 0 \implies K = \sqrt{L}
```

**Trade-off:** ~33% more compute (recompute \( L - K \) activations).

---

### 8-bit Optimizer States

**Adam state per parameter:**
- First moment \( m_t \): FP32 (4 bytes)
- Second moment \( v_t \): FP32 (4 bytes)
- Total: 8 bytes per parameter

**8-bit Adam:**

Block-wise quantization:

```math
m_t^{int8} = \text{round}\left(\frac{m_t}{s_m}\right), \quad s_m = \max(|m_t|) / 127
```

**Memory:** 2 bytes (int8) + 2 bytes (scale) per block ≈ 2 bytes per parameter.

**4× reduction** in optimizer memory!

---

### LoRA: Low-Rank Adaptation

**Full fine-tuning:**

```math
W_{new} = W_0 + \Delta W
```

\( \Delta W \) is full rank: \( d \times d \) parameters.

**LoRA:**

```math
W_{new} = W_0 + BA
```

where \( B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times d}, r \ll d \).

**Parameter reduction:**

```math
\frac{|LoRA|}{|Full|} = \frac{2dr}{d^2} = \frac{2r}{d}
```

For \( d = 4096, r = 16 \): 0.8% of full fine-tuning parameters!

**Mathematical justification:**

Pre-trained weights occupy a low-rank subspace. Fine-tuning for a specific task adds a low-rank perturbation:

```math
\Delta W \approx \sum_{i=1}^r \sigma_i u_i v_i^T
```

LoRA directly parameterizes this low-rank structure.

---

### Gradient Accumulation

**Effective batch size:**

```math
B_{eff} = B_{micro} \times K_{accum}
```

**Update rule:**
```python
for step in range(steps):
    for i in range(K_accum):
        loss = model(batch_i) / K_accum
        loss.backward()  # Accumulate gradients
    optimizer.step()
    optimizer.zero_grad()
```

**Memory:** Only need \( B_{micro} \) in memory.

**Equivalence:** Mathematically equivalent to single batch of size \( B_{eff} \).

---

### Memory-Efficient Attention for Training

**Standard attention backward:**

Need to store \( Q, K, V, A \) (attention matrix) for backward pass.

```math
M_{attn} = O(N^2)
```

**FlashAttention backward:**

Recompute attention during backward pass.

```math
M_{attn} = O(N)
```

**Trade-off:** ~2× compute, but fits longer sequences.

---

## 🧮 Key Derivations

### Training Memory Formula

For model with \( N \) parameters, batch size \( B \), sequence length \( L \):

```math
M_{total} = M_{model} + M_{grad} + M_{opt} + M_{act}
M_{total} = N \cdot b_{weight} + N \cdot b_{grad} + N \cdot b_{opt} + B \cdot L \cdot d \cdot b_{act}
```

For FP32 Adam:
- \( b_{weight} = 4 \)
- \( b_{grad} = 4 \)
- \( b_{opt} = 8 \) (two FP32 moments)
- Total: \( 16N + \text{activations} \)

For mixed precision with 8-bit Adam:
- \( b_{weight} = 2 \) (FP16) + 4 (FP32 master) = 6
- \( b_{grad} = 2 \)
- \( b_{opt} = 2 \)
- Total: \( 10N + \text{activations} \)

**40% memory reduction!**

---

### torch.compile() Optimizations

**Operator fusion:**

```math
y = \text{gelu}(\text{dropout}(\text{linear}(x)))
```

Fused into single kernel: 1 memory read/write instead of 3.

**Memory reduction:** \( 3 \times \) less memory traffic.

**Computation graph optimization:**
- Dead code elimination
- Common subexpression elimination
- Layout optimization

**Speedup:** 1.5-2× typical.

---

## 🎯 Where Used

| Concept | Applications |
|---------|-------------|
| Mixed Precision | All modern LLM training |
| Gradient Checkpointing | Memory-constrained training |
| LoRA | Efficient fine-tuning |
| 8-bit Optimizer | Large model training |

---

## 🗺️ Navigation

| ⬅️ Previous | 🏠 Home | ➡️ Next |
|:-----------:|:-------:|:-------:|
| [← Efficient Transformers](../11_efficient_transformers/README.md) | [Efficient ML](../README.md) | [On-Device Training →](../13_on_device_training/README.md) |

---
## 📚 References

| Type | Resource | Link |
|------|----------|------|
| 📄 | Mixed Precision Training | [arXiv](https://arxiv.org/abs/1710.03740) |
| 📄 | Gradient Checkpointing | [arXiv](https://arxiv.org/abs/1604.06174) |
| 📄 | 8-bit Adam | [arXiv](https://arxiv.org/abs/2110.02861) |
| 📄 | LoRA | [arXiv](https://arxiv.org/abs/2106.09685) |
| 🎥 | MIT 6.5940 TinyML | [Course](https://hanlab.mit.edu/courses/2024-fall-65940) |

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=1ABC9C&height=80&section=footer" width="100%"/>
</p>
