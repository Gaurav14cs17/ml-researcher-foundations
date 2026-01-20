<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=1ABC9C&height=120&section=header&text=Lecture%2010%20MCUNet%20%26%20TinyML&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-09-1ABC9C?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

# Lecture 10: MCUNet & TinyML

[← Back to Course](../) | [← Previous](../09_knowledge_distillation/) | [Next: Efficient Transformers →](../11_efficient_transformers/)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/ml-researcher-foundations/blob/main/09_efficient_ml/10_mcunet_tinyml/demo.ipynb) ← **Try the code!**

---

## 📺 Video Lecture

| Resource | Link |
|----------|------|
| 🎥 **Lecture Video** | [Watch on YouTube](https://www.youtube.com/watch?v=IxQtK2SjWWM&list=PL80kAHvQbh-pT4lCkDT53zT8DKmhE0idB&index=10) |
| 📊 **Slides** | [MIT Course Page](https://hanlab.mit.edu/courses/2023-fall-65940) |
| ⏱️ **Duration** | ~90 minutes |

### 📝 Video Summary

This lecture covers **TinyML** for deploying ML on microcontrollers:

- **TinyML constraints**: Running ML on 256KB SRAM, 1MB Flash
- **MCUNet**: Co-design of network architecture and inference engine
- **TinyNAS**: Architecture search under extreme memory constraints
- **TinyEngine**: Memory-efficient inference engine for MCUs
- **Peak memory optimization**: Scheduling to minimize memory footprint
- **Applications**: Wake word detection, visual wake words, anomaly detection

> 💡 *"MCUNet brings ImageNet-level accuracy to microcontrollers with only 320KB of SRAM—a 1000× improvement."* — Prof. Song Han

---

![Overview](overview.png)

## What is TinyML?

Running machine learning on **microcontrollers** (MCUs):

| Device | RAM | Flash | Compute |
|--------|-----|-------|---------|
| Server GPU | 80GB | TB | 312 TFLOPS |
| Smartphone | 6GB | 128GB | 10 TFLOPS |
| Raspberry Pi | 4GB | 32GB | 13 GFLOPS |
| **MCU (STM32)** | **320KB** | **1MB** | **0.1 GFLOPS** |

**MCUs have 1000x less memory than phones!**

---

## TinyML Challenges

1. **Memory** — Model + activations must fit in KB
2. **No OS** — Direct hardware access
3. **No floating point** — Many MCUs only support INT
4. **Limited compute** — 100MHz vs 3GHz

---

## 📐 Mathematical Foundations & Proofs

### Memory Constraint Formalization

**Peak memory during inference:**

$$
M_{peak} = \max_l \left(M_{input}^l + M_{output}^l + M_{weights}^l\right)
$$

**Constraint:**

$$
M_{peak} \leq \text{SRAM}_{available}
$$

For STM32F746: $\text{SRAM} = 320\text{KB}$

---

### TinyNAS Search Objective

**Optimization problem:**

$$
\max_\alpha \text{Acc}(\alpha)
\text{s.t.} \quad M_{peak}(\alpha) \leq S_{max}
\quad\quad\quad M_{weights}(\alpha) \leq F_{max}
$$

where:
- $S_{max}$ = SRAM constraint
- $F_{max}$ = Flash constraint
- $\alpha$ = architecture parameters

---

### Peak Memory Analysis

For layer $l$ with input $X \in \mathbb{R}^{C_{in} \times H \times W}$ and output $Y \in \mathbb{R}^{C_{out} \times H' \times W'}$:

**Memory requirement:**

$$
M_l = C_{in} \cdot H \cdot W + C_{out} \cdot H' \cdot W'
$$

(assuming in-place computation for weights)

**Peak memory for sequential network:**

$$
M_{peak} = \max_l (M_{input}^l + M_{output}^l)
$$

---

### Patch-Based Inference

**Standard inference:**

$$
M = C \times H \times W
$$

**Patch-based (divide into $P \times P$ patches):**

$$
M_{patch} = C \times \frac{H}{P} \times \frac{W}{P} = \frac{M}{P^2}
$$

**Memory reduction:** $P^2 \times$

**Trade-off:** More computation due to overlapping regions.

---

### Layer Scheduling Optimization

**Problem:** Find layer execution order $\pi$ to minimize peak memory.

**Formulation:**

$$
\min_\pi \max_{t} M_{\pi(t)}
$$

**For networks with skip connections:**

The optimal schedule depends on when to compute branches.

**Example:** ResNet block
```
x → conv1 → bn1 → relu → conv2 → bn2 → + → relu
↓                                       ↑
+---------------------------------------+ (skip)
```

**Schedule A:** Compute skip path first, store result
**Schedule B:** Compute main path first, compute skip just before add

Optimal choice depends on relative sizes.

---

### Inverted Bottleneck Memory Analysis

For inverted bottleneck with expansion ratio $e$:

**Input:** $C \times H \times W$
**After expansion:** $eC \times H \times W$
**After depthwise:** $eC \times H \times W$
**Output:** $C' \times H \times W$

**Peak memory (naive):**

$$
M_{peak} = C \times H \times W + eC \times H \times W = (1+e) \cdot C \cdot H \cdot W
$$

**With fused operations:**

$$
M_{peak} = C \times H \times W + C' \times H \times W
$$

(Fusing avoids materializing expanded tensor)

---

### In-Place Depthwise Optimization

**Standard depthwise conv:**
- Read input: $C \times H \times W$
- Write output: $C \times H' \times W'$
- Memory: $C(HW + H'W')$

**In-place (when stride=1, padding=same):**
- Overwrite input with output
- Memory: $C \times H \times W$

**Constraint:** Can only use when output fits in input's memory.

---

### Im2col-Free Convolution

**Standard im2col:**
1. Reshape input to matrix: $M_{im2col} = C_{in} \times k^2 \times H' \times W'$
2. Matrix multiply with reshaped kernel
3. Memory: $O(C_{in} \cdot k^2 \cdot H' \cdot W')$

**Im2col-free:**
- Direct loop-based computation
- No intermediate buffer needed
- Memory: $O(1)$ extra

**Trade-off:** Slower but much less memory.

---

### Quantization for MCUs

**INT8 quantization benefits:**
1. **Memory:** 4× reduction (FP32 → INT8)
2. **Compute:** Use SIMD instructions (4 ops per cycle)
3. **No FPU:** Many MCUs lack floating-point unit

**Per-layer quantization:**

$$
W_l^{int8} = \text{round}\left(\frac{W_l}{s_l}\right)
$$

---

## 🧮 Key Derivations

### FLOPS Budget Estimation

**Available FLOPS:**

$$
F_{available} = \text{freq} \times \text{cycles/FLOP} \times t_{budget}
$$

For STM32F746 at 216MHz, 1 FLOP/cycle:

$$
F_{available} = 216\text{M} \times 1 \times 0.1\text{s} = 21.6 \text{ MFLOPS}
$$

**MobileNetV2 0.35× at 96×96:**

$$
F_{required} \approx 20 \text{ MFLOPS}
$$

Feasible!

---

### Model Size Breakdown

**Weights (INT8):**

$$
M_{weights} = \sum_l |\theta_l| \text{ bytes}
$$

**Must fit in Flash:** $M_{weights} \leq 1\text{MB}$

**Activations (INT8):**

$$
M_{act} = \max_l (|X_l| + |Y_l|) \text{ bytes}
$$

**Must fit in SRAM:** $M_{act} \leq 320\text{KB}$

---

### Energy Consumption Model

**Energy per inference:**

$$
E = P \times t = P \times \frac{\text{FLOPs}}{\text{FLOPS}}
$$

For MCU at 100mW computing 100 MFLOPs:

$$
E = 0.1\text{W} \times \frac{10^8}{10^8} = 0.1 \text{ Joule}
$$

**Battery life (3.7V, 500mAh):**

$$
\text{Inferences} = \frac{3.7 \times 0.5 \times 3600}{0.1} = 66,600
$$

---

## 🎯 Where Used

| Concept | Applications |
|---------|-------------|
| MCUNet | Visual wake words, IoT |
| TinyNAS | Device-specific architecture |
| TinyEngine | Optimized MCU inference |
| Patch-based Inference | Memory-constrained vision |

---

## 🗺️ Navigation

| ⬅️ Previous | 🏠 Home | ➡️ Next |
|:-----------:|:-------:|:-------:|
| [← Knowledge Distillation](../09_knowledge_distillation/README.md) | [Efficient ML](../README.md) | [Efficient Transformers →](../11_efficient_transformers/README.md) |

---

## 📚 References

| Type | Resource | Link |
|------|----------|------|
| 📄 | MCUNet | [arXiv](https://arxiv.org/abs/2007.10319) |
| 📄 | MCUNetV2 | [arXiv](https://arxiv.org/abs/2110.15352) |
| 📖 | TinyML Book | [O'Reilly](https://www.oreilly.com/library/view/tinyml/9781492052036/) |
| 🌐 | TinyML Foundation | [Website](https://www.tinyml.org/) |
| 💻 | TensorFlow Lite Micro | [TF](https://www.tensorflow.org/lite/microcontrollers) |
| 🎥 | MIT 6.5940 TinyML | [Course](https://hanlab.mit.edu/courses/2024-fall-65940) |

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=1ABC9C&height=80&section=footer" width="100%"/>
</p>
