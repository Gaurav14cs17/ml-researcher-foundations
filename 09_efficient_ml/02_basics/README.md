<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=1ABC9C&height=120&section=header&text=Lecture%202%20Neural%20Network%20Basics&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-09-1ABC9C?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

# Lecture 2: Neural Network Basics for Efficiency

[← Back to Course](../) | [← Previous](../01_introduction/) | [Next: Pruning I →](../03_pruning_sparsity_1/)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/ml-researcher-foundations/blob/main/09_efficient_ml/02_basics/demo.ipynb) ← **Try the code!**

---

## 📺 Video Lecture

| Resource | Link |
|----------|------|
| 🎥 **Lecture Video** | [Watch on YouTube](https://www.youtube.com/watch?v=rCFvPEQTxKI&list=PL80kAHvQbh-pT4lCkDT53zT8DKmhE0idB&index=2) |
| 📊 **Slides** | [MIT Course Page](https://hanlab.mit.edu/courses/2023-fall-65940) |
| ⏱️ **Duration** | ~90 minutes |

### 📝 Video Summary

This lecture establishes the **computational foundations** for understanding efficient ML:

- **Roofline model**: Understanding compute-bound vs memory-bound operations

- **FLOPs counting**: How to calculate computational cost of each layer type

- **Memory hierarchy**: GPU memory architecture (SRAM, HBM, DRAM)

- **Layer analysis**: Convolution, Linear, Attention complexity breakdowns

- **Hardware considerations**: CPU vs GPU vs TPU vs MCU trade-offs

- **Profiling tools**: PyTorch profiler, NVIDIA Nsight

> 💡 *"Most deep learning operations today are memory-bound, not compute-bound. Understanding this is key to optimization."* — Prof. Song Han

---

![Overview](overview.png)

## Compute Primitives

### FLOPS vs Memory

Understanding efficiency requires knowing what's expensive:

| Operation | Compute | Memory |
|-----------|---------|--------|
| Matrix multiply | High | Low |
| Attention | O(N²) | O(N²) |
| Element-wise | Low | High (bandwidth limited) |

---

## Roofline Model

The **roofline model** helps understand whether your code is:

- **Compute-bound**: Limited by FLOPS (matrix ops)

- **Memory-bound**: Limited by memory bandwidth (element-wise ops)

```
         /----------------- Compute ceiling
        /
       /
      /
-----/  <-- Memory bandwidth ceiling
    |
Arithmetic Intensity (FLOPS/Byte)

```

---

## Key Neural Network Layers

### 1. Convolution

```python
# Memory: O(C_in × C_out × K × K)
# Compute: O(C_in × C_out × K² × H × W)

```

### 2. Linear (Dense)

```python
# Memory: O(in_features × out_features)
# Compute: O(batch × in_features × out_features)

```

### 3. Attention

```python
# Memory: O(N²) for attention matrix
# Compute: O(N² × d) for QK^T and attention × V

```

---

## Hardware Considerations

| Hardware | Good At | Limited By |
|----------|---------|------------|
| CPU | Flexibility | Parallelism |
| GPU | Massive parallelism | Memory bandwidth |
| TPU | Matrix ops | Flexibility |
| MCU | Energy efficiency | Everything |

---

## Efficiency Metrics

1. **Latency** - Time for single inference

2. **Throughput** - Inferences per second

3. **Energy** - Joules per inference

4. **Model size** - Parameters × bytes per param

5. **Peak memory** - Max RAM during inference

---

## 📐 Mathematical Foundations & Proofs

### Roofline Model Analysis

#### Arithmetic Intensity Definition

$$I = \frac{\text{FLOPs}}{\text{Bytes accessed}}$$

#### Attainable Performance

$$P = \min\left(\pi, I \times \beta\right)$$

where:

- \( \pi \) = Peak compute (FLOPS)

- \( \beta \) = Peak memory bandwidth (bytes/s)

- \( I \) = Arithmetic intensity

**Proof of Roofline Bound:**

1. Performance cannot exceed peak compute: \( P \leq \pi \)

2. Performance limited by data delivery rate: \( P \leq I \times \beta \)

3. Therefore: \( P = \min(\pi, I \times \beta) \)

**Ridge Point:** The intensity where compute and memory bounds meet:

$$I_{\text{ridge}} = \frac{\pi}{\beta}$$

For NVIDIA A100: \( I_{\text{ridge}} = \frac{312 \text{ TFLOPS}}{2 \text{ TB/s}} = 156 \text{ FLOPs/byte} \)

---

### Layer Complexity Analysis

#### Fully Connected Layer

For input \( x \in \mathbb{R}^{n} \), weight \( W \in \mathbb{R}^{m \times n} \):

$$\text{FLOPs} = 2mn
\text{Memory (weights)} = mn \times b
\text{Arithmetic Intensity} = \frac{2mn}{mn \times b} = \frac{2}{b}$$

For FP16 (b=2): \( I = 1 \) → **Memory-bound!**

#### Convolution Layer

For input \( X \in \mathbb{R}^{C_{in} \times H \times W} \), kernel \( K \in \mathbb{R}^{C_{out} \times C_{in} \times k \times k} \):

$$\text{FLOPs} = 2 \times C_{in} \times C_{out} \times k^2 \times H_{out} \times W_{out}$$

**Proof:**
Each output pixel requires:

- \( C_{in} \times k^2 \) multiplications (convolve over input channels and kernel)

- \( C_{in} \times k^2 - 1 \approx C_{in} \times k^2 \) additions

Total output pixels: \( C_{out} \times H_{out} \times W_{out} \)

$$\text{FLOPs} = C_{out} \times H_{out} \times W_{out} \times 2 \times C_{in} \times k^2$$

#### Self-Attention Complexity

For sequence length \( N \), hidden dimension \( d \):

**Step 1: QKV Projections**

$$\text{FLOPs}_{QKV} = 3 \times 2Nd^2 = 6Nd^2$$

**Step 2: Attention Scores** \( A = QK^T \)

$$\text{FLOPs}_{scores} = 2N^2d$$

**Step 3: Attention × Values** \( O = \text{softmax}(A)V \)

$$\text{FLOPs}_{output} = 2N^2d$$

**Step 4: Output Projection**

$$\text{FLOPs}_{proj} = 2Nd^2$$

**Total:**

$$\text{FLOPs}_{attention} = 8Nd^2 + 4N^2d = 4Nd(2d + N)$$

For \( N \gg d \): **O(N²)** dominates (quadratic in sequence length)

---

### Memory Access Patterns

#### Cache Efficiency

For a matrix multiplication \( C = AB \) with tiling:

$$\text{Cache misses} = O\left(\frac{mn + nk + mk}{B}\right)$$

where B is the cache block size.

With optimal tiling (block size \( b \)):

$$\text{Cache misses} = O\left(\frac{mnk}{b\sqrt{M}}\right)$$

where M is cache size. This is the **I/O complexity lower bound** (Hong & Kung, 1981).

#### Memory Bandwidth Utilization

$$\text{Utilization} = \frac{\text{Actual Bandwidth}}{\text{Peak Bandwidth}} = \frac{\text{Data Accessed} / \text{Time}}{\beta}$$

Goal: Achieve >80% bandwidth utilization for memory-bound ops.

---

### Data Type Precision Analysis

| Type | Bits | Range | Precision | Memory Savings |
|------|------|-------|-----------|----------------|
| FP32 | 32 | ±3.4e38 | 7 digits | 1× |
| FP16 | 16 | ±65504 | 3 digits | 2× |
| BF16 | 16 | ±3.4e38 | 2 digits | 2× |
| INT8 | 8 | -128 to 127 | Exact | 4× |

**BF16 vs FP16:**
- BF16: 8 exponent bits, 7 mantissa bits (same range as FP32)

- FP16: 5 exponent bits, 10 mantissa bits (more precision, less range)

BF16 preferred for training (no overflow issues).

---

## 🧮 Key Derivations

### Compute vs Memory Bound Classification

An operation is **compute-bound** if:

$$I > I_{\text{ridge}} = \frac{\pi}{\beta}$$

An operation is **memory-bound** if:

$$I < I_{\text{ridge}}$$

**Example: Batch Size Effect on Linear Layer**

For batch size B, input dim n, output dim m:

$$I = \frac{2Bmn}{(Bn + mn + Bm) \times b} \approx \frac{2Bmn}{mn \times b} = \frac{2B}{b}$$

For B=1 with FP16: I = 1 (memory-bound)
For B=128 with FP16: I = 128 (compute-bound)

**Conclusion:** Larger batch sizes → more compute-bound → better hardware utilization.

---

## 🎯 Where Used

| Concept | Applications |
|---------|-------------|
| Roofline Analysis | Performance optimization |
| FLOPs Counting | Model comparison |
| Memory Profiling | Batch size optimization |
| Arithmetic Intensity | Kernel optimization |

---

## 🗺️ Navigation

| ⬅️ Previous | 🏠 Home | ➡️ Next |
|:-----------:|:-------:|:-------:|
| [← Introduction](../01_introduction/README.md) | [Efficient ML](../README.md) | [Pruning & Sparsity I →](../03_pruning_sparsity_1/README.md) |

---
## 📚 References

| Type | Resource | Link |
|------|----------|------|
| 📄 | Roofline Model | [Berkeley](https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf) |
| 📄 | I/O Complexity (Hong & Kung) | [ACM](https://dl.acm.org/doi/10.1145/800076.802486) |
| 💻 | PyTorch Profiler | [PyTorch](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) |
| 🎥 | MIT 6.5940 TinyML | [Course](https://hanlab.mit.edu/courses/2024-fall-65940) |

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=1ABC9C&height=80&section=footer" width="100%"/>
</p>
