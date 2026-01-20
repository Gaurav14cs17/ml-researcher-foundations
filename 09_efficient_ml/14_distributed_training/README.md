<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=1ABC9C&height=120&section=header&text=Lecture%2014%20Distributed%20Training&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-09-1ABC9C?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

# Lecture 14: Distributed Training

[← Back to Course](../) | [← Previous](../13_on_device_training/) | [Next: Efficient Vision →](../15_efficient_vision_models/)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/ml-researcher-foundations/blob/main/09_efficient_ml/14_distributed_training/demo.ipynb) ← **Try the code!**

---

## 📺 Video Lecture

| Resource | Link |
|----------|------|
| 🎥 **Lecture Video** | [Watch on YouTube](https://www.youtube.com/watch?v=0fKrtyghN8s&list=PL80kAHvQbh-pT4lCkDT53zT8DKmhE0idB&index=14) |
| 📊 **Slides** | [MIT Course Page](https://hanlab.mit.edu/courses/2023-fall-65940) |
| ⏱️ **Duration** | ~90 minutes |

### 📝 Video Summary

This lecture covers **distributed training** for large models:

- **Why distributed**: Models too large for single GPU
- **Data parallelism**: Same model, different data batches
- **ZeRO**: Partitioning optimizer states, gradients, parameters
- **Tensor parallelism**: Splitting matrices across GPUs
- **Pipeline parallelism**: Splitting layers across GPUs
- **3D parallelism**: Combining all three for maximum scale

> 💡 *"Training GPT-3 required 175B parameters across thousands of GPUs—distributed training is essential for frontier models."* — Prof. Song Han

---

![Overview](overview.png)

## Why Distributed Training?

| Model | Parameters | Memory (FP16) | GPUs Needed |
|-------|------------|---------------|-------------|
| GPT-2 | 1.5B | 3GB | 1 |
| GPT-3 | 175B | 350GB | 44+ |
| PaLM | 540B | 1TB | 135+ |

---

## 📐 Mathematical Foundations & Proofs

### Data Parallelism

**Setup:** N GPUs, each with model replica.

**Forward:** Each GPU processes $B/N$ samples.

**Gradient aggregation:**

```math
g = \frac{1}{N} \sum_{i=1}^{N} g_i
```

**AllReduce operation:** Each GPU ends with same averaged gradient.

**Equivalence to large batch:**

Data parallel with N GPUs and batch B/N per GPU = single GPU with batch B.

**Proof:** 

```math
g_{DP} = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{B/N} \sum_{j \in \text{batch}_i} \nabla \mathcal{L}_j = \frac{1}{B} \sum_{j=1}^{B} \nabla \mathcal{L}_j = g_{single}
```

---

### ZeRO (Zero Redundancy Optimizer)

**Standard DDP memory per GPU:**

```math
M_{DDP} = M_{model} + M_{grad} + M_{opt}
```

All GPUs have full copies—redundant!

**ZeRO partitioning:**

| Stage | Partitioned | Memory/GPU |
|-------|-------------|------------|
| ZeRO-1 | Optimizer states | $M_{model} + M_{grad} + M_{opt}/N$ |
| ZeRO-2 | + Gradients | $M_{model} + M_{grad}/N + M_{opt}/N$ |
| ZeRO-3 | + Parameters | $(M_{model} + M_{grad} + M_{opt})/N$ |

**ZeRO-3 memory:**

```math
M_{ZeRO-3} = \frac{M_{total}}{N} + M_{activation}
```

**Linear scaling with N GPUs!**

---

### Tensor Parallelism

**Split weight matrices across GPUs:**

For $Y = XW$, partition $W$ column-wise:

```math
W = [W_1 | W_2 | ... | W_N]
```

Each GPU $i$ computes:

```math
Y_i = X W_i
```

**Concatenate results:**

```math
Y = [Y_1 | Y_2 | ... | Y_N]
```

**For MLP in transformer:**

Column parallel (first linear):

```math
Y = XW_1 = [XW_1^{(1)} | XW_1^{(2)}]
```

Row parallel (second linear):

```math
Z = YW_2 = Y^{(1)}W_2^{(1)} + Y^{(2)}W_2^{(2)}
```

**AllReduce after row parallel.**

---

### Pipeline Parallelism

**Partition layers across GPUs:**

GPU 0: Layers 0-10
GPU 1: Layers 11-20
GPU 2: Layers 21-30

**Micro-batching to reduce bubble:**

Without micro-batching:
```
GPU0: [----F----][----B----]
GPU1:            [----F----][----B----]
GPU2:                       [----F----][----B----]
```

Bubble time = 2/3 of total.

With K micro-batches:

```math
\text{Bubble fraction} = \frac{P-1}{K + P - 1}
```

where P = pipeline stages.

**For K >> P:** Bubble → 0.

---

### 3D Parallelism

**Combining all three:**

- **Data parallel:** Across nodes (slow network)
- **Tensor parallel:** Within node (fast NVLink)
- **Pipeline parallel:** Across nodes (overlaps communication)

**Total parallelism:**

```math
N_{total} = N_{DP} \times N_{TP} \times N_{PP}
```

**Memory per GPU:**

```math
M = \frac{M_{model}}{N_{TP} \times N_{PP}} + M_{activation}
```

---

### Communication Costs

**Data parallel AllReduce:**

```math
T_{allreduce} = 2(N-1)/N \cdot M_{grad} / BW
```

Ring AllReduce achieves bandwidth-optimal $O(1)$ w.r.t. N GPUs.

**Tensor parallel AllReduce:**

```math
T_{TP} = M_{activation} / BW
```

Called 2× per transformer layer.

**Pipeline parallelism:**

```math
T_{PP} = (K-1) \times M_{activation} / BW
```

Communication overlapped with compute.

---

### Scaling Laws

**Chinchilla optimal training:**

```math
C = 6ND
```

where:
- $C$ = compute (FLOP)
- $N$ = parameters
- $D$ = training tokens

**Optimal ratio:**

```math
D \approx 20N
```

**Derivation:**

Loss scales as:

```math
L(N, D) \propto N^{-\alpha} + D^{-\beta}
```

For fixed compute $C = 6ND$, minimize loss:

```math
\frac{\partial L}{\partial N} = 0 \implies D/N = \text{const} \approx 20
```

---

## 🧮 Key Derivations

### AllReduce Complexity

**Naive AllReduce:** $O(N)$ communication rounds.

**Ring AllReduce:** $O(1)$ with respect to N.

Algorithm:
1. Reduce-scatter: Each GPU sends/receives $M/N$ data $N-1$ times
2. All-gather: Each GPU sends/receives $M/N$ data $N-1$ times

Total: $2(N-1) \times M/N \approx 2M$ data transferred.

---

### Megatron-LM Attention Parallelism

**Q, K, V projections (column parallel):**

```math
[Q, K, V] = X[W_Q, W_K, W_V]
```

Split $W_Q, W_K, W_V$ across heads → each GPU handles subset of heads.

**Output projection (row parallel):**

```math
O = \text{Concat}(head_1, ..., head_H) W_O
```

**Communication:** Single AllReduce after attention block.

---

## 🎯 Where Used

| Concept | Applications |
|---------|-------------|
| DDP | Standard distributed training |
| ZeRO | Large model training |
| Tensor Parallel | LLM training (GPT, LLaMA) |
| Pipeline Parallel | Very large models |

---

## 🗺️ Navigation

| ⬅️ Previous | 🏠 Home | ➡️ Next |
|:-----------:|:-------:|:-------:|
| [← On-Device Training](../13_on_device_training/README.md) | [Efficient ML](../README.md) | [Efficient Vision Models →](../15_efficient_vision_models/README.md) |

---

## 📚 References

| Type | Resource | Link |
|------|----------|------|
| 📄 | ZeRO | [arXiv](https://arxiv.org/abs/1910.02054) |
| 📄 | Megatron-LM | [arXiv](https://arxiv.org/abs/1909.08053) |
| 📄 | GPipe | [arXiv](https://arxiv.org/abs/1811.06965) |
| 📄 | Chinchilla | [arXiv](https://arxiv.org/abs/2203.15556) |
| 🎥 | MIT 6.5940 TinyML | [Course](https://hanlab.mit.edu/courses/2024-fall-65940) |

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=1ABC9C&height=80&section=footer" width="100%"/>
</p>
