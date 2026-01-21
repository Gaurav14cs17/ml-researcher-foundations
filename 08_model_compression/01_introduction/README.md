<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=E74C3C&height=120&section=header&text=Introduction%20to%20Model%20Compression&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-08-E74C3C?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 📐 Mathematical Foundations

### 1. Compression Ratio Definition

**Definition:** The compression ratio quantifies how much smaller a compressed model is compared to the original:

$$CR = \frac{|M_{original}|}{|M_{compressed}|} = \frac{\sum_{l} |W_l| \cdot b_{original}}{\sum_{l} |W_l| \cdot b_{compressed}}$$

Where:

- $|W_l|$ = number of weights in layer $l$

- $b$ = bits per weight

**Example:**

```
Original: 340M params × 32 bits = 10.88 Gbits = 1.36 GB
INT8: 340M params × 8 bits = 2.72 Gbits = 340 MB
CR = 1360 / 340 = 4×

```

### 2. Compression-Accuracy Trade-off (Formal Definition)

**Optimization Problem:**

$$\min_{M_c \in \mathcal{M}} \text{Size}(M_c) \quad \text{subject to} \quad \mathcal{L}(M_c) - \mathcal{L}(M) \leq \epsilon$$

Where:

- $\mathcal{M}$ = space of compressed models

- $\mathcal{L}$ = loss function (lower is better)

- $\epsilon$ = acceptable accuracy degradation

**Theorem (Pareto Optimality):**
The set of optimal compressions forms a Pareto frontier. A model $M_c^*$ is Pareto optimal if there exists no $M_c$ such that:

1. $\text{Size}(M_c) \leq \text{Size}(M_c^*)$ AND

2. $\mathcal{L}(M_c) \leq \mathcal{L}(M_c^*)$
with at least one strict inequality.

### 3. Information-Theoretic Bounds

**Shannon's Source Coding Theorem:**

The minimum average description length for weights is bounded by their entropy:

$$H(W) \leq \mathbb{E}[\text{bits per weight}]$$

Where entropy is:

$$H(W) = -\sum_{w} p(w) \log_2 p(w)$$

**For continuous weights (differential entropy):**

$$h(W) = -\int p(w) \log_2 p(w) dw$$

**Gaussian Distribution Case:**

$$h(W) = \frac{1}{2} \log_2(2\pi e \sigma^2) \approx 4.13 \text{ bits (for } \sigma = 1)$$

This theoretical limit explains why 4-bit quantization often works!

### 4. Memory and Compute Complexity

**Memory Footprint:**

$$\text{Memory}(M) = \sum_{l=1}^{L} |W_l| \cdot \frac{b_l}{8} \text{ bytes}$$

**Inference FLOPs:**

$$\text{FLOPs} = \sum_{l=1}^{L} 2 \cdot |W_l| \cdot \text{input-size}_l$$

**Roofline Model (Memory-Bound vs Compute-Bound):**

$$\text{Achieved FLOPS} = \min\left(\text{Peak FLOPS}, \text{Bandwidth} \times \text{Arithmetic Intensity}\right)
\text{Arithmetic Intensity} = \frac{\text{FLOPs}}{\text{Bytes Accessed}}$$

For LLMs: Usually memory-bound → compression helps!

### 5. Compression Taxonomy

| Method | Mathematical Principle | Compression Factor |
|--------|----------------------|-------------------|
| Quantization | $W \in \mathbb{R} \to W_q \in \{0,1,...,2^b-1\}$ | $32/b$ |
| Pruning | $W \to W \odot M$ where $M \in \{0,1\}$ | $1/\text{sparsity}$ |
| Low-Rank | $W \approx UV^T$ where $U,V \in \mathbb{R}^{n \times r}$ | $\frac{mn}{r(m+n)}$ |
| Distillation | $f_{student} \approx f_{teacher}$ | Architecture-dependent |

---

## 📂 Topics

| File | Topic | Key Concepts |
|------|-------|--------------|
| [../02_quantization/](../02_quantization/) | Quantization | INT8, INT4, GPTQ, AWQ |
| [../03_pruning/](../03_pruning/) | Pruning | Magnitude, Structured, Lottery Ticket |
| [../04_knowledge_distillation/](../04_knowledge_distillation/) | Knowledge Distillation | Teacher-Student |
| [../08_peft/](../08_peft/) | PEFT | LoRA, Adapters |

---

## 🎯 The Problem

```
Modern AI models are HUGE:

GPT-4:        ~1.8 Trillion parameters → ~3.6 TB (FP16)
LLaMA-70B:    70 Billion parameters   → 140 GB (FP16)
Stable Diff:  1 Billion parameters    → 4 GB
BERT-Large:   340 Million parameters  → 1.3 GB

Problems:
• Won't fit in GPU memory
• Too slow for real-time
• Too expensive to serve
• Can't run on mobile/edge

```

**Memory Calculation:**

$$\text{GPU Memory} \geq \text{Model} + \text{Activations} + \text{Gradients} + \text{Optimizer States}$$

For training with Adam:

$$\text{Total} = P \times (2 + 2 + 4 + 8) = 16P \text{ bytes}$$

For 7B model: $7 \times 10^9 \times 16 = 112$ GB just for training!

---

## 💡 The Solution: Compression

```
Original Model          Compressed Model
+----------------+      +----------------+

|                |      |                |
|   340M params  | -->  |   66M params   |  (DistilBERT)

|   1.3 GB       |      |   260 MB       |
|   100ms        |      |   30ms         |
|                |      |                |
+----------------+      +----------------+
    BERT-Large              DistilBERT

Same task, 5x smaller, 3x faster, <1% accuracy drop!

```

**Compression Stack:**

$$M_{compressed} = Q(P(D(M_{original})))$$

Where:

- $D$ = Distillation (architecture change)

- $P$ = Pruning (remove weights)

- $Q$ = Quantization (reduce precision)

---

## ⚖️ Trade-offs Visualization

<img src="./images/tradeoffs.svg" width="100%">

**Empirical Scaling Law for Compression:**

$$\mathcal{L}_{compressed} \approx \mathcal{L}_{original} + \alpha \cdot CR^{\beta}$$

Where:

- $\alpha, \beta$ depend on method and model

- Typical: $\beta \in [0.5, 2]$

---

## 🔗 Where This Topic Is Used

| Topic | How Introduction Concepts Apply |
|-------|--------------------------------|
| **Production Systems** | Understand cost-accuracy tradeoffs |
| **Mobile Deployment** | Know why compression is needed |
| **LLM Serving** | Quantization reduces serving cost |
| **Fine-tuning** | LoRA makes fine-tuning affordable |

---

## 📚 References & Resources

### 📄 Survey Papers

| Type | Title | Link |
|------|-------|------|
| 📄 | Model Compression Survey | [arXiv](https://arxiv.org/abs/2103.13630) |
| 📄 | Efficient Deep Learning | [arXiv](https://arxiv.org/abs/2106.08962) |
| 📄 | LLM Compression Survey | [arXiv](https://arxiv.org/abs/2308.07633) |
| 🇨🇳 | 为什么需要模型压缩 | [知乎](https://zhuanlan.zhihu.com/p/628120082) |
| 🇨🇳 | 大模型部署指南 | [机器之心](https://www.jiqizhixin.com/articles/2023-08-31-2) |

---

⬅️ [Back: Model Compression](../README.md) | ➡️ [Next: Quantization](../02_quantization/README.md)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=E74C3C&height=80&section=footer" width="100%"/>
</p>
