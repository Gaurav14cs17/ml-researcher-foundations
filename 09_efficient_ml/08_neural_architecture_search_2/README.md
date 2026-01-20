<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=1ABC9C&height=120&section=header&text=Lecture%208%20Neural%20Architecture%20Search%20II&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-09-1ABC9C?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

# Lecture 8: Neural Architecture Search (Part II)

[← Back to Course](../) | [← Previous](../07_neural_architecture_search_1/) | [Next: Distillation →](../09_knowledge_distillation/)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/ml-researcher-foundations/blob/main/09_efficient_ml/08_neural_architecture_search_2/demo.ipynb) ← **Try the code!**

---

## 📺 Video Lecture

| Resource | Link |
|----------|------|
| 🎥 **Lecture Video** | [Watch on YouTube](https://www.youtube.com/watch?v=RUaFUo7rOXk&list=PL80kAHvQbh-pT4lCkDT53zT8DKmhE0idB&index=8) |
| 📊 **Slides** | [MIT Course Page](https://hanlab.mit.edu/courses/2023-fall-65940) |
| ⏱️ **Duration** | ~90 minutes |

### 📝 Video Summary

This lecture covers **hardware-aware NAS** and efficient deployment:

- **Hardware-aware NAS**: Optimizing for actual device latency, not just FLOPs
- **ProxylessNAS**: Train directly on target hardware metrics
- **Once-for-All (OFA)**: Train one network containing many sub-networks
- **EfficientNet**: Compound scaling of depth, width, and resolution
- **MnasNet**: Multi-objective optimization for mobile deployment
- **Practical deployment**: From search to real-world deployment

> 💡 *"FLOPs don't equal speed—you must optimize for the actual target hardware."* — Prof. Song Han

---

![Overview](overview.png)

## Hardware-Aware NAS

Standard NAS optimizes accuracy. **Hardware-aware NAS** optimizes for real hardware:

```
Objective: max Accuracy
Subject to: Latency ≤ 20ms on iPhone
            Memory ≤ 500MB

```

---

## Once-for-All (OFA)

Train ONE network that contains MANY sub-networks:

```
Full Network: depth=20, width=1.0, resolution=224

Sub-networks:
- depth=12, width=0.5, resolution=160 (mobile)
- depth=16, width=0.75, resolution=192 (tablet)
- depth=20, width=1.0, resolution=224 (server)

```

---

## EfficientNet: Compound Scaling

Instead of NAS, scale depth/width/resolution together:

```
depth = α^φ
width = β^φ  
resolution = γ^φ

α × β² × γ² ≈ 2 (FLOPS constraint)

```

---

## 📐 Mathematical Foundations & Proofs

### Hardware-Aware Objective

**Multi-objective NAS:**

```math
\max_\alpha \text{Acc}(\alpha) \quad \text{s.t.} \quad \text{Latency}(\alpha) \leq T

```

**Scalarized form:**

```math
\max_\alpha \text{Acc}(\alpha) \times \left(\frac{\text{Latency}(\alpha)}{T}\right)^w

```

where \( w < 0 \) penalizes exceeding target \( T \).

**Latency estimation:**

```math
\text{Latency}(\alpha) = \sum_{l=1}^L \text{Lat}(o_l, h_l, w_l, c_l)

```

Use **lookup tables** for per-operation latency on target hardware.

---

### ProxylessNAS: Differentiable Latency

Make latency differentiable for gradient-based search:

**Expected latency:**

```math
\mathbb{E}[\text{Lat}] = \sum_{o \in \mathcal{O}} p(o) \cdot \text{Lat}(o)

```

where \( p(o) = \text{softmax}(\alpha)_o \).

**Gradient:**

```math
\frac{\partial \mathbb{E}[\text{Lat}]}{\partial \alpha_i} = \sum_{o} \text{Lat}(o) \cdot \frac{\partial p(o)}{\partial \alpha_i}

```

Using softmax derivative:

```math
\frac{\partial p(o)}{\partial \alpha_i} = p(o)(\mathbb{1}[o=i] - p(i))

```

---

### EfficientNet Compound Scaling

**Hypothesis:** Depth, width, and resolution should scale together.

**Formulation:**

```math
d = \alpha^\phi, \quad w = \beta^\phi, \quad r = \gamma^\phi

```

**Constraint:** FLOPs scale as:

```math
\text{FLOPs} \propto d \cdot w^2 \cdot r^2

```

For constant FLOPs increase of \( 2^\phi \):

```math
\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2

```

**Grid search** finds \( \alpha = 1.2, \beta = 1.1, \gamma = 1.15 \).

**Proof of optimality:**

For a given compute budget, jointly scaling all dimensions outperforms scaling any single dimension:

```math
\text{Acc}(\alpha^\phi d_0, \beta^\phi w_0, \gamma^\phi r_0) > \text{Acc}(2^\phi d_0, w_0, r_0)

```

This is empirically validated across multiple architectures.

---

### Once-for-All Network Training

**Progressive shrinking:**

1. **Phase 1:** Train largest network

```math
\min_W \mathcal{L}(f_{max}(x; W))

```

2. **Phase 2:** Support elastic depth (sample random depths)

```math
\min_W \mathbb{E}_{d \sim \mathcal{D}}[\mathcal{L}(f_d(x; W))]

```

3. **Phase 3:** Support elastic width (sample random widths)

```math
\min_W \mathbb{E}_{d,w \sim \mathcal{D} \times \mathcal{W}}[\mathcal{L}(f_{d,w}(x; W))]

```

4. **Phase 4:** Support elastic resolution

**Knowledge distillation during training:**

```math
\mathcal{L}_{KD} = \text{KL}(f_{small}(x) \| f_{large}(x))

```

Larger sub-networks teach smaller ones.

---

### Elastic Dimensions

**Elastic depth:** Sample \( d \in \{d_{min}, ..., d_{max}\} \) layers.

**Elastic width:** Sample channels \( c \in \{c_{min}, ..., c_{max}\} \).

**Elastic kernel:** Sample kernel size \( k \in \{3, 5, 7\} \).

**Weight reuse for elastic width:**

```math
W_{small}[:c, :c'] = W_{full}[:c, :c']

```

Take the first \( c \) output channels and \( c' \) input channels.

**Weight transformation for elastic kernel:**

```math
W_{3\times3} = \text{CenterCrop}(W_{5\times5})

```

or use learnable transformation matrices.

---

### Subnet Search (Deployment)

After OFA training, find optimal subnet for target device:

**Search problem:**

```math
\max_{d,w,k,r} \text{Acc}(d,w,k,r; W_{OFA})
\text{s.t.} \quad \text{Latency}(d,w,k,r) \leq T

```

**Evolutionary search:**
1. Initialize population of random configurations
2. Evaluate accuracy using trained OFA weights (no retraining!)
3. Mutate top-k configurations
4. Repeat until convergence

**Search time:** ~3 hours on CPU (vs. days for traditional NAS).

---

### MnasNet Reward Function

**Multi-objective reward:**

```math
R(\alpha) = \text{Acc}(\alpha) \times \left[\frac{\text{Lat}(\alpha)}{T}\right]^w

```

where:

```math
w = \begin{cases} 
\beta & \text{if } \text{Lat}(\alpha) \leq T \\
\alpha & \text{if } \text{Lat}(\alpha) > T
\end{cases}

```

Typically \( \alpha = \beta = -0.07 \).

**Pareto frontier:** The set of architectures where no architecture dominates another in both accuracy and latency.

---

### Latency Prediction

**Lookup table approach:**

For each operation type \( o \) and configuration \( (c_{in}, c_{out}, k, H, W) \):

```math
\text{Lat}(o, c_{in}, c_{out}, k, H, W) = \text{LUT}[o, c_{in}, c_{out}, k, H, W]

```

**Neural predictor approach:**

Train a small MLP to predict latency:

```math
\hat{\text{Lat}}(\alpha) = g(\alpha; \theta)

```

More flexible, handles unseen configurations.

---

## 🧮 Key Derivations

### Cost of OFA vs Individual Training

**Individual training:**

```math
\text{Cost}_{ind} = N_{devices} \times \text{Cost}_{single}

```

For 10 devices: 10× training cost.

**OFA training:**

```math
\text{Cost}_{OFA} = 1 \times \text{Cost}_{supernet} + N_{devices} \times \text{Cost}_{search}

```

Since \( \text{Cost}_{search} \ll \text{Cost}_{single} \):

```math
\text{Cost}_{OFA} \approx 1.2 \times \text{Cost}_{single}

```

**Savings:** 8× for 10 target devices.

---

### Architecture Space Coverage

OFA network with:
- 5 depth choices
- 4 width choices per layer
- 3 kernel choices per layer
- 5 resolution choices

**Total sub-networks:**

```math

|\mathcal{A}| = 5 \times 4^{20} \times 3^{20} \times 5 \approx 10^{19}

```

All accessible with single training!

---

## 🎯 Where Used

| Concept | Applications |
|---------|-------------|
| ProxylessNAS | Mobile-optimized models |
| Once-for-All | Multi-device deployment |
| EfficientNet | State-of-the-art CNNs |
| MnasNet | Google mobile models |

---

## 🗺️ Navigation

| ⬅️ Previous | 🏠 Home | ➡️ Next |
|:-----------:|:-------:|:-------:|
| [← NAS I](../07_neural_architecture_search_1/README.md) | [Efficient ML](../README.md) | [Knowledge Distillation →](../09_knowledge_distillation/README.md) |

---
## 📚 References

| Type | Resource | Link |
|------|----------|------|
| 📄 | ProxylessNAS | [arXiv](https://arxiv.org/abs/1812.00332) |
| 📄 | Once-for-All | [arXiv](https://arxiv.org/abs/1908.09791) |
| 📄 | MnasNet | [arXiv](https://arxiv.org/abs/1807.11626) |
| 📄 | EfficientNet | [arXiv](https://arxiv.org/abs/1905.11946) |
| 🎥 | MIT 6.5940 TinyML | [Course](https://hanlab.mit.edu/courses/2024-fall-65940) |

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=1ABC9C&height=80&section=footer" width="100%"/>
</p>
