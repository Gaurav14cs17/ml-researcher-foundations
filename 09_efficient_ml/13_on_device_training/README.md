<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=1ABC9C&height=120&section=header&text=Lecture%2013%20On-Device%20Training&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-09-1ABC9C?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

# Lecture 13: On-Device Training

[← Back to Course](../) | [← Previous](../12_efficient_training/) | [Next: Distributed Training →](../14_distributed_training/)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/ml-researcher-foundations/blob/main/09_efficient_ml/13_on_device_training/demo.ipynb) ← **Try the code!**

---

## 📺 Video Lecture

| Resource | Link |
|----------|------|
| 🎥 **Lecture Video** | [Watch on YouTube](https://www.youtube.com/watch?v=0fKrtyghN8s&list=PL80kAHvQbh-pT4lCkDT53zT8DKmhE0idB&index=13) |
| 📊 **Slides** | [MIT Course Page](https://hanlab.mit.edu/courses/2023-fall-65940) |
| ⏱️ **Duration** | ~90 minutes |

### 📝 Video Summary

This lecture covers **training on edge devices**:

- **Why on-device training**: Privacy, personalization, offline adaptation
- **Memory challenges**: Training needs 10-20× more memory than inference
- **TinyTL**: Bias-only training for minimal memory overhead
- **Sparse updates**: Only updating subset of parameters
- **Federated learning**: Training across devices without sharing data
- **Practical considerations**: Battery, thermal, and data challenges

> 💡 *"On-device training enables privacy-preserving personalization—your model improves without your data leaving the device."* — Prof. Song Han

---

![Overview](overview.png)

## Why On-Device Training?

| Benefit | Example |
|---------|---------|
| **Privacy** | Personal photos never leave phone |
| **Personalization** | Adapt to user's voice |
| **Continuous learning** | Improve from new data |
| **Offline** | No cloud connection needed |

---

## 📐 Mathematical Foundations & Proofs

### Training vs Inference Memory

**Inference:**

$$
M_{inf} = M_{weights} + M_{act}
$$

**Training:**

$$
M_{train} = M_{weights} + M_{grad} + M_{opt} + M_{act} + M_{act\_backward}
$$

**Ratio:**

$$
\frac{M_{train}}{M_{inf}} = 1 + 1 + 2 + \frac{M_{act\_backward}}{M_{act}} \approx 10-20\times
$$

---

### TinyTL: Bias-Only Training

**Full fine-tuning gradient:**

$$
\nabla_W \mathcal{L} = \frac{\partial \mathcal{L}}{\partial Y} X^T
$$

Requires storing $X$ (full activation).

**Bias gradient:**

$$
\nabla_b \mathcal{L} = \frac{\partial \mathcal{L}}{\partial Y} \mathbf{1}
$$

Only requires output gradient, not input activation!

**Memory savings:**

$$
\frac{M_{bias}}{M_{full}} = \frac{d_{out}}{d_{in} \times d_{out}} = \frac{1}{d_{in}}
$$

For $d_{in} = 1024$: ~1000× reduction in trainable parameters, ~10× reduction in activation memory.

---

### Sparse Update Selection

**Gradient importance score:**

$$
I_l = \left\| \frac{\partial \mathcal{L}}{\partial W_l} \right\|_F
$$

**Update only top-k layers:**

$$
\mathcal{L}_{update} = \{l : I_l > \tau\}
$$

**Memory reduction:**

$$
\frac{M_{sparse}}{M_{full}} = \frac{k}{L}
$$

---

### Federated Learning Mathematics

**FedAvg algorithm:**

1. Server broadcasts global model $w_t$
2. Each client $k$ trains locally:

$$
w_{t+1}^k = w_t - \eta \sum_{i=1}^{E} \nabla \mathcal{L}_k(w_t^{(i)})
$$

3. Server aggregates:

$$
w_{t+1} = \sum_{k=1}^{K} \frac{n_k}{n} w_{t+1}^k
$$

where $n_k$ is samples on client $k$, $n = \sum_k n_k$.

**Convergence (convex case):**

$$
\mathbb{E}[\mathcal{L}(w_T)] - \mathcal{L}(w^*) \leq O\left(\frac{1}{\sqrt{KT}}\right)
$$

---

### Communication Efficiency

**Gradient compression:**
- **Top-k sparsification:** Only send largest gradients
- **Quantization:** Reduce precision of gradients

**Error feedback:**

$$
e_{t+1} = g_t - Q(g_t + e_t)
\tilde{g}_t = Q(g_t + e_t)
$$

Accumulate quantization error, add to next gradient.

**Convergence preserved** with error feedback (Stich et al., 2018).

---

### Differential Privacy in Federated Learning

**DP-SGD:**

$$
g_t^{DP} = \text{clip}(g_t, C) + \mathcal{N}(0, \sigma^2 C^2 I)
$$

**Privacy budget:**

$$
(\epsilon, \delta)\text{-DP}
$$

Composition over rounds:

$$
\epsilon_{total} = \sqrt{2T \ln(1/\delta)} \cdot \epsilon_{step}
$$

---

## 🧮 Key Derivations

### Activation Memory for Bias-Only Training

**Full training (storing input):**

$$
M_{act}^{full} = B \times L \times d_{in} \times d_{out}
$$

**Bias training (storing output only):**

$$
M_{act}^{bias} = B \times L \times d_{out}
$$

**Reduction:**

$$
\frac{M_{act}^{bias}}{M_{act}^{full}} = \frac{1}{d_{in}}
$$

---

### Energy Consumption

**Training energy:**

$$
E_{train} = P \times t = P \times \frac{B \times \text{epochs} \times \text{FLOPs\_per\_sample}}{\text{FLOPS}}
$$

**Battery constraint:**

$$
E_{train} < E_{battery} \times \text{fraction}
$$

Typically limit training to 10% of battery.

---

## 🎯 Where Used

| Concept | Applications |
|---------|-------------|
| TinyTL | Mobile personalization |
| Federated Learning | Privacy-preserving ML |
| Sparse Update | Edge fine-tuning |
| DP-SGD | Private federated learning |

---

## 🗺️ Navigation

| ⬅️ Previous | 🏠 Home | ➡️ Next |
|:-----------:|:-------:|:-------:|
| [← Efficient Training](../12_efficient_training/README.md) | [Efficient ML](../README.md) | [Distributed Training →](../14_distributed_training/README.md) |

---

## 📚 References

| Type | Resource | Link |
|------|----------|------|
| 📄 | TinyTL | [arXiv](https://arxiv.org/abs/2007.11622) |
| 📄 | Federated Learning | [arXiv](https://arxiv.org/abs/1602.05629) |
| 📄 | On-Device Training Under 256KB | [arXiv](https://arxiv.org/abs/2206.15472) |
| 📄 | DP-SGD | [arXiv](https://arxiv.org/abs/1607.00133) |
| 🎥 | MIT 6.5940 TinyML | [Course](https://hanlab.mit.edu/courses/2024-fall-65940) |

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=1ABC9C&height=80&section=footer" width="100%"/>
</p>
