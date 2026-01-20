<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=1ABC9C&height=120&section=header&text=Lecture%203%20Pruning%20%26%20Sparsity%20Part%20I&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-09-1ABC9C?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

# Lecture 3: Pruning & Sparsity (Part I)

[← Back to Course](../) | [← Previous](../02_basics/) | [Next: Pruning II →](../04_pruning_sparsity_2/)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/ml-researcher-foundations/blob/main/09_efficient_ml/03_pruning_sparsity_1/demo.ipynb) ← **Try the code!**

---

## 📺 Video Lecture

| Resource | Link |
|----------|------|
| 🎥 **Lecture Video** | [Watch on YouTube](https://www.youtube.com/watch?v=sZzc6tAtTrM&list=PL80kAHvQbh-pT4lCkDT53zT8DKmhE0idB&index=3) |
| 📊 **Slides** | [MIT Course Page](https://hanlab.mit.edu/courses/2023-fall-65940) |
| ⏱️ **Duration** | ~90 minutes |

### 📝 Video Summary

This lecture introduces **neural network pruning** as a fundamental compression technique:

- **Why pruning works**: Neural networks are over-parameterized; many weights are redundant
- **Pruning pipeline**: Train → Prune → Fine-tune → Deploy
- **Pruning criteria**: Magnitude-based, gradient-based, and Taylor expansion methods
- **Structured vs Unstructured**: Trade-offs between compression ratio and hardware efficiency
- **Iterative pruning**: Why gradual pruning outperforms one-shot approaches
- **Seminal results**: 90%+ sparsity with <1% accuracy drop on ImageNet

> 💡 *"Not all weights are created equal—many can be removed with minimal impact on accuracy."* — Prof. Song Han

---

![Overview](overview.png)

## What is Pruning?

**Pruning** removes unnecessary weights from a neural network to make it smaller and faster.

> "Not all weights are created equal — many can be removed with minimal impact on accuracy."

---

## The Pruning Pipeline

```
Train Full Model → Prune Weights → Fine-tune → Deploy
     100%              30%           30%        30%
```

---

## Types of Pruning

### 1. Unstructured Pruning
Remove individual weights anywhere in the network.

```python
# Before: Dense weight matrix
W = [[0.1, 0.5, 0.2],
     [0.8, 0.3, 0.1],
     [0.2, 0.9, 0.4]]

# After: Sparse (zeros scattered)
W = [[0.1, 0.0, 0.2],
     [0.8, 0.0, 0.0],
     [0.0, 0.9, 0.4]]
```

**Pros:** High compression ratios
**Cons:** Needs sparse hardware/libraries

### 2. Structured Pruning
Remove entire channels, filters, or attention heads.

```python
# Before: 64 channels
# After: 32 channels (remove entire channels)
```

**Pros:** Works on standard hardware
**Cons:** Lower compression ratios

---

## Pruning Criteria

How do we decide which weights to remove?

| Criterion | Formula | Intuition |
|-----------|---------|-----------|
| **Magnitude** | \|w\| | Small weights matter less |
| **Gradient** | \|∂L/∂w\| | Low gradient = low impact |
| **Taylor** | \|w × ∂L/∂w\| | Combines both |

---

## Magnitude Pruning

The simplest and most common approach:

```python
def magnitude_prune(weights, sparsity=0.9):
    """Remove smallest 90% of weights"""
    threshold = np.percentile(np.abs(weights), sparsity * 100)
    mask = np.abs(weights) > threshold
    return weights * mask
```

---

## Iterative Pruning

Better results come from pruning gradually:

```
Iteration 1: 0% → 50% sparse, fine-tune
Iteration 2: 50% → 75% sparse, fine-tune
Iteration 3: 75% → 90% sparse, fine-tune
```

This works better than pruning 90% all at once!

---

## Results on ImageNet

| Model | Pruning Ratio | Top-1 Accuracy |
|-------|--------------|----------------|
| AlexNet (original) | 0% | 57.2% |
| AlexNet (pruned) | 89% | 57.2% |
| VGG-16 (original) | 0% | 68.5% |
| VGG-16 (pruned) | 92% | 68.3% |

**Key Insight:** Can remove 90%+ weights with <1% accuracy drop!

---

## Key Paper

📄 **[Learning both Weights and Connections for Efficient Neural Networks](https://arxiv.org/abs/1506.02626)** (Han et al., 2015)

This paper introduced the standard pruning pipeline used today.

---

## 📐 Mathematical Foundations & Proofs

### Magnitude-Based Pruning Criterion

The magnitude-based importance score:

```math
I(w_{ij}) = |w_{ij}|
```

Pruning mask:

```math
m_{ij} = \mathbb{1}[|w_{ij}| > \tau]
```

where \( \tau \) is the threshold for target sparsity \( s \):

```math
\tau = \text{Percentile}(|W|, 100 \times s)
```

**Theoretical Justification:**

For a linear layer \( y = Wx + b \), the contribution of weight \( w_{ij} \) to output:

```math
\Delta y_i = w_{ij} \cdot x_j
```

Expected contribution magnitude:

```math
\mathbb{E}[|\Delta y_i|] = |w_{ij}| \cdot \mathbb{E}[|x_j|]
```

If inputs are normalized, \( \mathbb{E}[|x_j|] \approx \text{const} \), so:

```math
\text{Importance} \propto |w_{ij}|
```

---

### Taylor Expansion Criterion

First-order Taylor approximation of loss change when pruning weight \( w \):

```math
\Delta \mathcal{L}(w) = \mathcal{L}(w=0) - \mathcal{L}(w) \approx -\frac{\partial \mathcal{L}}{\partial w} \cdot w
```

**Proof:**

Taylor expansion around current weight value:

```math
\mathcal{L}(w + \Delta w) \approx \mathcal{L}(w) + \frac{\partial \mathcal{L}}{\partial w} \Delta w + \frac{1}{2} \frac{\partial^2 \mathcal{L}}{\partial w^2} (\Delta w)^2
```

Setting \( w \to 0 \) means \( \Delta w = -w \):

```math
\mathcal{L}(0) \approx \mathcal{L}(w) - \frac{\partial \mathcal{L}}{\partial w} \cdot w
```

Therefore:

```math
\Delta \mathcal{L} = \mathcal{L}(0) - \mathcal{L}(w) \approx -\frac{\partial \mathcal{L}}{\partial w} \cdot w
```

Importance score:

```math
I(w) = \left| w \cdot \frac{\partial \mathcal{L}}{\partial w} \right|
```

---

### Second-Order (Optimal Brain Damage) Criterion

Using second-order Taylor expansion:

```math
\Delta \mathcal{L}(w) \approx -g_w \cdot w + \frac{1}{2} H_{ww} \cdot w^2
```

where \( g_w = \frac{\partial \mathcal{L}}{\partial w} \) and \( H_{ww} = \frac{\partial^2 \mathcal{L}}{\partial w^2} \).

At a local minimum, \( g_w \approx 0 \), so:

```math
\Delta \mathcal{L}(w) \approx \frac{1}{2} H_{ww} \cdot w^2
```

**Optimal Brain Damage** (LeCun et al., 1990): Prune weights with smallest \( H_{ww} \cdot w^2 \).

---

### Structured Pruning: Filter Importance

For a convolutional filter \( F_i \in \mathbb{R}^{C_{in} \times k \times k} \):

**L1-norm criterion:**

```math
I(F_i) = \sum_{c,h,w} |F_{i,c,h,w}|
```

**L2-norm criterion:**

```math
I(F_i) = \sqrt{\sum_{c,h,w} F_{i,c,h,w}^2}
```

**Geometric median criterion** (more robust):

```math
I(F_i) = \sum_{j \neq i} \|F_i - F_j\|_2
```

Filters similar to others are more redundant → prune them.

---

### Sparsity-Accuracy Trade-off

Empirical observation follows a characteristic curve:

```math
\text{Accuracy}(s) \approx \text{Acc}_0 - \alpha \cdot \exp\left(\frac{s - s_0}{\beta}\right)
```

where:
- \( s \) = sparsity level
- \( s_0 \) = critical sparsity (where accuracy starts dropping)
- \( \alpha, \beta \) = model-specific constants

**Key insight:** Accuracy is relatively stable until critical sparsity, then drops rapidly.

---

### Iterative Pruning Schedule

**Cubic sparsity schedule** (commonly used):

```math
s_t = s_f + (s_0 - s_f)\left(1 - \frac{t - t_0}{n\Delta t}\right)^3
```

where:
- \( s_0 \) = initial sparsity (usually 0)
- \( s_f \) = final target sparsity
- \( t_0 \) = start step
- \( n \) = number of pruning steps
- \( \Delta t \) = pruning interval

**Why cubic?** Removes more weights early (when easier to recover) and fewer later (when more critical).

---

## 🧮 Key Derivations

### Compression Ratio Calculation

For unstructured sparsity \( s \):

```math
\text{Compression Ratio} = \frac{1}{1-s}
```

At 90% sparsity: CR = 10×

**With sparse storage (CSR format):**

```math
\text{Memory} = \text{nnz} \times (\text{value\_size} + \text{index\_size})
```

Actual compression:

```math
\text{CR}_{actual} = \frac{n \times \text{value\_size}}{\text{nnz} \times (\text{value\_size} + \text{index\_size})}
```

For 90% sparsity with FP32 values and INT32 indices:

```math
\text{CR}_{actual} = \frac{n \times 4}{0.1n \times (4 + 4)} = \frac{4}{0.8} = 5\times
```

---

### Gradient Flow Through Sparse Networks

During fine-tuning with pruning mask \( M \):

**Forward:** \( Y = (M \odot W) X \)

**Backward:**

```math
\frac{\partial \mathcal{L}}{\partial W} = M \odot \left(\frac{\partial \mathcal{L}}{\partial Y} X^T\right)
```

Gradients only flow through unpruned weights.

---

## 🎯 Where Used

| Concept | Applications |
|---------|-------------|
| Magnitude Pruning | Edge deployment, Mobile inference |
| Structured Pruning | CNN acceleration on standard hardware |
| Iterative Pruning | Training efficient models from scratch |
| Taylor Pruning | When gradient information is available |

---

## 🗺️ Navigation

| ⬅️ Previous | 🏠 Home | ➡️ Next |
|:-----------:|:-------:|:-------:|
| [← Basics](../02_basics/README.md) | [Efficient ML](../README.md) | [Pruning & Sparsity II →](../04_pruning_sparsity_2/README.md) |

---
## 📚 References

| Type | Resource | Link |
|------|----------|------|
| 📄 | Learning Weights and Connections | [arXiv](https://arxiv.org/abs/1506.02626) |
| 📄 | Optimal Brain Damage | [NeurIPS 1990](https://papers.nips.cc/paper/1989/hash/6c9882bbac1c7093bd25041881277658-Abstract.html) |
| 📄 | Neural Network Pruning Survey | [arXiv](https://arxiv.org/abs/2102.00554) |
| 💻 | PyTorch Pruning Tutorial | [PyTorch](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html) |
| 🎥 | MIT 6.5940 TinyML | [Course](https://hanlab.mit.edu/courses/2024-fall-65940) |

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=1ABC9C&height=80&section=footer" width="100%"/>
</p>
