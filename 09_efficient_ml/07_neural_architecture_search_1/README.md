<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=1ABC9C&height=120&section=header&text=Lecture%207%20Neural%20Architecture%20Search%20I&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-09-1ABC9C?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

# Lecture 7: Neural Architecture Search (Part I)

[← Back to Course](../) | [← Previous](../06_quantization_2/) | [Next: NAS II →](../08_neural_architecture_search_2/)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/ml-researcher-foundations/blob/main/09_efficient_ml/07_neural_architecture_search_1/demo.ipynb) ← **Try the code!**

---

## 📺 Video Lecture

| Resource | Link |
|----------|------|
| 🎥 **Lecture Video** | [Watch on YouTube](https://www.youtube.com/watch?v=RUaFUo7rOXk&list=PL80kAHvQbh-pT4lCkDT53zT8DKmhE0idB&index=7) |
| 📊 **Slides** | [MIT Course Page](https://hanlab.mit.edu/courses/2023-fall-65940) |
| ⏱️ **Duration** | ~90 minutes |

### 📝 Video Summary

This lecture introduces **Neural Architecture Search (NAS)** for automated model design:

- **Why NAS**: Manual architecture design is time-consuming and suboptimal

- **NAS components**: Search space, search strategy, performance estimation

- **RL-based NAS**: Using reinforcement learning to sample architectures (NASNet)

- **DARTS**: Differentiable architecture search with continuous relaxation

- **Cell-based search**: Searching for repeatable building blocks

- **Cost comparison**: From 2000 GPU-days to 1 GPU-day

> 💡 *"NAS automates the tedious process of neural network design—let algorithms find the optimal architecture for your problem."* — Prof. Song Han

---

![Overview](overview.png)

## What is NAS?

**Neural Architecture Search** automates the design of neural networks.

> Instead of humans designing architectures, let algorithms find them!

---

## The NAS Problem

```
Given: Search space, Dataset, Hardware constraints
Find: Architecture that maximizes accuracy 
      while meeting latency/memory targets

```

---

## Components of NAS

### 1. Search Space
What architectures are possible?

| Component | Options |
|-----------|---------|
| Operations | Conv3x3, Conv5x5, MaxPool, Skip |
| Channels | 16, 32, 64, 128 |
| Layers | 10, 20, 50 |
| Connections | Sequential, skip, dense |

### 2. Search Strategy
How do we explore the space?

- Random search

- Reinforcement learning

- Evolutionary algorithms

- Gradient-based (DARTS)

### 3. Performance Estimation
How do we evaluate architectures?

- Train to convergence (slow!)

- Early stopping

- Weight sharing

- Predictor models

---

## DARTS: Differentiable NAS

Make architecture search differentiable!

```python
# Instead of discrete choice:
output = op_1(x)  # or op_2(x) or op_3(x)

# Use soft weighting:
output = α_1 * op_1(x) + α_2 * op_2(x) + α_3 * op_3(x)

# After training, pick argmax(α)

```

**Cost:** Single GPU, ~1 day

---

## 📐 Mathematical Foundations & Proofs

### NAS as Optimization Problem

**Objective:**

```math
\alpha^* = \arg\max_{\alpha \in \mathcal{A}} \text{Val-Acc}(w^*(\alpha), \alpha)

```

subject to:

```math
w^*(\alpha) = \arg\min_w \mathcal{L}_{train}(w, \alpha)

```

where:

- \( \alpha \) = architecture parameters

- \( w \) = network weights

- \( \mathcal{A} \) = search space

This is a **bi-level optimization** problem.

---

### DARTS Continuous Relaxation

**Discrete search space:**

```math
\bar{o}^{(i,j)}(x) = \sum_{o \in \mathcal{O}} \mathbb{1}[\alpha^{(i,j)} = o] \cdot o(x)

```

**Continuous relaxation:**

```math
\bar{o}^{(i,j)}(x) = \sum_{o \in \mathcal{O}} \frac{\exp(\alpha_o^{(i,j)})}{\sum_{o'} \exp(\alpha_{o'}^{(i,j)})} \cdot o(x)

```

This is a **softmax-weighted mixture** of all operations.

**After training:** Discretize by selecting highest-weight operation:

```math
o^* = \arg\max_o \alpha_o^{(i,j)}

```

---

### DARTS Bi-Level Optimization

**Outer loop (architecture):**

```math
\min_\alpha \mathcal{L}_{val}(w^*(\alpha), \alpha)

```

**Inner loop (weights):**

```math
w^*(\alpha) = \arg\min_w \mathcal{L}_{train}(w, \alpha)

```

**Approximation (for tractability):**

Instead of fully solving inner loop, take one gradient step:

```math
w' = w - \xi \nabla_w \mathcal{L}_{train}(w, \alpha)

```

Then update architecture:

```math
\alpha' = \alpha - \eta \nabla_\alpha \mathcal{L}_{val}(w', \alpha)

```

---

### Architecture Gradient Derivation

The gradient of validation loss w.r.t. architecture:

```math
\nabla_\alpha \mathcal{L}_{val}(w^*(\alpha), \alpha) = \nabla_\alpha \mathcal{L}_{val} - \xi \nabla_\alpha \nabla_w \mathcal{L}_{train} \cdot \nabla_{w'} \mathcal{L}_{val}

```

**First-order approximation (faster):**

```math
\nabla_\alpha \mathcal{L}_{val}(w^*(\alpha), \alpha) \approx \nabla_\alpha \mathcal{L}_{val}(w, \alpha)

```

Ignores the second term (Hessian-vector product).

**Second-order approximation (more accurate):**

Using finite differences:

```math
\nabla_\alpha^2 \mathcal{L}_{val} \approx \frac{\nabla_\alpha \mathcal{L}_{val}(w^+) - \nabla_\alpha \mathcal{L}_{val}(w^-)}{2\epsilon}

```

where \( w^\pm = w \pm \epsilon \nabla_{w'} \mathcal{L}_{val} \).

---

### Search Space Size

For a cell with \( N \) nodes and \( K \) operations:

**Number of possible cells:**

```math

|\mathcal{A}_{cell}| = K^{\binom{N}{2}} = K^{N(N-1)/2}

```

**Example:** \( N=4 \), \( K=8 \):

```math
|\mathcal{A}_{cell}| = 8^6 = 262,144 \text{ cells}

```

With normal + reduction cells:

```math

|\mathcal{A}_{total}| = |\mathcal{A}_{cell}|^2 \approx 10^{10}

```

**Exhaustive search is infeasible** → need efficient search strategies.

---

### Weight Sharing for Efficient Evaluation

**Problem:** Training each architecture from scratch is expensive.

**Solution:** Train a **supernet** containing all architectures:

```math
\mathcal{M}_{super}(x; W) = \sum_{\alpha \in \mathcal{A}} p(\alpha) \cdot f(x; W, \alpha)

```

**Path sampling:** At each step, sample an architecture \( \alpha \) and train:

```math
W \leftarrow W - \eta \nabla_W \mathcal{L}(W, \alpha)

```

**Evaluation:** Use shared weights without retraining.

**Approximation quality:**

```math
\text{Acc}_{shared}(\alpha) \approx \text{Acc}_{scratch}(\alpha)

```

Correlation: ~0.8-0.9 (good enough for ranking).

---

### RL-based NAS (NASNet)

**Formulation:** Architecture search as RL problem.

- **State:** Current partial architecture

- **Action:** Select next operation/connection

- **Reward:** Validation accuracy of complete architecture

**Policy gradient update:**

```math
\nabla_\theta J(\theta) = \mathbb{E}_{a \sim \pi_\theta}[R(a) \nabla_\theta \log \pi_\theta(a)]

```

**REINFORCE with baseline:**

```math
\nabla_\theta J(\theta) = \mathbb{E}[(R - b) \nabla_\theta \log \pi_\theta(a)]

```

where \( b \) is the moving average of rewards (baseline).

**Problem:** High variance, needs many samples (expensive).

---

### Evolutionary NAS

**Algorithm:**
1. Initialize population of random architectures
2. Evaluate fitness (accuracy) of each
3. Select top-k (elitism)
4. Mutate/crossover to create offspring
5. Repeat

**Mutation operators:**
- Change operation type

- Add/remove layers

- Modify connections

**Crossover:** Combine cells from two parents

**Advantage:** No differentiability required.

---

## 🧮 Key Derivations

### Memory Cost of DARTS

DARTS maintains all operations simultaneously:

```math
\text{Memory} = \sum_{o \in \mathcal{O}} \text{Memory}(o)

```

For \( K \) operations: \( K \times \) memory of single-path model.

**Solution:** Partial channel connections (PC-DARTS).

### FLOPs of Searched Architecture

For found architecture with operations \( \{o_1, ..., o_L\} \):

```math
\text{FLOPs} = \sum_{l=1}^L \text{FLOPs}(o_l)

```

Can add FLOPs constraint to search:

```math
\min_\alpha \mathcal{L}_{val} + \lambda \cdot \text{FLOPs}(\alpha)

```

---

## 🎯 Where Used

| Concept | Applications |
|---------|-------------|
| NAS | EfficientNet, MobileNetV3 |
| DARTS | Efficient architecture discovery |
| Cell-based Search | Transferable architectures |
| Weight Sharing | Fast evaluation |

---

## 🗺️ Navigation

| ⬅️ Previous | 🏠 Home | ➡️ Next |
|:-----------:|:-------:|:-------:|
| [← Quantization II](../06_quantization_2/README.md) | [Efficient ML](../README.md) | [NAS II →](../08_neural_architecture_search_2/README.md) |

---
## 📚 References

| Type | Resource | Link |
|------|----------|------|
| 📄 | NASNet | [arXiv](https://arxiv.org/abs/1707.07012) |
| 📄 | DARTS | [arXiv](https://arxiv.org/abs/1806.09055) |
| 📄 | EfficientNet | [arXiv](https://arxiv.org/abs/1905.11946) |
| 📄 | PC-DARTS | [arXiv](https://arxiv.org/abs/1907.05737) |
| 🎥 | MIT 6.5940 TinyML | [Course](https://hanlab.mit.edu/courses/2024-fall-65940) |

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=1ABC9C&height=80&section=footer" width="100%"/>
</p>
