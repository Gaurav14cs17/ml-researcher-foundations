<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=1ABC9C&height=120&section=header&text=Lecture%209%20Knowledge%20Distillation&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-09-1ABC9C?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

# Lecture 9: Knowledge Distillation

[← Back to Course](../) | [← Previous](../08_neural_architecture_search_2/) | [Next: MCUNet →](../10_mcunet_tinyml/)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/ml-researcher-foundations/blob/main/09_efficient_ml/09_knowledge_distillation/demo.ipynb) ← **Try the code!**

---

## 📺 Video Lecture

| Resource | Link |
|----------|------|
| 🎥 **Lecture Video** | [Watch on YouTube](https://www.youtube.com/watch?v=IxQtK2SjWWM&list=PL80kAHvQbh-pT4lCkDT53zT8DKmhE0idB&index=9) |
| 📊 **Slides** | [MIT Course Page](https://hanlab.mit.edu/courses/2023-fall-65940) |
| ⏱️ **Duration** | ~90 minutes |

### 📝 Video Summary

This lecture covers **knowledge distillation** for model compression:

- **Core idea**: Transfer knowledge from large "teacher" to small "student"
- **Soft labels**: Why teacher probabilities contain more information than hard labels
- **Temperature scaling**: Making probability distributions softer for better knowledge transfer
- **Feature distillation**: Matching intermediate representations, not just outputs
- **Self-distillation**: Using a model as its own teacher
- **LLM distillation**: Creating Alpaca/Vicuna from GPT-4

> 💡 *"Soft labels from the teacher encode dark knowledge—relationships between classes that hard labels don't capture."* — Prof. Song Han

---

![Overview](overview.png)

## What is Knowledge Distillation?

Transfer knowledge from a large "teacher" model to a small "student" model.

```
Teacher (Large): ResNet-152, 60M params, 78% acc
     ↓ Distill knowledge
Student (Small): MobileNet, 3M params, 76% acc (was 72% without distillation!)

```

---

## Why Does Distillation Work?

**Soft labels contain more information than hard labels!**

| Class | Hard Label | Soft Label (Teacher) |
|-------|-----------|---------------------|
| Cat | 1 | 0.75 |
| Dog | 0 | 0.15 |
| Car | 0 | 0.05 |
| ... | 0 | 0.05 |

The soft label says: "This is probably a cat, but it looks a bit like a dog."

---

## Temperature Scaling

Higher temperature → softer probability distribution:

```
T=1:  [0.9, 0.05, 0.05]  # Very confident
T=4:  [0.6, 0.2, 0.2]    # More uniform (more information)
T=20: [0.4, 0.3, 0.3]    # Very soft

```

---

## 📐 Mathematical Foundations & Proofs

### Distillation Loss Function

**Combined loss:**

```math
\mathcal{L} = \alpha \cdot \mathcal{L}_{soft} + (1-\alpha) \cdot \mathcal{L}_{hard}

```

**Hard target loss (standard cross-entropy):**

```math
\mathcal{L}_{hard} = -\sum_i y_i \log(p_i^S)

```

**Soft target loss (KL divergence):**

```math
\mathcal{L}_{soft} = T^2 \cdot \text{KL}(p^T \| p^S)

```

where:
- \( p^T = \text{softmax}(z^T / T) \) = teacher's soft predictions
- \( p^S = \text{softmax}(z^S / T) \) = student's soft predictions
- \( T \) = temperature

**Why \( T^2 \) scaling?**

The gradients of soft loss are scaled by \( 1/T^2 \) due to temperature. The \( T^2 \) factor compensates to maintain gradient magnitude.

---

### Temperature Effect on Softmax

**Standard softmax (\( T=1 \)):**

```math
p_i = \frac{\exp(z_i)}{\sum_j \exp(z_j)}

```

**Temperature-scaled softmax:**

```math
p_i^{(T)} = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}

```

**Proof of softening effect:**

As \( T \to \infty \):

```math
\lim_{T \to \infty} p_i^{(T)} = \frac{1}{K}

```

All classes become equally likely (maximum entropy).

As \( T \to 0 \):

```math
\lim_{T \to 0} p_i^{(T)} = \mathbb{1}[i = \arg\max_j z_j]

```

Becomes hard one-hot (minimum entropy).

---

### Dark Knowledge

**Observation:** Even incorrect class probabilities contain useful information.

For a cat image, the teacher might output:
- P(cat) = 0.80
- P(dog) = 0.15
- P(car) = 0.05

The student learns:
1. This is a cat
2. Cats look somewhat like dogs (both are animals)
3. Cats don't look like cars

**Formal interpretation:**

The soft labels encode **class similarities**:

```math
\text{sim}(c_i, c_j) \propto \mathbb{E}_{x}[p^T_i(x) \cdot p^T_j(x)]

```

---

### Gradient Analysis

**Gradient of soft loss w.r.t. student logits:**

```math
\frac{\partial \mathcal{L}_{soft}}{\partial z_i^S} = \frac{1}{T}(p_i^S - p_i^T)

```

**Interpretation:**
- If \( p_i^S > p_i^T \): Gradient is positive → decrease \( z_i^S \)
- If \( p_i^S < p_i^T \): Gradient is negative → increase \( z_i^S \)

The student learns to match the teacher's full distribution, not just the correct class.

---

### Feature Distillation

Match intermediate representations, not just outputs:

```math
\mathcal{L}_{feat} = \sum_{l \in \mathcal{L}} \|g_l(F_l^S) - F_l^T\|_2^2

```

where:
- \( F_l^S \) = student feature at layer \( l \)
- \( F_l^T \) = teacher feature at layer \( l \)
- \( g_l \) = projection to match dimensions

**FitNets:** Learn \( g_l \) as trainable projector.

**Attention transfer:**

```math
\mathcal{L}_{AT} = \sum_l \left\| \frac{A_l^S}{\|A_l^S\|_2} - \frac{A_l^T}{\|A_l^T\|_2} \right\|_2^2

```

where \( A_l = \sum_c |F_l^c|^2 \) is the spatial attention map.

---

### Self-Distillation

**Observation:** A model can distill from itself!

**Process:**
1. Train model \( M_1 \)
2. Use \( M_1 \) as teacher, train \( M_2 \) (same architecture)
3. \( M_2 \) typically has 1-2% higher accuracy

**Explanation:**
- Training with soft labels provides richer supervision
- Regularization effect prevents overfitting
- Label smoothing is implicit

**Born-Again Networks (BAN):**

```math
\text{Acc}(M_{n+1}) > \text{Acc}(M_n)

```

for several generations of self-distillation.

---

### DistilBERT

**Initialization:** Start from teacher weights (take every other layer):

```math
W^S_l = W^T_{2l}

```

**Training objectives:**
1. **MLM loss:** Standard masked language modeling
2. **Distillation loss:** Match teacher's soft predictions
3. **Cosine embedding loss:** Match hidden states

```math
\mathcal{L} = \mathcal{L}_{MLM} + \alpha \mathcal{L}_{KD} + \beta \mathcal{L}_{cos}

```

**Result:** 40% smaller, 60% faster, 97% of BERT performance.

---

### LLM Distillation

**Challenge:** Can't access teacher weights (GPT-4 is proprietary).

**Solution:** Distill from outputs only.

**Alpaca approach:**
1. Generate (instruction, response) pairs using GPT-4
2. Fine-tune LLaMA on these pairs
3. Student learns to mimic GPT-4's behavior

**Mathematical formulation:**

```math
\min_\theta \mathbb{E}_{(x,y) \sim \text{GPT-4}}[-\log p_\theta(y|x)]

```

This is maximum likelihood on teacher-generated data.

---

## 🧮 Key Derivations

### Optimal Temperature

**Empirical finding:** \( T = 4-8 \) works well for classification.

**Intuition:**
- \( T = 1 \): Hard labels, limited knowledge transfer
- \( T = 20 \): Too soft, loses discriminative information
- \( T \approx 4 \): Good balance

**Optimal \( T \) depends on:**
- Number of classes
- Teacher confidence
- Task difficulty

---

### Capacity Gap

**Problem:** If student is too small, it can't absorb teacher's knowledge.

**Teacher-Student capacity ratio:**

```math
r = \frac{|\theta_S|}{|\theta_T|}

```

**Empirical observation:**
- \( r > 0.1 \): Good distillation
- \( r < 0.01 \): Capacity gap issues

**Solution: Teacher Assistant (TA):**

```math
\text{Teacher} \to \text{TA} \to \text{Student}

```

Intermediate-sized model bridges the gap.

---

### Distillation vs Data Augmentation

**Distillation:** Uses teacher to generate soft labels

```math
\mathcal{L}_{dist} = \text{KL}(p^T(x) \| p^S(x))

```

**Data augmentation:** Applies transformations to inputs

```math
\mathcal{L}_{aug} = \text{CE}(y, p^S(t(x)))

```

**Combination is best:**

```math
\mathcal{L}_{total} = \mathcal{L}_{dist}(t(x)) + \mathcal{L}_{hard}(t(x))

```

---

## 🎯 Where Used

| Concept | Applications |
|---------|-------------|
| Logit Distillation | Compressing CNNs, DistilBERT |
| Feature Distillation | FitNets, attention transfer |
| Self-Distillation | Improving single model accuracy |
| LLM Distillation | Alpaca, Vicuna from GPT-4 |

---

## 🗺️ Navigation

| ⬅️ Previous | 🏠 Home | ➡️ Next |
|:-----------:|:-------:|:-------:|
| [← NAS II](../08_neural_architecture_search_2/README.md) | [Efficient ML](../README.md) | [MCUNet & TinyML →](../10_mcunet_tinyml/README.md) |

---
## 📚 References

| Type | Resource | Link |
|------|----------|------|
| 📄 | Hinton Distillation Paper | [arXiv](https://arxiv.org/abs/1503.02531) |
| 📄 | FitNets | [arXiv](https://arxiv.org/abs/1412.6550) |
| 📄 | DistilBERT | [arXiv](https://arxiv.org/abs/1910.01108) |
| 📄 | Born-Again Networks | [arXiv](https://arxiv.org/abs/1805.04770) |
| 🎥 | MIT 6.5940 TinyML | [Course](https://hanlab.mit.edu/courses/2024-fall-65940) |

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=1ABC9C&height=80&section=footer" width="100%"/>
</p>
