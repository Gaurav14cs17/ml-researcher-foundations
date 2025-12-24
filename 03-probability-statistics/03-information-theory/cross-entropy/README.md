# Cross-Entropy

> **The most common loss function for classification**

---

## 🎯 Visual Overview

<img src="./images/cross-entropy.svg" width="100%">

*Caption: Cross-entropy measures the "distance" between predicted probabilities and true labels. It heavily penalizes confident wrong predictions. The loss approaches 0 when predictions match the true labels.*

---

## 📂 Overview

Cross-entropy is the standard loss function for classification. It combines softmax activation with negative log-likelihood into a stable, differentiable objective.

---

## 🔑 Formulas

| Type | Formula | Use Case |
|------|---------|----------|
| **Binary** | -[y·log(p) + (1-y)·log(1-p)] | Binary classification |
| **Categorical** | -Σᵢ yᵢ·log(pᵢ) | Multi-class |
| **Sparse** | -log(p_true) | Multi-class (label indices) |

---

## 📐 Why Cross-Entropy?

```
MSE for classification:  L = (y - p)²  → gradients vanish when p ≈ 0 or 1

Cross-entropy:           L = -log(p)   → strong gradient when wrong
                                       → approaches 0 when correct

Gradient: ∂L/∂logit = p - y  (elegant!)
```

---

## 💻 Code

```python
import torch.nn.functional as F

# Multi-class classification (most common)
loss = F.cross_entropy(logits, labels)  # logits: (B, C), labels: (B,)

# Binary classification
loss = F.binary_cross_entropy_with_logits(logits, labels.float())

# With label smoothing (regularization)
loss = F.cross_entropy(logits, labels, label_smoothing=0.1)
```

---

## 📐 DETAILED MATHEMATICAL THEORY

### 1. Cross-Entropy: Complete Definition

**Information-Theoretic Definition:**

```
Cross-entropy H(P, Q) measures the expected number of bits 
needed to encode data from distribution P using a code optimized for Q.

Discrete:
  H(P, Q) = -Σₓ P(x)·log Q(x)
          = 𝔼_P[-log Q(x)]

Continuous:
  H(P, Q) = -∫ p(x)·log q(x) dx

Relationship to KL divergence:
  H(P, Q) = H(P) + D_KL(P||Q)
  
where H(P) is the entropy of P.
```

**Key Properties:**

```
1. Non-negative: H(P, Q) ≥ 0
2. Minimum at equality: H(P, Q) ≥ H(P), with equality iff P = Q
3. Not symmetric: H(P, Q) ≠ H(Q, P) in general
4. Lower bounded by entropy: H(P, Q) ≥ H(P)
```

---

### 2. Cross-Entropy for Classification

**Binary Classification:**

```
True label: y ∈ {0, 1}
Prediction: p ∈ [0, 1]  (probability of class 1)

Binary cross-entropy:
  L(y, p) = -[y·log(p) + (1-y)·log(1-p)]

Cases:
  • y = 1: L = -log(p)     (penalize if p small)
  • y = 0: L = -log(1-p)   (penalize if p large)

Properties:
  • L → 0 when prediction correct (p → y)
  • L → ∞ when prediction wrong with high confidence
  • Gradient: ∂L/∂p = (p-y)/(p(1-p))
```

**Multi-Class Classification (Categorical Cross-Entropy):**

```
True label: y ∈ {1,...,C}  (class index)
Prediction: p = [p₁,...,p_C] where Σᵢpᵢ = 1  (probability distribution)

Cross-entropy loss:
  L(y, p) = -log(p_y)
          = -Σᵢ₌₁ᶜ 𝟙{y=i}·log(pᵢ)

One-hot encoding (y as vector):
  y = [0,...,0,1,0,...,0]  (1 at position y)
  L = -Σᵢ yᵢ·log(pᵢ)

Properties:
  • Focuses on probability of correct class
  • Heavily penalizes confident wrong predictions
  • Gradient: ∂L/∂pᵢ = -yᵢ/pᵢ
```

---

### 3. Softmax + Cross-Entropy: The Perfect Pair

**Why Softmax?**

```
Raw network output (logits): z = [z₁,...,z_C] ∈ ℝᶜ

Softmax:
  pᵢ = exp(zᵢ) / Σⱼ exp(zⱼ)

Properties:
  • Σᵢ pᵢ = 1 (valid probability distribution)
  • pᵢ > 0 (no zero probabilities)
  • Differentiable everywhere
  • max(zᵢ) → p ≈ one-hot (confident prediction)
```

**Combined Gradient (The Magic!):**

```
Loss: L = -log(p_y) where pᵢ = softmax(z)ᵢ

Direct computation of ∂L/∂zᵢ:
  ∂L/∂zᵢ = ∂L/∂pⱼ · ∂pⱼ/∂zᵢ  (sum over j)

Softmax derivative:
  ∂pⱼ/∂zᵢ = pⱼ(δᵢⱼ - pᵢ)
  
where δᵢⱼ = 1 if i=j, 0 otherwise

Result (beautiful simplification!):
  ∂L/∂zᵢ = pᵢ - yᵢ
  
where yᵢ = 𝟙{class = i} (one-hot)

Interpretation:
  Gradient = prediction - truth
  
  Example:
    True class: 2 (out of 3 classes)
    y = [0, 1, 0]
    p = [0.1, 0.7, 0.2]
    ∇L = [0.1, -0.3, 0.2]
    
    Gradient pushes probability toward true class!
```

**Proof of ∂L/∂zᵢ = pᵢ - yᵢ:**

```
Step 1: Write loss
  L = -log(p_y) = -log(exp(z_y)/Σⱼexp(zⱼ))
    = -z_y + log(Σⱼexp(zⱼ))

Step 2: Differentiate
  Case i = y (true class):
    ∂L/∂zᵧ = -1 + exp(zᵧ)/Σⱼexp(zⱼ)
           = -1 + pᵧ
           = pᵧ - 1
           = pᵧ - yᵧ  (since yᵧ=1)
  
  Case i ≠ y (other classes):
    ∂L/∂zᵢ = 0 + exp(zᵢ)/Σⱼexp(zⱼ)
           = pᵢ
           = pᵢ - 0
           = pᵢ - yᵢ  (since yᵢ=0)

Both cases: ∂L/∂zᵢ = pᵢ - yᵢ ✓  QED
```

---

### 4. Numerical Stability: LogSumExp Trick

**Problem: Softmax Can Overflow**

```
pᵢ = exp(zᵢ) / Σⱼ exp(zⱼ)

If zᵢ large (e.g., z = [1000, 1001, 999]):
  exp(1000) ≈ 10⁴³⁴ → inf in float32!
```

**Solution: Subtract Maximum**

```
Mathematically equivalent:
  pᵢ = exp(zᵢ - max(z)) / Σⱼ exp(zⱼ - max(z))

Why it works:
  exp(zᵢ)/Σⱼexp(zⱼ) = exp(zᵢ - c)/Σⱼexp(zⱼ - c)  for any c
  
  Choose c = max(z):
    • Largest value becomes exp(0) = 1
    • All other values ≤ 1
    • No overflow!

Example:
  z = [1000, 1001, 999]
  z' = z - 1001 = [-1, 0, -2]
  p = [e⁻¹, e⁰, e⁻²] / (e⁻¹ + e⁰ + e⁻²)
    ≈ [0.268, 0.665, 0.099]
```

**Log-Softmax (Even More Stable):**

```
For cross-entropy, we need log(pᵢ):
  log pᵢ = log(exp(zᵢ)/Σⱼexp(zⱼ))
         = zᵢ - log(Σⱼexp(zⱼ))
         = zᵢ - LogSumExp(z)

LogSumExp(z) = log(Σⱼexp(zⱼ))
             = max(z) + log(Σⱼexp(zⱼ - max(z)))  (stable!)

Then cross-entropy:
  L = -log(p_y)
    = -(z_y - LogSumExp(z))
    = LogSumExp(z) - z_y

PyTorch: F.log_softmax(z) computes this stably
```

---

### 5. Cross-Entropy vs Mean Squared Error

**Why Not MSE for Classification?**

```
MSE: L = (1/2)||y - p||²

Problem 1: Saturating gradients
  For sigmoid activation σ(z):
    ∂L/∂z = (p - y)·p·(1-p)
            ↑       ↑
         error   derivative
    
  When p ≈ 0 or 1: σ'(z) ≈ 0 → vanishing gradient!
  Even if prediction is very wrong

Cross-entropy: L = -[y·log(p) + (1-y)·log(1-p)]
  ∂L/∂z = p - y
  
  No multiplication by σ'! Gradient doesn't vanish.

Problem 2: Wrong assumptions
  MSE assumes Gaussian noise: p(y|z) = N(y; f(z), σ²)
  But y ∈ {0,1} is discrete! Not Gaussian.
  
  Cross-entropy assumes Bernoulli: p(y|z) = Bernoulli(σ(z))
  Correct model for binary outcomes.
```

**Comparison:**

```
                MSE                     Cross-Entropy
Loss:           (y-p)²                  -y·log(p) - (1-y)·log(1-p)
Gradient:       (p-y)·σ'(z)             p - y
Saturation:     Yes (at extremes)       No
Convergence:    Slow                    Fast
Interpretation: L2 distance             Log-likelihood
Best for:       Regression              Classification
```

---

### 6. Probabilistic Interpretation: Maximum Likelihood

**Cross-Entropy = Negative Log-Likelihood**

```
Dataset: D = {(x₁,y₁), ..., (xₙ,yₙ)}

Model: p(y|x; θ) = predicted probability

Log-likelihood:
  ℓ(θ) = Σᵢ₌₁ⁿ log p(yᵢ|xᵢ; θ)

Maximum likelihood estimation:
  θ* = argmax_θ ℓ(θ)
     = argmin_θ -ℓ(θ)
     = argmin_θ Σᵢ₌₁ⁿ -log p(yᵢ|xᵢ; θ)

For classification:
  -log p(yᵢ|xᵢ; θ) = cross-entropy loss!

Therefore:
  Minimizing cross-entropy = Maximum likelihood estimation ✓
```

**Multi-Class Case:**

```
Model: p(y=c|x; θ) = p_c(x; θ)  (softmax output)

One sample:
  L(y, p) = -log p_y
  
Dataset:
  L = -(1/n)Σᵢ₌₁ⁿ log p_{yᵢ}(xᵢ; θ)
    = average negative log-likelihood
    = cross-entropy loss

Interpretation:
  Find parameters θ that maximize probability of observing the data
```

---

### 7. Label Smoothing: Regularized Cross-Entropy

**Problem: Overconfidence**

```
Model predicts: p = [0.001, 0.998, 0.001]
True label: y = [0, 1, 0]

Loss is low, but model is overconfident!
May not generalize well.
```

**Label Smoothing (Szegedy et al., 2016):**

```
Smooth the hard targets:
  y_smooth = (1 - α)·y + α/C
  
where:
  α ∈ [0,1] = smoothing parameter (typically 0.1)
  C = number of classes

Example (C=3, α=0.1):
  Original: y = [0, 1, 0]
  Smoothed: y = [0.033, 0.933, 0.033]

Effect:
  • True class: 1 → 0.9 + 0.1/3 = 0.933
  • Other classes: 0 → 0.1/3 = 0.033
```

**Why It Helps:**

```
1. Prevents overconfidence
   Model can't push p_y → 1 (would increase loss on other classes)

2. Implicit regularization
   Equivalent to adding entropy regularization:
     L_total = L_CE - α·H(p)
   
   Encourages higher entropy (less confident) predictions

3. Better generalization
   Empirically shown to improve test accuracy
   Used in: Inception, ResNet, Transformers

4. Calibration
   Model probabilities better match true frequencies
```

**Implementation:**

```python
def cross_entropy_with_label_smoothing(logits, labels, alpha=0.1):
    """
    logits: (batch_size, num_classes)
    labels: (batch_size,)  integer labels
    """
    num_classes = logits.size(-1)
    
    # Create smoothed labels
    smoothed_labels = torch.full_like(logits, alpha / num_classes)
    smoothed_labels.scatter_(-1, labels.unsqueeze(-1), 1.0 - alpha + alpha / num_classes)
    
    # Compute cross-entropy with soft labels
    log_probs = F.log_softmax(logits, dim=-1)
    loss = -(smoothed_labels * log_probs).sum(dim=-1).mean()
    
    return loss

# PyTorch built-in (since 1.10):
loss = F.cross_entropy(logits, labels, label_smoothing=0.1)
```

---

### 8. Focal Loss: Handling Class Imbalance

**Problem: Easy Examples Dominate**

```
In object detection:
  • Background patches: 99.9% (easy, low loss)
  • Object patches: 0.1% (hard, high loss)

Standard CE: L = -log(p_t)
  Easy examples (p_t ≈ 1): Still contribute to loss
  Model spends time on easy examples
```

**Focal Loss (Lin et al., 2017):**

```
FL(p_t) = -(1 - p_t)^γ · log(p_t)

where:
  p_t = p if y=1, else 1-p  (probability of true class)
  γ ≥ 0 = focusing parameter (typically 2)

Behavior:
  • Easy examples (p_t → 1): (1-p_t)^γ → 0, down-weighted!
  • Hard examples (p_t → 0): (1-p_t)^γ → 1, full weight

Example (γ=2):
  p_t = 0.9 (easy):   FL = 0.01 · (-log 0.9) ≈ 0.001
  p_t = 0.5 (medium): FL = 0.25 · (-log 0.5) ≈ 0.17
  p_t = 0.1 (hard):   FL = 0.81 · (-log 0.1) ≈ 1.87
  
  Hard example gets 1870× more weight than easy!
```

**With Class Balancing:**

```
FL(p_t) = -α_t·(1 - p_t)^γ · log(p_t)

where α_t is the weight for class t (e.g., inverse frequency)

Applications:
  • One-stage object detection (RetinaNet)
  • Semantic segmentation with imbalance
  • Any classification with hard negatives
```

---

### 9. Temperature Scaling in Cross-Entropy

**Knowledge Distillation Temperature:**

```
Softmax with temperature:
  pᵢ = exp(zᵢ/T) / Σⱼexp(zⱼ/T)

where T > 0 is temperature

Effect:
  • T = 1: Standard softmax
  • T > 1: "Softer" probabilities (more uniform)
  • T < 1: "Sharper" probabilities (more confident)

Example (z = [1, 2, 3]):
  T=1:   p = [0.09, 0.24, 0.67]
  T=2:   p = [0.16, 0.29, 0.55]  (softer)
  T=10:  p = [0.30, 0.33, 0.37]  (almost uniform)
  T=0.5: p = [0.02, 0.12, 0.86]  (sharper)
```

**Knowledge Distillation:**

```
Teacher model: p_T = softmax(z_T/T)
Student model: p_S = softmax(z_S/T)

Distillation loss:
  L_distill = T² · H(p_T, p_S)
           = T² · (-Σᵢ p_T,i · log p_S,i)

Factor T²: Compensates for gradient scaling
  ∂/∂z (log softmax(z/T)) scales as 1/T
  So loss scales as 1/T²
  Multiply by T² to get O(1) gradients

Why it works:
  • High temperature → soft targets reveal "dark knowledge"
  • Example: [0.9, 0.05, 0.05] vs [0.9, 0.08, 0.02]
    Both have same hard label, but soft labels differ!
  • Student learns relative confidences, not just top-1
```

---

### 10. Relationship to Other Losses

**Cross-Entropy ⟷ KL Divergence:**

```
H(P, Q) = H(P) + D_KL(P||Q)

For classification:
  P = true distribution (one-hot)
  Q = predicted distribution
  
  H(P) = 0  (one-hot has zero entropy)
  
Therefore:
  Minimizing H(P,Q) = Minimizing D_KL(P||Q)
  
  Cross-entropy and KL divergence are equivalent!
```

**Cross-Entropy ⟷ Maximum Likelihood:**

```
L_CE = -(1/n)Σᵢ log p(yᵢ|xᵢ)
     = -log likelihood / n

Minimizing cross-entropy = Maximum likelihood estimation
```

**Binary CE ⟷ Logistic Regression:**

```
Logistic regression:
  p(y=1|x) = σ(wᵀx) where σ(z) = 1/(1+e⁻ᶻ)

Likelihood:
  L = Πᵢ p(yᵢ|xᵢ)
    = Πᵢ σ(wᵀxᵢ)^yᵢ · (1-σ(wᵀxᵢ))^{1-yᵢ}

Log-likelihood:
  log L = Σᵢ [yᵢ·log σ(wᵀxᵢ) + (1-yᵢ)·log(1-σ(wᵀxᵢ))]

Negative log-likelihood:
  -log L = Σᵢ -[yᵢ·log pᵢ + (1-yᵢ)·log(1-pᵢ)]
         = Σᵢ BCE(yᵢ, pᵢ)

Binary cross-entropy is logistic regression loss!
```

---

### 11. Practical Implementation Tips

**Numerical Stability Checklist:**

```
✓ Use log_softmax instead of log(softmax(·))
✓ Use F.cross_entropy (combines log_softmax + nll_loss)
✓ Use F.binary_cross_entropy_with_logits (not sigmoid + BCE)
✓ Avoid manual softmax + log operations
✓ Clip probabilities away from 0/1 if manual: p = clip(p, 1e-7, 1-1e-7)
```

**Memory Optimization:**

```
# Bad: Stores intermediate probabilities
probs = F.softmax(logits, dim=-1)
loss = -torch.log(probs[range(len(probs)), labels]).mean()

# Good: Fused operation, less memory
loss = F.cross_entropy(logits, labels)

For large batch/vocabulary:
  Use F.cross_entropy with reduction='sum', then divide
  Avoids storing per-example losses
```

**Multi-GPU Training:**

```
When using DataParallel/DistributedDataParallel:
  
  # Gather labels across GPUs
  # Then compute loss

  Or use reduction='mean' (default) and PyTorch handles it
```

---

## 🔗 Where This Topic Is Used

| Application | Usage |
|-------------|-------|
| **Image Classification** | Primary loss function (ImageNet, CIFAR) |
| **Language Modeling** | Next token prediction (GPT, BERT) |
| **Object Detection** | Classification head (with focal loss) |
| **Machine Translation** | Sequence-to-sequence models |
| **Speech Recognition** | CTC loss uses cross-entropy |
| **Knowledge Distillation** | Teacher-student training |
| **Reinforcement Learning** | Policy gradient (actor loss) |


## 📚 References

| Type | Resource | Link |
|------|----------|------|
| 📄 | Label Smoothing Paper | [Rethinking Inception](https://arxiv.org/abs/1512.00567) |
| 📄 | Focal Loss Paper | [RetinaNet](https://arxiv.org/abs/1708.02002) |
| 📄 | Knowledge Distillation | [Hinton et al.](https://arxiv.org/abs/1503.02531) |
| 📖 | Deep Learning Book | Chapter 5 (ML Basics) |
| 🇨🇳 | 交叉熵详解 | [知乎](https://zhuanlan.zhihu.com/p/35709485) |
| 🇨🇳 | Softmax与交叉熵 | [CSDN](https://blog.csdn.net/u014380165/article/details/77284921) |

---

⬅️ [Back: Information Theory](../)

---

➡️ [Next: Entropy](../entropy/)
