<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=9C27B0&height=120&section=header&text=Entropy&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-03-9C27B0?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/entropy-concept.svg" width="100%">

*Caption: Entropy measures uncertainty in a distribution. Low entropy = concentrated probability (predictable). High entropy = uniform distribution (maximum uncertainty). The maximum entropy for n outcomes is log₂(n) bits.*

---

## 📂 Overview

Shannon entropy quantifies the average "surprise" or information content of a random variable. It's fundamental to information theory and has crucial applications in ML.

---

## 📐 Mathematical Definition

### Shannon Entropy

```
H(X) = -Σ p(x) log p(x) = E[-log p(X)]

Interpretation:
- Expected "surprise" of sampling from X
- Minimum bits needed to encode outcomes
- Measures uncertainty/randomness

Units:
- log₂: bits (most common in CS)
- logₑ: nats (natural units)
```

### Key Properties

```
1. Non-negativity: H(X) ≥ 0
2. Maximum: H(X) ≤ log|X|  (uniform distribution)
3. Concavity: H(λp₁ + (1-λ)p₂) ≥ λH(p₁) + (1-λ)H(p₂)
4. Chain rule: H(X,Y) = H(X) + H(Y|X)
```

### Examples

```
Fair coin:   H(X) = -½log(½) - ½log(½) = 1 bit
Biased coin: p=0.9 → H = -0.9log(0.9) - 0.1log(0.1) ≈ 0.47 bits
Fair die:    H(X) = log₂(6) ≈ 2.58 bits
Deterministic: H(X) = 0 (no uncertainty)
```

---

## 🔑 Related Quantities

| Concept | Formula | Meaning |
|---------|---------|---------|
| **Entropy** | H(X) = -Σ p(x) log p(x) | Uncertainty of X |
| **Joint Entropy** | H(X,Y) = -Σ p(x,y) log p(x,y) | Combined uncertainty |
| **Conditional** | H(Y\|X) = H(X,Y) - H(X) | Remaining uncertainty |
| **Mutual Info** | I(X;Y) = H(X) - H(X\|Y) | Shared information |
| **Cross-Entropy** | H(p,q) = -Σ p(x) log q(x) | Bits using wrong code |
| **KL Divergence** | D(p\|\|q) = H(p,q) - H(p) | Extra bits from q |

### Relationships

```
I(X;Y) = H(X) + H(Y) - H(X,Y)
       = H(X) - H(X|Y)
       = H(Y) - H(Y|X)

H(X,Y) = H(X) + H(Y|X) = H(Y) + H(X|Y)

D_KL(p||q) = H(p,q) - H(p) ≥ 0
```

---

## 🌍 ML Applications

| Application | How Entropy is Used |
|-------------|---------------------|
| **Cross-Entropy Loss** | Classification objective: -Σ yᵢ log ŷᵢ |
| **KL Divergence** | VAE regularization: D_KL(q(z\|x) \|\| p(z)) |
| **Entropy Regularization** | Encourage exploration in RL |
| **Decision Trees** | Information gain = H(Y) - H(Y\|X) |
| **Maximum Entropy** | Principle for making predictions |
| **Language Models** | Perplexity = 2^H(p) |

### Cross-Entropy in Classification

```
True label (one-hot): y = [0, 0, 1, 0, 0]
Predicted probs:      ŷ = [0.1, 0.1, 0.6, 0.1, 0.1]

Cross-Entropy Loss:
H(y, ŷ) = -Σ yᵢ log ŷᵢ = -log(0.6) ≈ 0.51

Lower is better!
Perfect prediction (ŷ[2]=1) → H = 0
```

---

## 💻 Code Examples

### Computing Entropy

```python
import numpy as np
import torch
import torch.nn.functional as F

def entropy_numpy(probs):
    """Shannon entropy in bits"""
    probs = np.array(probs)
    probs = probs[probs > 0]  # Avoid log(0)
    return -np.sum(probs * np.log2(probs))

def entropy_torch(probs):
    """Shannon entropy (nats)"""
    return -(probs * probs.log()).sum(-1)

# Examples
fair_coin = [0.5, 0.5]
biased_coin = [0.9, 0.1]
uniform_die = [1/6] * 6

print(f"Fair coin: {entropy_numpy(fair_coin):.4f} bits")      # 1.0
print(f"Biased coin: {entropy_numpy(biased_coin):.4f} bits")  # 0.469
print(f"Fair die: {entropy_numpy(uniform_die):.4f} bits")     # 2.585
```

### Cross-Entropy Loss

```python
import torch.nn.functional as F

# Logits (raw scores) and labels
logits = torch.tensor([[2.0, 0.5, -1.0, 0.1]])  # 1 sample, 4 classes
labels = torch.tensor([0])  # True class is 0

# Cross-entropy loss (combines softmax + negative log-likelihood)
loss = F.cross_entropy(logits, labels)
print(f"Cross-entropy loss: {loss.item():.4f}")

# Manual computation
probs = F.softmax(logits, dim=-1)
print(f"Probabilities: {probs}")
manual_loss = -torch.log(probs[0, labels[0]])
print(f"Manual loss: {manual_loss.item():.4f}")
```

### Entropy Regularization in RL

```python
def policy_loss_with_entropy(logits, actions, advantages, entropy_coef=0.01):
    """
    Policy gradient loss with entropy bonus
    High entropy = more exploration
    """
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    
    # Policy loss
    action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze()
    policy_loss = -(action_log_probs * advantages).mean()
    
    # Entropy bonus (encourages exploration)
    entropy = -(probs * log_probs).sum(-1).mean()
    
    # Total loss (subtract entropy to maximize it)
    return policy_loss - entropy_coef * entropy
```

### Maximum Entropy Distribution

```python
from scipy.optimize import minimize
from scipy.special import entr

def max_entropy_distribution(constraints):
    """
    Find maximum entropy distribution subject to constraints
    Example: E[X] = μ constraint on categorical
    """
    n = len(constraints['values'])
    
    def neg_entropy(p):
        return -np.sum(entr(p))  # Minimize negative entropy
    
    def mean_constraint(p):
        return np.dot(p, constraints['values']) - constraints['mean']
    
    def sum_constraint(p):
        return np.sum(p) - 1
    
    result = minimize(
        neg_entropy,
        x0=np.ones(n) / n,
        constraints=[
            {'type': 'eq', 'fun': sum_constraint},
            {'type': 'eq', 'fun': mean_constraint}
        ],
        bounds=[(0, 1)] * n
    )
    
    return result.x
```

---

## 📊 Entropy in Practice

| Context | High Entropy Means | Low Entropy Means |
|---------|-------------------|-------------------|
| **Classification** | Uncertain prediction | Confident prediction |
| **RL Policy** | Exploring | Exploiting |
| **VAE Latent** | Diverse samples | Concentrated posterior |
| **Language Model** | Uncertain next word | Predictable text |

---

## 🔗 Connection to Other Topics

```
Entropy
    |
    +-- Cross-Entropy (loss function)
    |       +-- → Classification training
    |
    +-- KL Divergence (relative entropy)
    |       +-- → VAE, Information Bottleneck
    |
    +-- Mutual Information
    |       +-- → Representation learning
    |
    +-- Maximum Entropy Principle
            +-- → Exponential family, Softmax
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | Cross-Entropy | [../cross-entropy/](../cross-entropy/) |
| 📖 | KL Divergence | [../kl-divergence/](../kl-divergence/) |
| 📖 | Mutual Information | [../mutual-information/](../mutual-information/) |
| 📄 | Cover & Thomas | [Information Theory](https://www.wiley.com/en-us/Elements+of+Information+Theory%2C+2nd+Edition-p-9780471241959) |
| 🇨🇳 | 信息熵详解 | [知乎](https://zhuanlan.zhihu.com/p/26486223) |
| 🇨🇳 | 交叉熵损失函数 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88393322) |
| 🇨🇳 | 信息论基础 | [B站](https://www.bilibili.com/video/BV1VW411M7PW) |
| 🇨🇳 | 最大熵原理 | [机器之心](https://www.jiqizhixin.com/articles/2017-04-18-5)

---

## 🔗 Where Information Theory Is Used

| Application | Concept Used |
|-------------|--------------|
| **Classification Loss** | Cross-entropy as training objective |
| **Knowledge Distillation** | KL divergence between teacher/student |
| **VAE Training** | KL divergence regularization |
| **Language Modeling** | Perplexity = exp(cross-entropy) |
| **Model Compression** | Information bottleneck |
| **RL Exploration** | Entropy bonus for policy diversity |
| **Mutual Information** | Representation learning (InfoNCE) |

---

⬅️ [Back: Cross-Entropy](../01_cross_entropy/) | ➡️ [Next: KL Divergence](../03_kl_divergence/)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=9C27B0&height=80&section=footer" width="100%"/>
</p>
