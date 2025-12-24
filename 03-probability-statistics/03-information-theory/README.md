# 📡 Information Theory

> **Measuring information and comparing distributions**

<img src="./images/kl-divergence.svg" width="100%">

---

## 📐 Mathematical Foundations

### Entropy (Shannon)
```
Discrete: H(X) = -Σₓ p(x) log₂ p(x)  (bits)
Continuous: h(X) = -∫ f(x) log f(x) dx

Properties:
• H(X) ≥ 0
• H(X) = 0 ⟺ X is deterministic
• H(X) ≤ log|X| (max when uniform)
```

### Cross-Entropy
```
H(p, q) = -Σₓ p(x) log q(x)
        = H(p) + D_KL(p || q)

For classification:
H(p_true, p_model) = -log p_model(y_true)
```

### KL Divergence
```
D_KL(p || q) = Σₓ p(x) log(p(x)/q(x))
             = E_p[log p - log q]

Properties:
• D_KL ≥ 0 (Gibbs inequality)
• D_KL = 0 ⟺ p = q
• NOT symmetric: D_KL(p||q) ≠ D_KL(q||p)
```

### Mutual Information
```
I(X; Y) = H(X) - H(X|Y)
        = H(Y) - H(Y|X)
        = D_KL(p(x,y) || p(x)p(y))

= 0 ⟺ X, Y independent
```

---

## 📂 Topics in This Folder

| Folder | Topics | ML Application |
|--------|--------|----------------|
| [entropy/](./entropy/) | Discrete, differential, max-entropy | Uncertainty measurement |
| [cross-entropy/](./cross-entropy/) | H(p,q), as loss function | 🔥 Classification loss |
| [kl-divergence/](./kl-divergence/) | D_KL, properties, variational | 🔥 VAE, distillation |
| [mutual-information/](./mutual-information/) | I(X;Y), InfoNCE | Feature selection |

---

## 🎯 The Core Ideas

```
Entropy: How much uncertainty/information?
H(X) = -Σ p(x) log p(x)

Cross-entropy: How many bits to encode p using q?
H(p,q) = -Σ p(x) log q(x)

KL Divergence: Extra bits due to using q instead of p
D_KL(p||q) = H(p,q) - H(p) = Σ p(x) log(p(x)/q(x))

Relationship:
H(p,q) = H(p) + D_KL(p||q)
         ----   ------------
         minimum  extra bits
         possible  (always ≥ 0)
```

---

## 🌍 ML Applications

| Concept | Application | Example |
|---------|-------------|---------|
| Cross-entropy | Classification loss | nn.CrossEntropyLoss |
| KL divergence | VAE regularization | KL(q(z\|x) \|\| p(z)) |
| KL divergence | Knowledge distillation | Soft target matching |
| Mutual information | Contrastive learning | InfoNCE loss |
| Entropy | Exploration in RL | Max-entropy RL |

---

## 🔥 Why Cross-Entropy is THE Loss Function

```
Classification: Given true label y, predict P(y|x)

Cross-entropy loss:
L = -log P(y=true|x)

Why this works:
1. Minimizing L = maximizing log P(y|x) = MLE!
2. L = H(p_true, p_model) where p_true is one-hot
3. Gradients are simple: ∂L/∂logit = p_model - p_true
4. Well-calibrated probabilities (not just rankings)

Compare to MSE for classification:
• MSE: (1 - p)² when correct
• CE:  -log(p) when correct
• CE penalizes confident wrong predictions more!
```

---

## 💻 Code Examples

```python
import torch
import torch.nn.functional as F

# Cross-entropy loss
logits = torch.randn(32, 10)  # 32 samples, 10 classes
labels = torch.randint(0, 10, (32,))
ce_loss = F.cross_entropy(logits, labels)

# KL divergence between two distributions
p = F.softmax(torch.randn(32, 10), dim=1)
q = F.softmax(torch.randn(32, 10), dim=1)
kl = F.kl_div(q.log(), p, reduction='batchmean')  # D_KL(p||q)

# Entropy
def entropy(p):
    return -(p * torch.log(p + 1e-10)).sum(dim=-1)

H_p = entropy(p).mean()
```

---

## 📚 Resources

| Type | Title | Link |
|------|-------|------|
| 📖 | Elements of Information Theory | Cover & Thomas |
| 🎥 | Information Theory Basics | [3Blue1Brown](https://www.youtube.com/watch?v=v68zYyaEmEA) |
| 📄 | Variational Inference | [arXiv](https://arxiv.org/abs/1601.00670) |

---

## 🔗 Where This Topic Is Used

| Topic | How Info Theory Is Used |
|-------|------------------------|
| **Cross-Entropy Loss** | Classification (nn.CrossEntropyLoss) |
| **VAE** | KL divergence regularization |
| **Knowledge Distillation** | KL between teacher-student |
| **RLHF** | KL penalty in PPO objective |
| **DPO** | Implicit KL in preference loss |
| **Contrastive Learning** | InfoNCE loss (mutual information) |
| **Max-Entropy RL** | Entropy bonus for exploration |
| **Language Models** | Perplexity = exp(cross-entropy) |
| **Compression** | Entropy = minimum bits |
| **Variational Inference** | ELBO = reconstruction - KL |

### Concepts Used In

| Concept | Used By |
|---------|---------|
| **Cross-Entropy** | ALL classification models |
| **KL Divergence** | VAE, RLHF, DPO, distillation |
| **Mutual Information** | Contrastive learning, InfoNCE |
| **Entropy** | RL exploration, uncertainty |

### Prerequisite For

```
Information Theory --> Understanding loss functions
                  --> VAE (ELBO derivation)
                  --> RLHF/DPO (KL penalty)
                  --> Knowledge distillation
                  --> Contrastive learning
```

---

⬅️ [Back: 02-Multivariate](../02-multivariate/) | ➡️ [Next: 04-Estimation](../04-estimation/)

