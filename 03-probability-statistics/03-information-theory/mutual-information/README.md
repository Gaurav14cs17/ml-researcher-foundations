# Mutual Information

> **Quantifying shared information between variables**

---

## 🎯 Visual Overview

<img src="./images/mutual-information.svg" width="100%">

*Caption: I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X). Measures how much knowing X reduces uncertainty about Y. Symmetric and non-negative. Used in feature selection, InfoNCE, and representation learning.*

---

## 📂 Overview

Mutual information is the gold standard for measuring statistical dependence - it captures all types of relationships, not just linear ones. It's central to InfoGAN, contrastive learning, and feature selection.

---

## 📐 Mathematical Definitions

### Mutual Information
```
I(X; Y) = H(X) - H(X|Y)
        = H(Y) - H(Y|X)
        = H(X) + H(Y) - H(X,Y)
        = E[log(p(x,y)/(p(x)p(y)))]

Properties:
• I(X; Y) ≥ 0
• I(X; Y) = 0 ⟺ X, Y independent
• I(X; Y) = I(Y; X)  (symmetric)
• I(X; X) = H(X)
```

### KL Divergence Form
```
I(X; Y) = D_KL(p(x,y) || p(x)p(y))

= ∫∫ p(x,y) log(p(x,y)/(p(x)p(y))) dx dy
```

### Conditional Mutual Information
```
I(X; Y | Z) = H(X|Z) - H(X|Y,Z)
            = E_Z[I(X; Y)|Z]

Data Processing Inequality:
X → Y → Z ⟹ I(X; Z) ≤ I(X; Y)
```

### InfoNCE Loss (Contrastive Learning)
```
L_NCE = -E[log(f(x,y⁺) / (f(x,y⁺) + Σᵢf(x,yᵢ⁻)))]

Lower bound on I(X; Y):
I(X; Y) ≥ log(N) - L_NCE

Where N = number of negatives + 1
```

---

## 💻 Code Examples

```python
import numpy as np
from sklearn.feature_selection import mutual_info_classif
import torch
import torch.nn.functional as F

# Discrete MI estimation
X = np.random.randint(0, 10, (1000, 5))
y = np.random.randint(0, 2, 1000)
mi_scores = mutual_info_classif(X, y)  # MI between each feature and y

# InfoNCE Loss (SimCLR, CLIP)
def info_nce_loss(z1, z2, temperature=0.5):
    """
    z1, z2: embeddings of positive pairs [batch_size, dim]
    """
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    
    # Compute similarity matrix
    sim = F.cosine_similarity(z.unsqueeze(0), z.unsqueeze(1), dim=2)
    sim = sim / temperature
    
    # Positive pairs: (i, i+batch_size)
    labels = torch.arange(batch_size, device=z.device)
    labels = torch.cat([labels + batch_size, labels])
    
    # Mask self-similarity
    mask = torch.eye(2 * batch_size, device=z.device).bool()
    sim.masked_fill_(mask, float('-inf'))
    
    return F.cross_entropy(sim, labels)

# MINE (Mutual Information Neural Estimation)
class MINE(nn.Module):
    def __init__(self, x_dim, y_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + y_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )
    
    def forward(self, x, y):
        joint = self.net(torch.cat([x, y], dim=1))
        y_shuffle = y[torch.randperm(y.size(0))]
        marginal = self.net(torch.cat([x, y_shuffle], dim=1))
        return joint.mean() - torch.log(torch.exp(marginal).mean())
```

---

---

## 🔗 Where This Topic Is Used

| Application | Usage |
|-------------|-------|
| **Machine Learning** | Core concept |
| **Deep Learning** | Foundation for neural networks |
| **Research** | Important for understanding papers |

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | InfoNCE Paper | [arXiv](https://arxiv.org/abs/1807.03748) |
| 📄 | MINE Paper | [arXiv](https://arxiv.org/abs/1801.04062) |
| 📖 | Cover & Thomas | [Book](https://www.wiley.com/en-us/Elements+of+Information+Theory%2C+2nd+Edition-p-9780471241959) |
| 🇨🇳 | 互信息详解 | [知乎](https://zhuanlan.zhihu.com/p/26486223) |
| 🇨🇳 | 对比学习原理 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88619088) |
| 🇨🇳 | 信息论基础 | [B站](https://www.bilibili.com/video/BV1R4411V7tZ) |

---

<- [Back](../)

---

⬅️ [Back: mutual-information](../)

---

⬅️ [Back: Kl Divergence](../kl-divergence/)
