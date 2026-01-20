<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=9C27B0&height=120&section=header&text=Mutual%20Information&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-03-9C27B0?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/mutual-information.svg" width="100%">

*Caption: I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X). Measures how much knowing X reduces uncertainty about Y.*

---

## 📂 Overview

Mutual information is the gold standard for measuring statistical dependence - it captures all types of relationships, not just linear ones. It's central to InfoGAN, contrastive learning, and feature selection.

---

## 📐 Definition

### Mutual Information

$$
I(X; Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)
= H(X) + H(Y) - H(X, Y)
= \mathbb{E}_{p(x,y)}\left[\log\frac{p(x,y)}{p(x)p(y)}\right]
= D_{KL}(p(x,y) \| p(x)p(y))
$$

### Properties

1. **Non-negativity:** $I(X; Y) \geq 0$
2. **Zero iff independent:** $I(X; Y) = 0 \iff X \perp Y$
3. **Symmetry:** $I(X; Y) = I(Y; X)$
4. **Self-information:** $I(X; X) = H(X)$

---

## 📐 Proof of Key Properties

### Theorem: $I(X; Y) \geq 0$

**Proof:**

$$
I(X; Y) = D_{KL}(p(x,y) \| p(x)p(y)) \geq 0
$$

by Gibbs' inequality (KL divergence is non-negative). $\quad \blacksquare$

### Theorem: $I(X; Y) = 0 \iff X \perp Y$

**Proof:**

$I(X; Y) = 0$ iff $D\_{KL}(p(x,y) \| p(x)p(y)) = 0$

iff $p(x,y) = p(x)p(y)$ for all $x, y$

iff $X$ and $Y$ are independent. $\quad \blacksquare$

---

## 📐 Alternative Expressions

### In Terms of Entropies

$$
I(X; Y) = H(X) - H(X|Y)
= H(Y) - H(Y|X)
= H(X) + H(Y) - H(X,Y)
$$

### Venn Diagram Interpretation

```
    H(X)             H(Y)
   +---------+   +---------+
   |         |   |         |
   |    H(X|Y)|I(X;Y)|H(Y|X) |
   |         |   |         |
   +---------+   +---------+
         +----+----+
           H(X,Y)
```

---

## 📐 Conditional Mutual Information

### Definition

$$
I(X; Y | Z) = H(X|Z) - H(X|Y, Z)
= \mathbb{E}_{p(z)}[I(X; Y | Z = z)]
$$

### Chain Rule for Mutual Information

$$
I(X_1, X_2, \ldots, X_n; Y) = \sum_{i=1}^{n} I(X_i; Y | X_1, \ldots, X_{i-1})
$$

---

## 📐 Data Processing Inequality

### Theorem

For a Markov chain $X \to Y \to Z$:

$$
I(X; Z) \leq I(X; Y)
$$

**Interpretation:** Processing cannot increase information.

### Proof

$$
I(X; Y, Z) = I(X; Z) + I(X; Y | Z)
= I(X; Y) + I(X; Z | Y)
$$

For Markov chain $X \to Y \to Z$: $I(X; Z | Y) = 0$

Therefore:

$$
I(X; Z) + I(X; Y | Z) = I(X; Y)
$$

Since $I(X; Y | Z) \geq 0$:

$$
I(X; Z) \leq I(X; Y) \quad \blacksquare
$$

---

## 📐 Mutual Information for Gaussians

### Theorem

For jointly Gaussian $(X, Y)$ with correlation $\rho$:

$$
I(X; Y) = -\frac{1}{2}\log(1 - \rho^2)
$$

### Proof

For bivariate Gaussian:

$$
H(X, Y) = \frac{1}{2}\log((2\pi e)^2 |\boldsymbol{\Sigma}|) = \frac{1}{2}\log((2\pi e)^2 \sigma_X^2 \sigma_Y^2 (1 - \rho^2))
H(X) = \frac{1}{2}\log(2\pi e \sigma_X^2)
H(Y) = \frac{1}{2}\log(2\pi e \sigma_Y^2)
$$

Therefore:

$$
I(X; Y) = H(X) + H(Y) - H(X, Y) = -\frac{1}{2}\log(1 - \rho^2) \quad \blacksquare
$$

---

## 📐 InfoNCE Loss

### Contrastive Learning Connection

**InfoNCE Loss:**

$$
\mathcal{L}_{NCE} = -\mathbb{E}\left[\log\frac{f(x, y^+)}{f(x, y^+) + \sum_{j=1}^{N-1} f(x, y_j^-)}\right]
$$

where $y^+$ is the positive sample and $y\_j^-$ are negative samples.

### Lower Bound on Mutual Information

$$
I(X; Y) \geq \log(N) - \mathcal{L}_{NCE}
$$

**Implication:** Minimizing InfoNCE maximizes a lower bound on MI.

---

## 📐 Estimation Methods

### 1. Binning (Discrete)

For discrete variables:

$$
\hat{I}(X; Y) = \sum_{x, y} \hat{p}(x, y) \log\frac{\hat{p}(x, y)}{\hat{p}(x)\hat{p}(y)}
$$

### 2. MINE (Mutual Information Neural Estimation)

$$
I(X; Y) = \sup_T \mathbb{E}_{p(x,y)}[T] - \log\mathbb{E}_{p(x)p(y)}[e^T]
$$

where $T$ is a neural network.

### 3. Variational Bounds

**Lower bound (ELBO-style):**

$$
I(X; Y) \geq \mathbb{E}_{p(x,y)}[\log q(y|x)] + H(Y)
$$

---

## 💻 Code Examples

```python
import numpy as np
from sklearn.feature_selection import mutual_info_classif
import torch
import torch.nn as nn
import torch.nn.functional as F

# Discrete MI estimation
X = np.random.randint(0, 10, (1000, 5))
y = np.random.randint(0, 2, 1000)
mi_scores = mutual_info_classif(X, y)
print(f"MI scores: {mi_scores}")

# MI for Gaussian
def gaussian_mutual_info(rho):
    """I(X;Y) = -0.5 * log(1 - ρ²) for bivariate Gaussian"""
    return -0.5 * np.log(1 - rho**2)

print(f"MI for ρ=0.5: {gaussian_mutual_info(0.5):.4f}")
print(f"MI for ρ=0.9: {gaussian_mutual_info(0.9):.4f}")
print(f"MI for ρ=0.0: {gaussian_mutual_info(0.0):.4f}")  # = 0

# InfoNCE Loss (SimCLR, CLIP)
def info_nce_loss(z1, z2, temperature=0.5):
    """
    z1, z2: embeddings of positive pairs [batch_size, dim]
    """
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # [2*batch_size, dim]
    
    # Compute similarity matrix
    z_norm = F.normalize(z, dim=1)
    sim = z_norm @ z_norm.T / temperature  # [2*batch_size, 2*batch_size]
    
    # Mask out self-similarity
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float('-inf'))
    
    # Positive pairs: (i, i+batch_size) and (i+batch_size, i)
    labels = torch.arange(batch_size, device=z.device)
    labels = torch.cat([labels + batch_size, labels])
    
    return F.cross_entropy(sim, labels)

# MINE (Mutual Information Neural Estimation)
class MINE(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x, y):
        """
        Compute MINE lower bound on I(X; Y)
        """

        # Joint samples (x, y) from p(x, y)
        joint = self.net(torch.cat([x, y], dim=1))
        
        # Marginal samples: shuffle y to break dependency
        y_shuffle = y[torch.randperm(y.size(0))]
        marginal = self.net(torch.cat([x, y_shuffle], dim=1))
        
        # MINE objective: E[T] - log(E[exp(T)])
        mi_lower_bound = joint.mean() - torch.log(torch.exp(marginal).mean() + 1e-8)
        return mi_lower_bound

def train_mine(X, Y, epochs=1000, lr=1e-3):
    """Train MINE to estimate I(X; Y)"""
    mine = MINE(X.shape[1], Y.shape[1])
    optimizer = torch.optim.Adam(mine.parameters(), lr=lr)
    
    X_t = torch.FloatTensor(X)
    Y_t = torch.FloatTensor(Y)
    
    for epoch in range(epochs):
        mi_estimate = mine(X_t, Y_t)
        loss = -mi_estimate  # Maximize MI
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return mi_estimate.item()

# Example: Estimate MI between correlated Gaussians
rho = 0.7
cov = np.array([[1, rho], [rho, 1]])
samples = np.random.multivariate_normal([0, 0], cov, 1000)
X, Y = samples[:, :1], samples[:, 1:]

theoretical_mi = gaussian_mutual_info(rho)
estimated_mi = train_mine(X, Y)
print(f"Theoretical MI: {theoretical_mi:.4f}")
print(f"Estimated MI (MINE): {estimated_mi:.4f}")
```

---

## 🌍 ML Applications

| Application | How MI is Used |
|-------------|----------------|
| **Feature Selection** | Select features with high $I(X\_i; Y)$ |
| **Contrastive Learning** | InfoNCE maximizes MI lower bound |
| **InfoGAN** | Maximize MI between latent code and output |
| **Information Bottleneck** | Trade-off: $I(X; Z)$ vs $I(Z; Y)$ |
| **Representation Learning** | Learn representations that preserve MI |
| **Neural Compression** | Rate-distortion: minimize $I(X; Z)$ subject to quality |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | InfoNCE Paper | [arXiv](https://arxiv.org/abs/1807.03748) |
| 📄 | MINE Paper | [arXiv](https://arxiv.org/abs/1801.04062) |
| 📖 | Cover & Thomas | [Book](https://www.wiley.com/en-us/Elements+of+Information+Theory%2C+2nd+Edition-p-9780471241959) |
| 🇨🇳 | 互信息详解 | [知乎](https://zhuanlan.zhihu.com/p/26486223) |
| 🇨🇳 | 对比学习原理 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88619088) |

---

⬅️ [Back: KL Divergence](../03_kl_divergence/) | ➡️ [Back: Information Theory](../)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=9C27B0&height=80&section=footer" width="100%"/>
</p>
