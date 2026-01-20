<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=120&section=header&text=Loss%20Functions&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-06-45B7D1?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/loss-functions-complete.svg" width="100%">

*Caption: Loss functions measure prediction error. Cross-entropy for classification, MSE for regression, contrastive for embeddings.*

---

## 📐 Mathematical Foundations

### Maximum Likelihood Connection

Most loss functions arise from Maximum Likelihood Estimation (MLE).

Given data $\{(x_i, y_i)\}_{i=1}^n$ and model $p_\theta(y|x)$:

```math
\theta^* = \arg\max_\theta \prod_{i=1}^n p_\theta(y_i|x_i)

```

Taking negative log:

```math
\theta^* = \arg\min_\theta -\sum_{i=1}^n \log p_\theta(y_i|x_i)

```

**This is the Negative Log-Likelihood (NLL) loss!**

---

## 📊 Classification Losses

### Binary Cross-Entropy (BCE)

**Model:** $p(y=1|x) = \sigma(f_\theta(x))$ where $\sigma(z) = \frac{1}{1+e^{-z}}$

**Loss:**

```math
\mathcal{L}_{BCE} = -\frac{1}{n}\sum_{i=1}^n \left[y_i \log(p_i) + (1-y_i) \log(1-p_i)\right]

```

**Gradient (w.r.t. logit $z$):**

```math
\frac{\partial \mathcal{L}}{\partial z} = \sigma(z) - y = p - y

```

**Proof:**

```math
\frac{\partial}{\partial z}[-y\log\sigma(z) - (1-y)\log(1-\sigma(z))]
= -\frac{y}{\sigma(z)}\sigma'(z) + \frac{1-y}{1-\sigma(z)}\sigma'(z)
= -\frac{y}{\sigma(z)}\sigma(z)(1-\sigma(z)) + \frac{1-y}{1-\sigma(z)}\sigma(z)(1-\sigma(z))
= -y(1-\sigma(z)) + (1-y)\sigma(z) = \sigma(z) - y \quad \checkmark

```

### Multi-class Cross-Entropy

**Model:** $p(y=c|x) = \text{softmax}(f_\theta(x))_c = \frac{e^{z_c}}{\sum_j e^{z_j}}$

**Loss:**

```math
\mathcal{L}_{CE} = -\frac{1}{n}\sum_{i=1}^n \sum_{c=1}^C y_{ic} \log(p_{ic})

```

For one-hot labels (only true class $c^*$ has $y_{c^*}=1$):

```math
\mathcal{L}_{CE} = -\frac{1}{n}\sum_{i=1}^n \log(p_{i,c^*_i})

```

**Gradient (w.r.t. logit $z_c$):**

```math
\frac{\partial \mathcal{L}}{\partial z_c} = p_c - y_c = \text{softmax}(z)_c - y_c

```

### Focal Loss (for Imbalanced Data)

```math
\mathcal{L}_{focal} = -\alpha_t (1-p_t)^\gamma \log(p_t)

```

Where:

- $p_t = p$ if $y=1$, else $1-p$

- $\alpha_t$ = class weight

- $\gamma$ = focusing parameter (typically 2)

**Intuition:** Down-weights easy examples, focuses on hard ones.

When $\gamma=0$: Standard cross-entropy
When $\gamma>0$: Easy examples ($p_t \approx 1$) contribute less

### Label Smoothing

Instead of hard labels $y \in \{0, 1\}$:

```math
y_{smooth} = (1-\epsilon)y + \frac{\epsilon}{C}

```

**Effect:** Prevents overconfident predictions, improves calibration.

---

## 📊 Regression Losses

### Mean Squared Error (MSE / L2)

```math
\mathcal{L}_{MSE} = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2

```

**Connection to MLE:** Assumes Gaussian noise

```math
y = f_\theta(x) + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)

```

**Gradient:**

```math
\frac{\partial \mathcal{L}}{\partial \hat{y}} = \frac{2}{n}(\hat{y} - y)

```

**Properties:**
- Penalizes large errors heavily (quadratic)

- Sensitive to outliers

### Mean Absolute Error (MAE / L1)

```math
\mathcal{L}_{MAE} = \frac{1}{n}\sum_{i=1}^n |y_i - \hat{y}_i|

```

**Connection to MLE:** Assumes Laplace noise

```math
p(y|x) \propto \exp\left(-\frac{|y - f_\theta(x)|}{b}\right)

```

**Gradient:**

```math
\frac{\partial \mathcal{L}}{\partial \hat{y}} = \text{sign}(\hat{y} - y)

```

**Properties:**
- Constant gradient magnitude

- More robust to outliers than MSE

### Huber Loss (Smooth L1)

```math
\mathcal{L}_{Huber} = \begin{cases}
\frac{1}{2}(y-\hat{y})^2 & \text{if } |y-\hat{y}| < \delta \\
\delta |y-\hat{y}| - \frac{1}{2}\delta^2 & \text{otherwise}
\end{cases}

```

**Properties:**
- MSE for small errors (smooth)

- MAE for large errors (robust)

- Continuous derivative everywhere

---

## 📊 Contrastive & Metric Learning Losses

### Triplet Loss

Given anchor $a$, positive $p$, negative $n$:

```math
\mathcal{L}_{triplet} = \max(0, d(a,p) - d(a,n) + \text{margin})

```

Where $d(\cdot, \cdot)$ is a distance function (e.g., L2).

**Goal:** $d(a,p) + \text{margin} < d(a,n)$

### Contrastive Loss (Siamese)

```math
\mathcal{L}_{contrastive} = y \cdot d^2 + (1-y) \cdot \max(0, m - d)^2

```

Where:

- $y=1$ for similar pairs, $y=0$ for dissimilar

- $m$ = margin

### InfoNCE (NT-Xent) Loss

```math
\mathcal{L}_{InfoNCE} = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k=1}^{2N} \mathbf{1}_{k \neq i} \exp(\text{sim}(z_i, z_k)/\tau)}

```

Where:

- $(z_i, z_j)$ = positive pair

- $\tau$ = temperature

- $\text{sim}(u,v) = \frac{u^\top v}{\|u\| \|v\|}$ (cosine similarity)

**Used in:** SimCLR, CLIP, contrastive learning

---

## 📊 Generative Model Losses

### VAE Loss (ELBO)

```math
\mathcal{L}_{VAE} = \underbrace{\mathbb{E}_{q(z|x)}[\log p(x|z)]}_{\text{Reconstruction}} - \underbrace{D_{KL}(q(z|x) \| p(z))}_{\text{KL Regularization}}

```

For Gaussian decoder and prior:

```math
= -\frac{1}{2}\|x - \hat{x}\|^2 - \frac{1}{2}\sum_j \left(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2\right)

```

### GAN Losses

**Vanilla GAN:**

```math
\min_G \max_D \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]

```

**WGAN (Wasserstein):**

```math
\min_G \max_{D \in \mathcal{D}} \mathbb{E}_{x \sim p_{data}}[D(x)] - \mathbb{E}_{z \sim p_z}[D(G(z))]

```

Where $\mathcal{D}$ is the set of 1-Lipschitz functions.

---

## 🎯 Loss Selection Guide

| Task | Loss Function | When to Use |
|------|---------------|-------------|
| **Binary Classification** | BCE | Two classes |
| **Multi-class** | Cross-Entropy | Mutually exclusive classes |
| **Multi-label** | BCE per label | Multiple labels per sample |
| **Regression** | MSE | Gaussian errors |
| **Robust Regression** | Huber/MAE | Outliers present |
| **Embedding** | Triplet/InfoNCE | Similarity learning |
| **VAE** | ELBO | Generation with latent |
| **GAN** | Adversarial | High-fidelity generation |
| **Imbalanced** | Focal Loss | Class imbalance |

---

## 💻 Code Examples

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============== Classification ==============

# Binary Cross-Entropy (with logits for numerical stability)
bce_loss = nn.BCEWithLogitsLoss()
logits = model(x)  # Raw outputs (before sigmoid)
loss = bce_loss(logits, targets.float())

# Multi-class Cross-Entropy
ce_loss = nn.CrossEntropyLoss()
logits = model(x)  # Shape: (batch, num_classes)
targets = labels   # Shape: (batch,) containing class indices
loss = ce_loss(logits, targets)

# With label smoothing
ce_smooth = nn.CrossEntropyLoss(label_smoothing=0.1)

# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # p_t
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# ============== Regression ==============

# MSE (L2)
mse_loss = nn.MSELoss()
loss = mse_loss(predictions, targets)

# MAE (L1)
mae_loss = nn.L1Loss()
loss = mae_loss(predictions, targets)

# Huber (Smooth L1)
huber_loss = nn.SmoothL1Loss(beta=1.0)
loss = huber_loss(predictions, targets)

# ============== Contrastive ==============

def triplet_loss(anchor, positive, negative, margin=1.0):
    d_pos = F.pairwise_distance(anchor, positive)
    d_neg = F.pairwise_distance(anchor, negative)
    loss = F.relu(d_pos - d_neg + margin)
    return loss.mean()

def info_nce_loss(z_i, z_j, temperature=0.5):
    """
    InfoNCE loss for contrastive learning
    z_i, z_j: (batch_size, dim) - two views of same samples
    """
    batch_size = z_i.shape[0]
    
    # Normalize
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)
    
    # Concatenate
    z = torch.cat([z_i, z_j], dim=0)  # (2*batch, dim)
    
    # Compute similarity matrix
    sim = torch.mm(z, z.t()) / temperature  # (2*batch, 2*batch)
    
    # Create labels: positive pairs are at positions (i, batch+i) and (batch+i, i)
    labels = torch.cat([torch.arange(batch_size) + batch_size,
                        torch.arange(batch_size)]).to(z.device)
    
    # Mask out self-similarity
    mask = ~torch.eye(2 * batch_size, dtype=bool, device=z.device)
    sim = sim.masked_select(mask).view(2 * batch_size, -1)
    
    # Adjust labels
    labels = labels - (torch.arange(2 * batch_size, device=z.device) > labels).long()
    
    return F.cross_entropy(sim, labels)

# ============== VAE ==============

def vae_loss(recon_x, x, mu, log_var):
    """
    VAE loss = Reconstruction + KL divergence
    """
    # Reconstruction loss (BCE for binary, MSE for continuous)
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    return recon_loss + kl_loss

```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | Focal Loss Paper | [arXiv](https://arxiv.org/abs/1708.02002) |
| 📄 | InfoNCE Paper | [arXiv](https://arxiv.org/abs/1807.03748) |
| 📄 | Label Smoothing | [arXiv](https://arxiv.org/abs/1512.00567) |
| 📄 | VAE Paper | [arXiv](https://arxiv.org/abs/1312.6114) |
| 📄 | WGAN Paper | [arXiv](https://arxiv.org/abs/1701.07875) |
| 🇨🇳 | 损失函数详解 | [知乎](https://zhuanlan.zhihu.com/p/35709485) |

---

## 🔗 Where Loss Functions Are Used

| Application | Loss Function |
|-------------|---------------|
| **Image Classification** | Cross-Entropy |
| **Object Detection** | Focal Loss + IoU |
| **Language Modeling** | Cross-Entropy (next token) |
| **Contrastive Learning** | InfoNCE |
| **VAE** | Reconstruction + KL |
| **GAN** | Adversarial losses |
| **Regression** | MSE/Huber |

---

➡️ [Next: Transfer Learning](../07_transfer_learning/README.md)

---

⬅️ [Back: Main](../README.md)

---

➡️ [Next: Transfer Learning](../07_transfer_learning/README.md)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=80&section=footer" width="100%"/>
</p>
