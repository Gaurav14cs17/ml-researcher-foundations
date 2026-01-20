<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=120&section=header&text=Self-Supervised%20Learning&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-06-45B7D1?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/contrastive-learning-complete.svg" width="100%">

*Caption: Self-supervised learning creates supervisory signals from the data itself. Contrastive methods pull similar pairs together and push dissimilar pairs apart.*

---

## 📐 Mathematical Foundations

### Core Idea

**Traditional Supervised:** Learn $f: x \rightarrow y$ using labeled pairs $(x, y)$

**Self-Supervised:** Learn representations without labels by solving pretext tasks.

The learned representations transfer to downstream tasks.

---

## 📐 Contrastive Learning

### InfoNCE Loss (NT-Xent)

For query $q$, positive key $k^+$, and negative keys $\{k^-\}$:

```math
\mathcal{L}_{InfoNCE} = -\log \frac{\exp(\text{sim}(q, k^+)/\tau)}{\exp(\text{sim}(q, k^+)/\tau) + \sum_{k^-} \exp(\text{sim}(q, k^-)/\tau)}

```

Where:

- $\text{sim}(u, v) = \frac{u^\top v}{\|u\| \|v\|}$ (cosine similarity)

- $\tau$ = temperature parameter

### Temperature's Effect

```math
\text{softmax}_\tau(z_i) = \frac{\exp(z_i/\tau)}{\sum_j \exp(z_j/\tau)}

```

- $\tau \rightarrow 0$: Hard selection (argmax)

- $\tau \rightarrow \infty$: Uniform distribution

- $\tau \approx 0.07-0.5$: Typical range

### Gradient Analysis

For positive pair $(i, j)$:

```math
\frac{\partial \mathcal{L}}{\partial z_i} = \frac{1}{\tau}\left(p_{ij} - 1\right) z_j + \frac{1}{\tau}\sum_{k \neq j} p_{ik} z_k

```

Where $p\_{ij}$ is the softmax probability.

**Interpretation:** Push $z\_i$ toward $z\_j$, away from negatives.

---

## 📐 SimCLR Framework

### Architecture

```
Image x
    ↓
Augmentation → x̃₁    Augmentation → x̃₂
    ↓                      ↓
Encoder f(·)           Encoder f(·)  (shared weights)
    ↓                      ↓
   h₁                     h₂
    ↓                      ↓
Projection g(·)       Projection g(·)
    ↓                      ↓
   z₁                     z₂
    ↓                      ↓
        Contrastive Loss

```

### Loss for Batch of N Pairs

```math
\mathcal{L} = \frac{1}{2N}\sum_{i=1}^{N}[\ell(2i-1, 2i) + \ell(2i, 2i-1)]

```

Where:

```math
\ell(i, j) = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k=1}^{2N} \mathbf{1}_{k \neq i} \exp(\text{sim}(z_i, z_k)/\tau)}

```

### Augmentation Importance

Effective augmentations:

- Random crop + resize

- Color distortion (jitter, grayscale)

- Gaussian blur

- Random flip

**Key finding:** Strong augmentation crucial for good representations.

---

## 📐 MoCo (Momentum Contrast)

### Problem with SimCLR

Large batch size required for sufficient negatives (4096+).

### Solution: Memory Queue

Maintain a queue of $K$ encoded keys from previous batches.

### Momentum Encoder

```math
\theta_k \leftarrow m \cdot \theta_k + (1-m) \cdot \theta_q

```

Where $m \approx 0.999$ provides slowly evolving keys.

### Loss

```math
\mathcal{L} = -\log \frac{\exp(q \cdot k^+/\tau)}{\exp(q \cdot k^+/\tau) + \sum_{i=0}^{K} \exp(q \cdot k_i/\tau)}

```

---

## 📐 BYOL (Bootstrap Your Own Latent)

### Key Innovation: No Negatives!

**Online network:** $\theta$
**Target network:** $\xi$ (EMA of $\theta$)

```math
\xi \leftarrow m \cdot \xi + (1-m) \cdot \theta

```

### Loss

```math
\mathcal{L} = 2 - 2 \cdot \frac{\langle q_\theta(z_1), z'_2 \rangle}{\|q_\theta(z_1)\| \cdot \|z'_2\|}

```

Where $q\_\theta$ is a predictor network.

**Why doesn't it collapse?**
- Asymmetry between online and target

- Target provides stable targets

- Predictor prevents trivial solutions

---

## 📐 Masked Prediction

### BERT (NLP)

**Pretext task:** Predict masked tokens.

```math
\mathcal{L}_{MLM} = -\mathbb{E}\left[\sum_{i \in \mathcal{M}} \log p(x_i | x_{\backslash \mathcal{M}})\right]

```

Where $\mathcal{M}$ is the set of masked positions.

**Masking strategy:**
- 15% of tokens selected

- 80% → [MASK]

- 10% → random token

- 10% → unchanged

### MAE (Masked Autoencoder for Vision)

**Pretext task:** Reconstruct masked image patches.

```math
\mathcal{L}_{MAE} = \frac{1}{|\mathcal{M}|}\sum_{i \in \mathcal{M}} \|x_i - \hat{x}_i\|^2

```

**Key design:**
- High masking ratio (75%)

- Asymmetric encoder-decoder

- Encoder only sees unmasked patches

---

## 📐 CLIP (Contrastive Language-Image Pretraining)

### Dual-Encoder Architecture

```math
\text{sim}(I, T) = \frac{f_I(I)^\top f_T(T)}{\|f_I(I)\| \|f_T(T)\|}

```

### Contrastive Loss

```math
\mathcal{L}_{CLIP} = \frac{1}{2}\left(\mathcal{L}_{I \rightarrow T} + \mathcal{L}_{T \rightarrow I}\right)
\mathcal{L}_{I \rightarrow T} = -\frac{1}{N}\sum_i \log \frac{\exp(\text{sim}(I_i, T_i)/\tau)}{\sum_j \exp(\text{sim}(I_i, T_j)/\tau)}

```

### Zero-Shot Classification

For image $I$ and class names $\{c\_1, ..., c\_K\}$:

```math
p(c_k | I) = \frac{\exp(\text{sim}(I, \text{prompt}(c_k))/\tau)}{\sum_j \exp(\text{sim}(I, \text{prompt}(c_j))/\tau)}

```

Where $\text{prompt}(c) = $ "a photo of a {c}"

---

## 💻 Code Examples

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# SimCLR Loss
def simclr_loss(z1, z2, temperature=0.5):
    """
    z1, z2: (batch_size, dim) - two augmented views
    """
    batch_size = z1.shape[0]
    
    # Normalize embeddings
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    
    # Concatenate
    z = torch.cat([z1, z2], dim=0)  # (2*batch, dim)
    
    # Compute similarity matrix
    sim = torch.mm(z, z.t()) / temperature  # (2*batch, 2*batch)
    
    # Mask out self-similarity
    mask = torch.eye(2 * batch_size, device=z.device).bool()
    sim = sim.masked_fill(mask, float('-inf'))
    
    # Labels: positive pairs are at (i, batch+i) and (batch+i, i)
    labels = torch.cat([
        torch.arange(batch_size, 2 * batch_size),
        torch.arange(batch_size)
    ], dim=0).to(z.device)
    
    return F.cross_entropy(sim, labels)

# MoCo-style Queue
class MoCoQueue:
    def __init__(self, dim, K=65536):
        self.K = K
        self.queue = torch.randn(dim, K)
        self.queue = F.normalize(self.queue, dim=0)
        self.ptr = 0
    
    @torch.no_grad()
    def enqueue(self, keys):
        batch_size = keys.shape[0]
        
        if self.ptr + batch_size > self.K:
            self.ptr = 0
        
        self.queue[:, self.ptr:self.ptr + batch_size] = keys.T
        self.ptr += batch_size

# BYOL Loss
def byol_loss(online_proj, target_proj, predictor):
    """
    online_proj: output of online encoder + projector
    target_proj: output of target encoder + projector (no grad)
    predictor: online predictor network
    """
    online_pred = predictor(online_proj)
    
    # Normalize
    online_pred = F.normalize(online_pred, dim=-1)
    target_proj = F.normalize(target_proj, dim=-1)
    
    # Negative cosine similarity
    return 2 - 2 * (online_pred * target_proj).sum(dim=-1).mean()

# Momentum update for EMA
@torch.no_grad()
def momentum_update(online_encoder, target_encoder, m=0.999):
    for param_o, param_t in zip(online_encoder.parameters(), target_encoder.parameters()):
        param_t.data = m * param_t.data + (1 - m) * param_o.data

# CLIP-style loss
def clip_loss(image_embeddings, text_embeddings, temperature=0.07):
    """
    image_embeddings: (batch, dim)
    text_embeddings: (batch, dim)
    """
    # Normalize
    image_embeddings = F.normalize(image_embeddings, dim=-1)
    text_embeddings = F.normalize(text_embeddings, dim=-1)
    
    # Compute similarity
    logits = image_embeddings @ text_embeddings.T / temperature
    
    # Labels: diagonal elements are positive pairs
    labels = torch.arange(len(image_embeddings), device=logits.device)
    
    # Symmetric loss
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    
    return (loss_i2t + loss_t2i) / 2

# MAE Loss
def mae_loss(pred_patches, target_patches, mask):
    """
    pred_patches: (batch, num_patches, patch_dim) - predicted patches
    target_patches: (batch, num_patches, patch_dim) - original patches
    mask: (batch, num_patches) - 1 for masked, 0 for unmasked
    """
    loss = (pred_patches - target_patches) ** 2
    loss = loss.mean(dim=-1)  # Mean over patch dimension
    loss = (loss * mask).sum() / mask.sum()  # Mean over masked patches
    return loss

```

---

## 📊 Comparison Table

| Method | Negatives | Batch Size | Key Innovation |
|--------|-----------|------------|----------------|
| **SimCLR** | In-batch | Large (4096+) | Strong augmentations |
| **MoCo** | Queue | Small (256) | Momentum encoder |
| **BYOL** | None | Medium | Predictor + EMA |
| **SimSiam** | None | Medium | Stop-gradient |
| **MAE** | N/A | Medium | High masking ratio |
| **CLIP** | In-batch | Large | Multi-modal |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | SimCLR | [arXiv](https://arxiv.org/abs/2002.05709) |
| 📄 | MoCo | [arXiv](https://arxiv.org/abs/1911.05722) |
| 📄 | BYOL | [arXiv](https://arxiv.org/abs/2006.07733) |
| 📄 | MAE | [arXiv](https://arxiv.org/abs/2111.06377) |
| 📄 | CLIP | [arXiv](https://arxiv.org/abs/2103.00020) |
| 📄 | BERT | [arXiv](https://arxiv.org/abs/1810.04805) |
| 🎥 | Yann LeCun: Self-Supervised Learning | [YouTube](https://www.youtube.com/watch?v=8L10w1KoOU8) |
| 🇨🇳 | 自监督学习详解 | [知乎](https://zhuanlan.zhihu.com/p/108906502) |

---

## 🔗 Applications

| Domain | Method | Application |
|--------|--------|-------------|
| **Vision** | SimCLR, MoCo, MAE | Pretrain for classification, detection |
| **NLP** | BERT, GPT | Pretrain for NLU, generation |
| **Multi-modal** | CLIP | Zero-shot classification, retrieval |
| **Speech** | Wav2Vec | Speech recognition |
| **Video** | VideoMAE | Action recognition |

---

⬅️ [Back: Data Augmentation](../08_data_augmentation/README.md) | ➡️ [Next: NAS](../10_neural_architecture_search/README.md)

---

⬅️ [Back: Main](../README.md)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=80&section=footer" width="100%"/>
</p>
