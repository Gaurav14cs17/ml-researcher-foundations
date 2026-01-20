<!-- Navigation -->
<p align="center">
  <a href="../03_kernel_methods/">⬅️ Prev: Kernel Methods</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../">📚 ML Theory</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../05_risk_minimization/">Next: Risk Minimization ➡️</a>
</p>

---

<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=120&section=header&text=Representation%20Learning&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-05-4ECDC4?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/representation.svg" width="100%">

*Caption: Representation learning transforms raw data into useful features automatically.*

---

## 📂 Overview

**Representation learning** is about learning features from data that make downstream tasks easier. It replaces hand-crafted feature engineering with learned representations.

---

## 📐 Mathematical Framework

### What is a Good Representation?

A good representation $z = f_\theta(x)$ should:

1. **Task-relevant information:** $I(Z; Y)$ is high
2. **Invariance:** $Z$ unchanged under irrelevant transformations
3. **Disentanglement:** Factors of variation are separated
4. **Compactness:** $\dim(Z) \ll \dim(X)$

### Information Bottleneck Principle

$$
\max_\theta I(Z; Y) - \beta I(Z; X)
$$

where:
- $I(Z; Y)$: Information about labels (maximize)
- $I(Z; X)$: Information from input (compress)
- $\beta$: Trade-off parameter

---

## 📂 Topics in This Section

| Folder | Topic | Key Concepts |
|--------|-------|--------------|
| [01_embeddings/](./01_embeddings/) | Embeddings | Word2Vec, CLIP, contrastive |
| [02_feature_learning/](./02_feature_learning/) | Feature Learning | CNNs, hierarchical features |
| [03_transfer/](./03_transfer/) | Transfer Learning | Pre-training, fine-tuning, LoRA |

---

## 📐 Paradigms

### Supervised Representation Learning

Learn features jointly with task:

$$
\min_{\theta, \phi} \mathbb{E}[\ell(g_\phi(f_\theta(x)), y)]
$$

### Self-Supervised Learning

Learn from data structure without labels:
- **Contrastive:** SimCLR, MoCo, CLIP
- **Masked:** BERT, MAE
- **Autoregressive:** GPT

### Disentangled Representations

**β-VAE objective:**

$$
\mathcal{L} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - \beta D_{KL}(q(z|x) \| p(z))
$$

Large $\beta$ encourages disentanglement.

---

## 💻 Code Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class RepresentationLearner(nn.Module):
    """
    Generic representation learning framework.
    """
    
    def __init__(self, encoder, projector=None):
        super().__init__()
        self.encoder = encoder
        self.projector = projector
    
    def encode(self, x):
        """Get representation z = f(x)."""
        return self.encoder(x)
    
    def project(self, z):
        """Project for contrastive learning."""
        if self.projector is not None:
            return self.projector(z)
        return z
    
    def forward(self, x):
        z = self.encode(x)
        return self.project(z)

class ContrastiveRepLearning(RepresentationLearner):
    """
    Contrastive representation learning (SimCLR-style).
    
    L = -log(exp(sim(zᵢ, zⱼ)/τ) / Σexp(sim(zᵢ, zₖ)/τ))
    """
    
    def __init__(self, encoder, hidden_dim=2048, proj_dim=128, temperature=0.07):
        projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim)
        )
        super().__init__(encoder, projector)
        self.temperature = temperature
    
    def contrastive_loss(self, z1, z2):
        """InfoNCE loss for two views."""
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        batch_size = z1.size(0)
        z = torch.cat([z1, z2], dim=0)
        
        sim = z @ z.T / self.temperature
        mask = torch.eye(2 * batch_size, device=z.device).bool()
        sim.masked_fill_(mask, float('-inf'))
        
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size),
            torch.arange(batch_size)
        ]).to(z.device)
        
        return F.cross_entropy(sim, labels)

class MaskedAutoencoder(nn.Module):
    """
    Masked Autoencoder for representation learning.
    
    L = ||x_masked - decoder(encoder(x_visible))||²
    """
    
    def __init__(self, encoder, decoder, mask_ratio=0.75):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mask_ratio = mask_ratio
    
    def random_masking(self, x, mask_ratio):
        """Randomly mask patches."""
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))
        
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)
        
        return x_masked, mask, ids_restore
    
    def forward(self, x):

        # Patchify and mask
        x_masked, mask, ids_restore = self.random_masking(x, self.mask_ratio)
        
        # Encode visible patches
        z = self.encoder(x_masked)
        
        # Decode all patches
        x_recon = self.decoder(z, ids_restore)
        
        # Reconstruction loss on masked patches
        loss = (x_recon - x) ** 2
        loss = (loss.mean(dim=-1) * mask).sum() / mask.sum()
        
        return loss, x_recon

# Extract representations from pretrained models
def extract_features(model, dataloader, device='cuda'):
    """Extract features from a pretrained model."""
    model.eval()
    features = []
    labels = []
    
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            z = model.encode(x)
            features.append(z.cpu())
            labels.append(y)
    
    return torch.cat(features), torch.cat(labels)
```

---

## 📊 Representation Quality Metrics

| Metric | Measures | How |
|--------|----------|-----|
| Linear Probe | Linear separability | Train linear classifier |
| k-NN | Neighborhood quality | k-NN classification |
| Alignment | Positive pair similarity | Mean cosine similarity |
| Uniformity | Distribution on hypersphere | Log of average distance |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | Representation Learning Review | [Bengio et al.](https://arxiv.org/abs/1206.5538) |
| 📄 | SimCLR | [Chen et al.](https://arxiv.org/abs/2002.05709) |
| 📄 | MAE | [He et al.](https://arxiv.org/abs/2111.06377) |

---

⬅️ [Back: Kernel Methods](../03_kernel_methods/) | ➡️ [Next: Risk Minimization](../05_risk_minimization/)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<!-- Navigation -->
<p align="center">
  <a href="../03_kernel_methods/">⬅️ Prev: Kernel Methods</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../">📚 ML Theory</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../05_risk_minimization/">Next: Risk Minimization ➡️</a>
</p>

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=80&section=footer" width="100%"/>
</p>
