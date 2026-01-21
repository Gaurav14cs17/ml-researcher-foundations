<!-- Navigation -->
<p align="center">
  <a href="../">⬅️ Back: Learning Frameworks</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../02_supervised/">Next: Supervised ➡️</a>
</p>

---

<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=120&section=header&text=Self-Supervised%20Learning&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-05-4ECDC4?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/self-supervised.svg" width="100%">

*Caption: Self-supervised learning creates supervision signals from the data itself, enabling learning from massive unlabeled datasets.*

---

## 📂 Overview

**Self-supervised learning** is a paradigm where the model learns from the data's own structure without human annotations. It has revolutionized NLP (BERT, GPT) and computer vision (SimCLR, MAE).

---

## 📐 Mathematical Foundations

### Core Principle

Given unlabeled data \(\mathcal{D} = \{x_i\}_{i=1}^N\), create pseudo-labels from the data itself:

$$\mathcal{L}_{\text{SSL}} = \mathbb{E}_{x \sim \mathcal{D}}[\ell(f_\theta(x), \text{pretext}(x))]$$

---

## 📊 Contrastive Learning

### InfoNCE Loss (SimCLR, MoCo)

For positive pair \((x_i, x_j)\) (two augmentations of same image):

$$\mathcal{L}_{i,j} = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]} \exp(\text{sim}(z_i, z_k)/\tau)}$$

where:

- \(z_i = g(f(x_i))\) is the projection of encoded representation

- \(\text{sim}(u, v) = \frac{u^\top v}{\|u\| \|v\|}\) is cosine similarity

- \(\tau\) is temperature parameter

### Theoretical Justification

**Theorem (InfoNCE Bound):** InfoNCE loss provides a lower bound on mutual information:

$$I(X; Y) \geq \log(N) - \mathcal{L}_{\text{NCE}}$$

**Proof Sketch:**
The optimal critic \(f^*(x, y) = \frac{p(y|x)}{p(y)} + c\). InfoNCE with this critic recovers mutual information. Finite samples give a lower bound. \(\blacksquare\)

---

## 📊 Masked Prediction

### Masked Language Modeling (BERT)

Mask ~15% of tokens and predict them:

$$\mathcal{L}_{\text{MLM}} = -\mathbb{E}_{x \sim \mathcal{D}} \left[\sum_{i \in \mathcal{M}} \log P(x_i | x_{\backslash \mathcal{M}}; \theta)\right]$$

where \(\mathcal{M}\) = set of masked positions.

### Masked Autoencoding (MAE)

Mask ~75% of image patches and reconstruct:

$$\mathcal{L}_{\text{MAE}} = \frac{1}{|\mathcal{M}|}\sum_{i \in \mathcal{M}} \|x_i - \hat{x}_i\|^2$$

---

## 📊 Autoregressive Prediction

### Next Token Prediction (GPT)

$$\mathcal{L}_{\text{AR}} = -\sum_{t=1}^{T} \log P(x_t | x_{1:t-1}; \theta)$$

**Connection to Information Theory:**

This is equivalent to minimizing the cross-entropy between the true distribution and model distribution:

$$H(P, Q) = -\mathbb{E}_{x \sim P}[\log Q(x)] = H(P) + D_{\text{KL}}(P \| Q)$$

---

## 💻 Code Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    """
    NT-Xent loss (Normalized Temperature-scaled Cross Entropy).
    
    L = -log(exp(sim(z_i, z_j)/τ) / Σ_k exp(sim(z_i, z_k)/τ))
    
    This is the InfoNCE loss used in SimCLR.
    """
    
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, z_i, z_j):
        """
        Args:
            z_i: Embeddings of first augmentation [batch_size, dim]
            z_j: Embeddings of second augmentation [batch_size, dim]
        """
        batch_size = z_i.size(0)
        
        # Normalize embeddings (for cosine similarity)
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        # Concatenate embeddings
        z = torch.cat([z_i, z_j], dim=0)  # [2N, dim]
        
        # Compute similarity matrix
        sim = torch.mm(z, z.t()) / self.temperature  # [2N, 2N]
        
        # Create labels for positive pairs
        # Positive pair for i is at position i + batch_size
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size),
            torch.arange(batch_size)
        ]).to(z.device)
        
        # Mask out self-similarity (diagonal)
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)
        sim.masked_fill_(mask, float('-inf'))
        
        # Cross-entropy loss
        loss = F.cross_entropy(sim, labels)
        return loss

class MaskedLanguageModel(nn.Module):
    """
    BERT-style masked language modeling.
    
    L_MLM = -E[Σ_{i ∈ masked} log P(x_i | x_\M)]
    """
    
    def __init__(self, vocab_size, hidden_dim, num_layers=6, num_heads=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output = nn.Linear(hidden_dim, vocab_size)
        self.mask_token_id = vocab_size - 1  # Special [MASK] token
        
    def forward(self, input_ids, mask_positions, labels):
        """
        Args:
            input_ids: Token IDs with some replaced by [MASK]
            mask_positions: Boolean mask of which positions are masked
            labels: Original token IDs at masked positions
        """
        x = self.embedding(input_ids)
        x = self.transformer(x)
        
        # Get logits only at masked positions
        masked_output = x[mask_positions]
        logits = self.output(masked_output)
        
        # Cross-entropy loss
        loss = F.cross_entropy(logits, labels)
        return loss

class AutoregressiveLM(nn.Module):
    """
    GPT-style autoregressive language model.
    
    L_AR = -Σ_t log P(x_t | x_1, ..., x_{t-1})
    """
    
    def __init__(self, vocab_size, hidden_dim, num_layers=6, num_heads=8, max_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(max_len, hidden_dim)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers)
        self.output = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        
        x = self.embedding(input_ids) + self.pos_embedding(positions)
        
        # Causal mask: can only attend to previous tokens
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device), diagonal=1
        ).bool()
        
        x = self.transformer(x, mask=causal_mask)
        logits = self.output(x)
        
        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        return loss

# Example usage
if __name__ == "__main__":
    # Contrastive learning
    batch_size, dim = 32, 128
    z_i = torch.randn(batch_size, dim)
    z_j = torch.randn(batch_size, dim)
    
    contrastive = ContrastiveLoss(temperature=0.5)
    loss = contrastive(z_i, z_j)
    print(f"Contrastive loss: {loss.item():.4f}")

```

---

## 📊 Method Comparison

| Method | Pretext Task | Architecture | Key Innovation |
|--------|--------------|--------------|----------------|
| **SimCLR** | Contrastive | ResNet + MLP | Strong augmentations |
| **MoCo** | Contrastive | Momentum encoder | Memory bank |
| **BERT** | Masked tokens | Transformer encoder | Bidirectional |
| **GPT** | Next token | Transformer decoder | Autoregressive |
| **MAE** | Masked patches | ViT | High masking ratio (75%) |
| **CLIP** | Image-text matching | Dual encoder | Multi-modal |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | SimCLR | [arXiv](https://arxiv.org/abs/2002.05709) |
| 📄 | BERT | [arXiv](https://arxiv.org/abs/1810.04805) |
| 📄 | MAE | [arXiv](https://arxiv.org/abs/2111.06377) |
| 📄 | InfoNCE Theory | [arXiv](https://arxiv.org/abs/1807.03748) |

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<!-- Navigation -->
<p align="center">
  <a href="../">⬅️ Back: Learning Frameworks</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../02_supervised/">Next: Supervised ➡️</a>
</p>

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=80&section=footer" width="100%"/>
</p>
