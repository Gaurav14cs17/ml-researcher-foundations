<!-- Navigation -->
<p align="center">
  <a href="../">⬅️ Back: Representation</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../02_feature_learning/">Next: Feature Learning ➡️</a>
</p>

---

<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=120&section=header&text=Embeddings&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-05-4ECDC4?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/embeddings.svg" width="100%">

*Caption: Embeddings map discrete objects to continuous vector spaces where similarity is meaningful.*

---

## 📂 Overview

**Embeddings** are learned dense vector representations that capture semantic relationships. They transform discrete objects (words, images, users) into continuous vectors where similar items are close together.

---

## 📐 Mathematical Foundation

### Embedding Layer

An embedding is a mapping:
```math
E: \mathcal{V} \to \mathbb{R}^d
```

where \(\mathcal{V}\) is a discrete vocabulary and \(d\) is the embedding dimension.

Implemented as a lookup table \(E \in \mathbb{R}^{|\mathcal{V}| \times d}\):
```math
e_i = E[i] = E^\top \mathbf{1}_i
```

where \(\mathbf{1}_i\) is a one-hot vector.

---

## 📐 Word2Vec

### Skip-gram Model

Given word \(w_t\), predict context words \(w_{t+j}\):

```math
P(w_{t+j} | w_t) = \frac{\exp(v_{w_{t+j}}^\top v_{w_t})}{\sum_{w \in \mathcal{V}} \exp(v_w^\top v_{w_t})}
```

**Objective:** Maximize log-likelihood:
```math
\mathcal{L} = \sum_{t=1}^T \sum_{-c \leq j \leq c, j \neq 0} \log P(w_{t+j} | w_t)
```

### Negative Sampling

Approximate softmax with negative sampling:

```math
\log \sigma(v_{w_O}^\top v_{w_I}) + \sum_{i=1}^k \mathbb{E}_{w_i \sim P_n(w)}[\log \sigma(-v_{w_i}^\top v_{w_I})]
```

where \(P_n(w) \propto \text{freq}(w)^{3/4}\).

---

## 📐 Contrastive Learning

### InfoNCE Loss

For positive pair \((x, x^+)\) and negative samples \(\{x_i^-\}_{i=1}^{K}\):

```math
\mathcal{L}_{\text{NCE}} = -\log \frac{\exp(\text{sim}(f(x), f(x^+))/\tau)}{\exp(\text{sim}(f(x), f(x^+))/\tau) + \sum_{i=1}^K \exp(\text{sim}(f(x), f(x_i^-))/\tau)}
```

**Theorem:** InfoNCE optimizes a lower bound on mutual information:
```math
I(X; X^+) \geq \log K - \mathcal{L}_{\text{NCE}}
```

---

## 📐 Similarity Metrics

### Cosine Similarity

```math
\text{sim}(u, v) = \frac{u \cdot v}{\|u\| \|v\|} = \cos \theta
```

**Property:** Scale-invariant, range \([-1, 1]\).

### Euclidean Distance

```math
d(u, v) = \|u - v\|_2 = \sqrt{\sum_i (u_i - v_i)^2}
```

### Dot Product

```math
s(u, v) = u \cdot v = \sum_i u_i v_i
```

**Connection:** For normalized vectors: \(\text{cosine}(u,v) = u \cdot v\).

---

## 💻 Code Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EmbeddingLayer(nn.Module):
    """
    Standard embedding layer.
    E: V → ℝ^d
    """
    
    def __init__(self, vocab_size, embed_dim, padding_idx=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        
        # Initialize with uniform distribution
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
    
    def forward(self, indices):
        return self.embedding(indices)

class Word2VecSkipGram(nn.Module):
    """
    Skip-gram Word2Vec with negative sampling.
    
    P(context | word) ∝ exp(v_context · v_word)
    """
    
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embed_dim)
        self.out_embed = nn.Embedding(vocab_size, embed_dim)
        
        # Initialize
        nn.init.uniform_(self.in_embed.weight, -0.5/embed_dim, 0.5/embed_dim)
        nn.init.zeros_(self.out_embed.weight)
    
    def forward(self, center_word, context_word, neg_words):
        """
        center_word: [batch_size]
        context_word: [batch_size]
        neg_words: [batch_size, num_neg]
        """
        # Embeddings
        center_emb = self.in_embed(center_word)       # [B, D]
        context_emb = self.out_embed(context_word)    # [B, D]
        neg_emb = self.out_embed(neg_words)           # [B, K, D]
        
        # Positive score
        pos_score = torch.sum(center_emb * context_emb, dim=1)  # [B]
        pos_loss = F.logsigmoid(pos_score)
        
        # Negative score
        neg_score = torch.bmm(neg_emb, center_emb.unsqueeze(2)).squeeze()  # [B, K]
        neg_loss = F.logsigmoid(-neg_score).sum(dim=1)
        
        return -(pos_loss + neg_loss).mean()
    
    def get_embeddings(self):
        return self.in_embed.weight.data

class ContrastiveLearning(nn.Module):
    """
    InfoNCE contrastive learning.
    
    L = -log(exp(sim(z,z⁺)/τ) / Σexp(sim(z,zᵢ)/τ))
    """
    
    def __init__(self, encoder, projection_dim=128, temperature=0.07):
        super().__init__()
        self.encoder = encoder
        self.projector = nn.Sequential(
            nn.Linear(encoder.output_dim, 256),
            nn.ReLU(),
            nn.Linear(256, projection_dim)
        )
        self.temperature = temperature
    
    def forward(self, x1, x2):
        """x1, x2 are two augmentations of the same batch."""
        # Encode and project
        z1 = F.normalize(self.projector(self.encoder(x1)), dim=1)
        z2 = F.normalize(self.projector(self.encoder(x2)), dim=1)
        
        batch_size = z1.size(0)
        
        # Similarity matrix
        z = torch.cat([z1, z2], dim=0)  # [2B, D]
        sim = z @ z.T / self.temperature  # [2B, 2B]
        
        # Mask out self-similarity
        mask = torch.eye(2 * batch_size, device=z.device).bool()
        sim.masked_fill_(mask, float('-inf'))
        
        # Positive pairs
        labels = torch.cat([
            torch.arange(batch_size, 2*batch_size),
            torch.arange(batch_size)
        ]).to(z.device)
        
        loss = F.cross_entropy(sim, labels)
        return loss

def cosine_similarity_search(query, database, top_k=5):
    """
    Find most similar items in database.
    
    sim(u, v) = u·v / (||u|| ||v||)
    """
    # Normalize
    query_norm = F.normalize(query, dim=-1)
    db_norm = F.normalize(database, dim=-1)
    
    # Compute similarities
    similarities = query_norm @ db_norm.T
    
    # Get top-k
    top_values, top_indices = torch.topk(similarities, k=top_k, dim=-1)
    
    return top_indices, top_values

# Example: Sentence embeddings via mean pooling
def mean_pooling(token_embeddings, attention_mask):
    """
    Mean pooling over tokens (ignoring padding).
    
    sentence_emb = Σ(token_emb * mask) / Σ(mask)
    """
    mask_expanded = attention_mask.unsqueeze(-1).float()
    sum_embeddings = (token_embeddings * mask_expanded).sum(dim=1)
    sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
    return sum_embeddings / sum_mask
```

---

## 📊 Embedding Types

| Type | Input | Output Dim | Use Case |
|------|-------|------------|----------|
| Word2Vec | Words | 100-300 | Word similarity |
| BERT | Tokens | 768 | NLU tasks |
| CLIP | Image/Text | 512-1024 | Multi-modal |
| Node2Vec | Graphs | 64-256 | Graph ML |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | Word2Vec | [Mikolov et al.](https://arxiv.org/abs/1301.3781) |
| 📄 | CLIP | [Radford et al.](https://arxiv.org/abs/2103.00020) |
| 📄 | SimCLR | [Chen et al.](https://arxiv.org/abs/2002.05709) |

---

⬅️ [Back: Representation](../) | ➡️ [Next: Feature Learning](../02_feature_learning/)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<!-- Navigation -->
<p align="center">
  <a href="../">⬅️ Back: Representation</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../02_feature_learning/">Next: Feature Learning ➡️</a>
</p>

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=80&section=footer" width="100%"/>
</p>
