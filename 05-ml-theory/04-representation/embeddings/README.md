<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=120&section=header&text=Embeddings&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-05-4ECDC4?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/embeddings.svg" width="100%">

*Caption: Embeddings transform discrete objects (words, images, users) into dense vectors where similar items are close. Examples: Word2Vec, BERT embeddings, CLIP, collaborative filtering. Enable similarity search and transfer.*

---

## 📂 Overview

Embeddings are learned dense vector representations that capture semantic relationships. They enable similarity computation, transfer learning, and efficient retrieval.

---

## 📐 Mathematical Definitions

### Embedding Layer
```
E: V → ℝᵈ  (vocabulary to d-dimensional space)

E[i] = embedding vector for item i
Implemented as: E ∈ ℝ^{|V| × d} lookup table
```

### Word2Vec (Skip-gram)
```
P(context | word) = softmax(E_context · E_word)

Objective: maximize Σ log P(w_{t+j} | w_t)

Negative sampling approximation:
log σ(v_wO · v_wI) + Σ log σ(-v_wN · v_wI)
```

### Similarity Metrics
```
Cosine similarity:
sim(u, v) = u·v / (||u|| ||v||)

Euclidean distance:
d(u, v) = ||u - v||₂

Dot product:
s(u, v) = u · v
```

### Contrastive Learning (CLIP/SimCLR)
```
Pull together: positive pairs (same item, different views)
Push apart: negative pairs (different items)

InfoNCE Loss:
L = -log(exp(sim(z_i, z_j)/τ) / Σ_k exp(sim(z_i, z_k)/τ))
```

---

## 💻 Code Examples

```python
import torch
import torch.nn as nn

# Basic embedding layer
vocab_size = 10000
embed_dim = 256
embedding = nn.Embedding(vocab_size, embed_dim)

# Lookup
tokens = torch.tensor([1, 5, 100, 999])
vectors = embedding(tokens)  # [4, 256]

# Similarity search
def cosine_similarity(query, database):
    query_norm = query / query.norm(dim=-1, keepdim=True)
    db_norm = database / database.norm(dim=-1, keepdim=True)
    return query_norm @ db_norm.T

# Sentence embeddings (mean pooling)
def mean_pooling(token_embeddings, attention_mask):
    mask_expanded = attention_mask.unsqueeze(-1).float()
    sum_embeddings = (token_embeddings * mask_expanded).sum(dim=1)
    return sum_embeddings / mask_expanded.sum(dim=1)

# CLIP-style image-text similarity
class CLIP(nn.Module):
    def __init__(self, image_encoder, text_encoder, embed_dim):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1/0.07))
    
    def forward(self, images, texts):
        img_emb = self.image_encoder(images)
        txt_emb = self.text_encoder(texts)
        logits = self.logit_scale.exp() * img_emb @ txt_emb.T
        return logits
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | Word2Vec Paper | [arXiv](https://arxiv.org/abs/1301.3781) |
| 📄 | CLIP Paper | [arXiv](https://arxiv.org/abs/2103.00020) |
| 📄 | Sentence-BERT | [arXiv](https://arxiv.org/abs/1908.10084) |
| 🇨🇳 | 词向量详解 | [知乎](https://zhuanlan.zhihu.com/p/26306795) |
| 🇨🇳 | CLIP原理 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88619088) |
| 🇨🇳 | Embedding技术 | [B站](https://www.bilibili.com/video/BV1J94y1f7u5) |

---

<- [Back](../)

---

## 🔗 Where This Topic Is Used

| Application | Usage |
|-------------|-------|
| **Machine Learning** | Core concept for ML systems |
| **Deep Learning** | Foundation for neural networks |
| **Research** | Important for understanding papers |

---

---

⬅️ [Back: embeddings](../)

---

➡️ [Next: Feature Learning](../feature-learning/)

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=80&section=footer" width="100%"/>
</p>
