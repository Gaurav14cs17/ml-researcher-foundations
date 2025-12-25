<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=Attention Mechanisms&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---

# 👁️ Attention Mechanisms

> **Learning what to focus on**

---

## 📐 Types of Attention

```
Scaled Dot-Product:
  Attention(Q, K, V) = softmax(QKᵀ/√d_k)V

Additive (Bahdanau):
  score(q, k) = vᵀ tanh(W_q q + W_k k)

Multi-Head:
  MultiHead = Concat(head₁,...,headₕ)W^O
```

---

## 💻 Code Example

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, V)
```

---

## 🔗 Applications

| Type | Used In |
|------|---------|
| **Self-Attention** | Transformers |
| **Cross-Attention** | Encoder-Decoder |
| **Flash Attention** | Efficient LLMs |

---

⬅️ [Back: Architectures](../)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>

