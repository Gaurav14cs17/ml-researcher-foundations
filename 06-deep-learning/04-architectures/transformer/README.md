<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=120&section=header&text=Transformer%20Architecture&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-06-45B7D1?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 📐 Core Components

```
Self-Attention:
  Attention(Q, K, V) = softmax(QKᵀ/√d_k)V

Multi-Head Attention:
  MultiHead(Q,K,V) = Concat(head₁,...,headₕ)Wᴼ
  where headᵢ = Attention(QWᵢᴽ, KWᵢᴷ, VWᵢⱽ)

Transformer Block:
  x → LayerNorm → MultiHeadAttn → + → LayerNorm → FFN → +
       ↑_________________________↓     ↑______________↓
```

---

## 💻 Code Example

```python
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        x = x + self.ffn(self.ln2(x))
        return x
```

---

## 🔗 Applications

| Model | Type | Application |
|-------|------|-------------|
| **GPT** | Decoder-only | Text generation |
| **BERT** | Encoder-only | Understanding |
| **T5** | Encoder-Decoder | Seq2Seq |
| **ViT** | Encoder | Vision |

---

⬅️ [Back: Architectures](../)

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=80&section=footer" width="100%"/>
</p>
