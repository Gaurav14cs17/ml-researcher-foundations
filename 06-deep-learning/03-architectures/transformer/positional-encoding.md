<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=120&section=header&text=Positional%20Encoding&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-06-45B7D1?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Why Needed?

```
Self-attention is permutation equivariant:
Attention({x₁, x₂, x₃}) = same regardless of order

But order matters in language!
"Dog bites man" ≠ "Man bites dog"

Solution: Add position information to embeddings
```

---

## 📐 Sinusoidal Encoding (Original)

```
PE(pos, 2i) = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

Where:
• pos = position in sequence
• i = dimension index
• d = model dimension
```

---

## 📊 Types

| Type | Method | Used In |
|------|--------|---------|
| Sinusoidal | Fixed sin/cos | Original Transformer |
| Learned | Trainable embeddings | BERT, GPT |
| Relative | Relative positions | Transformer-XL |
| RoPE | Rotary embeddings | LLaMA, GPT-NeoX |
| ALiBi | Linear bias | BLOOM |

---

## 💻 Code

```python
import torch
import math

def sinusoidal_encoding(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
    )
    
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    return pe

# Learned positional embeddings
pos_embedding = torch.nn.Embedding(max_len, d_model)
```

---

---

➡️ [Next: Self Attention](./self-attention.md)

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=80&section=footer" width="100%"/>
</p>
