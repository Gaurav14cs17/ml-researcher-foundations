<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=Layer%20Normalization&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---

# Layer Normalization

> **The normalization for transformers**

---

## 📐 Algorithm

```
For each sample x:
    μ = (1/d) Σᵢ xᵢ           # Mean across features
    σ² = (1/d) Σᵢ (xᵢ - μ)²   # Variance across features
    x̂ᵢ = (xᵢ - μ) / √(σ² + ε)  # Normalize
    yᵢ = γx̂ᵢ + β               # Scale & shift (learnable)
```

---

## 📊 BatchNorm vs LayerNorm

| Aspect | BatchNorm | LayerNorm |
|--------|-----------|-----------|
| Normalizes over | Batch | Features |
| Batch size dependent | Yes | No |
| Same at train/test | No | Yes |
| Used in | CNNs | Transformers, RNNs |

---

## 🔑 Why for Transformers?

```
1. Sequence lengths vary (can't use batch statistics)
2. Works with any batch size (even 1)
3. Same behavior at train and test time
4. Works well with attention mechanism
```

---

## 💻 Code

```python
import torch.nn as nn

# Layer normalization
ln = nn.LayerNorm(d_model)  # Normalize over last dim

# In transformer
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.GELU(),
            nn.Linear(4*d_model, d_model)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # Pre-norm (modern style)
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        x = x + self.ff(self.ln2(x))
        return x
```

---

## 📊 Pre-Norm vs Post-Norm

```
Post-norm (original):  x = LN(x + Sublayer(x))
Pre-norm (modern):     x = x + Sublayer(LN(x))

Pre-norm: More stable, easier to train deep models
```

---

---

⬅️ [Back: Batch Norm](./batch-norm.md)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>
