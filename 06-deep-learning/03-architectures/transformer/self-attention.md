# Self-Attention

> **The core mechanism of transformers**

---

## 📐 Formula

```
Attention(Q, K, V) = softmax(QKᵀ/√d_k) V

Where:
• Q = XW_Q  (queries)
• K = XW_K  (keys)
• V = XW_V  (values)
• d_k = key dimension
```

---

## 🎯 Intuition

```
For each position:
1. Q: "What am I looking for?"
2. K: "What do I contain?"
3. Similarity: QKᵀ (dot product)
4. Attention weights: softmax (normalize)
5. Output: Weighted sum of V
```

---

## 📊 Step by Step

```
Input: X ∈ ℝⁿˣᵈ (n tokens, d dimensions)

1. Project: Q, K, V = XW_Q, XW_K, XW_V
2. Scores: S = QKᵀ/√d_k ∈ ℝⁿˣⁿ
3. Weights: A = softmax(S) ∈ ℝⁿˣⁿ
4. Output: O = AV ∈ ℝⁿˣᵈ
```

---

## 💻 Code

```python
import torch
import torch.nn.functional as F
import math

def self_attention(x, W_q, W_k, W_v):
    """
    x: (batch, seq_len, d_model)
    """
    Q = x @ W_q
    K = x @ W_k
    V = x @ W_v
    
    d_k = K.shape[-1]
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    attn_weights = F.softmax(scores, dim=-1)
    output = attn_weights @ V
    
    return output, attn_weights
```

---

## ⚠️ Complexity

```
Time: O(n²d)  - quadratic in sequence length!
Space: O(n²)  - attention matrix

This is why long sequences are expensive.
Solutions: Flash Attention, Linear Attention
```

---

---

⬅️ [Back: Positional Encoding](./positional-encoding.md)
