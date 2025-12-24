# Linear Transformations

> **The mathematics behind every neural network layer**

---

## 🎯 Visual Overview

<img src="./images/linear-transform.svg" width="100%">

*Caption: A linear transformation maps vectors while preserving vector addition and scalar multiplication. Matrix multiplication y = Ax transforms the standard basis vectors e₁, e₂ to the columns of A. Every dense layer, attention projection, and embedding lookup is a linear transformation.*

---

## 📂 Overview

Linear transformations are the foundation of neural networks. Understanding them helps you understand what layers actually do to your data.

---

## 🔑 Key Properties

| Property | Definition |
|----------|------------|
| **Linearity** | T(αu + βv) = αT(u) + βT(v) |
| **Matrix Form** | T(x) = Ax |
| **Preserves** | Lines, parallelism, origin |
| **Does Not Preserve** | Lengths, angles (unless orthogonal) |

---

## 📐 Common Transformations

| Transform | Matrix | Effect |
|-----------|--------|--------|
| **Scale** | diag(s₁, s₂) | Stretch/shrink |
| **Rotation** | [[cos, -sin], [sin, cos]] | Rotate |
| **Shear** | [[1, k], [0, 1]] | Slant |
| **Projection** | aaᵀ/\|\|a\|\|² | Project onto line |
| **Reflection** | I - 2aaᵀ | Mirror |

---

## 💻 Code

```python
import torch
import torch.nn as nn

# Dense layer is a linear transformation + bias (affine)
linear = nn.Linear(in_features=512, out_features=256)
y = linear(x)  # y = Wx + b

# Attention projections are linear
W_Q = nn.Linear(d_model, d_model)  # Query projection
W_K = nn.Linear(d_model, d_model)  # Key projection
W_V = nn.Linear(d_model, d_model)  # Value projection

Q = W_Q(x)  # Transform x to query space
K = W_K(x)  # Transform x to key space
V = W_V(x)  # Transform x to value space
```


## 🔗 Where This Topic Is Used

| Application | Usage |
|-------------|-------|
| **Machine Learning** | Core concept for ML systems |
| **Deep Learning** | Foundation for neural networks |
| **Research** | Important for understanding papers |


## 📚 References

| Type | Resource | Link |
|------|----------|------|
| 📖 | Textbook | See parent folder |
| 🎥 | Video Lectures | YouTube/Coursera |
| 🇨🇳 | 中文资源 | 知乎/B站 |

---

⬅️ [Back: Linear Algebra](../)

---

⬅️ [Back: Matrix Properties](../matrix-properties/) | ➡️ [Next: Vector Spaces](../vector-spaces/)
