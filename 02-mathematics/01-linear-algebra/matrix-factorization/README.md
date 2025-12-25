<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=Matrix Factorization&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---


# 📊 Matrix Factorization

> **Decomposing matrices for recommendations and embeddings**

---

## 🎯 Visual Overview

<img src="./images/matrix-factorization-recommender.svg" width="100%">

*Caption: Matrix factorization decomposes user-item matrix into user and item embeddings. Used in Netflix, Spotify recommendations.*

---

## 📐 Key Concept

```
User-Item Matrix R ≈ U × V^T

R: m×n rating matrix (users × items)
U: m×k user embeddings
V: n×k item embeddings

Objective:
min_{U,V} Σ_{observed (i,j)} (R_ij - U_i · V_j)² + λ(||U||² + ||V||²)
```

---

## 💻 Code Example

```python
import numpy as np
from sklearn.decomposition import NMF

# Ratings matrix (users x items)
R = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [0, 0, 5, 4]
])

# Non-negative matrix factorization
model = NMF(n_components=2, init='random', random_state=0)
U = model.fit_transform(R)
V = model.components_

# Reconstruct and predict missing ratings
R_pred = U @ V
print(f"Predicted ratings:\n{R_pred}")
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | Netflix Prize | [Paper](https://www.netflixprize.com/) |
| 🇨🇳 | 矩阵分解推荐 | [知乎](https://zhuanlan.zhihu.com/p/28577447) |

---

➡️ See [Decompositions](../decompositions/) for SVD

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>

