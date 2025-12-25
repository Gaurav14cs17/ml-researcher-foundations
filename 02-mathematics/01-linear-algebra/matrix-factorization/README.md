<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=00C853&height=120&section=header&text=Matrix%20Factorization&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-02-00C853?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

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

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=00C853&height=80&section=footer" width="100%"/>
</p>
