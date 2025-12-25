<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=00C853&height=120&section=header&text=Dimensionality%20Reduction&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-02-00C853?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/pca-tsne-comparison.svg" width="100%">

*Caption: PCA finds linear projections preserving variance. t-SNE preserves local structure for visualization.*

---

## 📐 Key Methods

### PCA (Principal Component Analysis)
```
Find directions of maximum variance:
X_centered = X - mean(X)
Σ = X^T X / n
Eigendecomposition: Σ = VΛV^T
Project: X_reduced = X @ V[:, :k]
```

### t-SNE
```
Preserve pairwise similarities:
1. Compute similarities in high-D: p_ij
2. Compute similarities in low-D: q_ij
3. Minimize KL(P || Q)

Good for visualization, not for reduction
```

---

## 💻 Code Examples

```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# PCA
pca = PCA(n_components=50)
X_pca = pca.fit_transform(X)
print(f"Variance explained: {pca.explained_variance_ratio_.sum():.2%}")

# t-SNE for visualization
tsne = TSNE(n_components=2, perplexity=30)
X_tsne = tsne.fit_transform(X)
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | t-SNE | [Paper](https://www.jmlr.org/papers/v9/vandermaaten08a.html) |
| 📄 | UMAP | [arXiv](https://arxiv.org/abs/1802.03426) |
| 🇨🇳 | 降维详解 | [知乎](https://zhuanlan.zhihu.com/p/32412043) |

---

⬅️ [Back: Linear Algebra](../) | ➡️ [Next: Decompositions](../decompositions/)

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=00C853&height=80&section=footer" width="100%"/>
</p>
