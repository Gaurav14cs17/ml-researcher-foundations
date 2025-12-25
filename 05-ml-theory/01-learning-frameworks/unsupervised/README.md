<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=120&section=header&text=Unsupervised%20Learning&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-05-4ECDC4?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/unsupervised.svg" width="100%">

*Caption: Unsupervised learning discovers hidden patterns in data without labels. Tasks include clustering (group similar items), dimensionality reduction (compress to key features), and density estimation (model P(x)).*

---

## 📂 Overview

Unsupervised learning finds structure in data without supervision. It's essential for discovering patterns, reducing dimensionality, and learning representations from abundant unlabeled data.

---

## 📐 Mathematical Foundations

### Clustering (K-Means)
```
Objective: min_μ Σᵢ Σⱼ ||xⱼ - μᵢ||² · 𝟙[xⱼ ∈ cluster i]

Algorithm:
1. Assign: zᵢ = argmin_k ||xᵢ - μₖ||²
2. Update: μₖ = (1/|Cₖ|) Σ_{i∈Cₖ} xᵢ
```

### PCA (Dimensionality Reduction)
```
Maximize variance: max_w wᵀΣw s.t. ||w||=1

Solution: w₁ = eigenvector of largest eigenvalue of Σ

Projection: X_reduced = X · W_k
Where W_k = [w₁|...|wₖ] (top k eigenvectors)
```

### VAE (Variational Autoencoder)
```
ELBO: L = E_q[log p(x|z)] - KL(q(z|x)||p(z))
      = Reconstruction - Regularization

Reparameterization trick:
z = μ + σ ⊙ ε, where ε ~ N(0,I)
```

### Density Estimation
```
Gaussian Mixture Model:
p(x) = Σₖ πₖ N(x; μₖ, Σₖ)

Fit via Expectation-Maximization (EM)
```

---

## 🔑 Key Topics

| Topic | Description |
|-------|-------------|
| Fundamentals | Core concepts |
| Applications | ML use cases |
| Implementation | Code examples |

---

## 💻 Example

```python
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Clustering
kmeans = KMeans(n_clusters=3)
labels = kmeans.fit_predict(X)

# Dimensionality Reduction
pca = PCA(n_components=50)
X_reduced = pca.fit_transform(X)
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | Bishop PRML Ch. 9 | [Book](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) |
| 📄 | VAE Paper | [arXiv](https://arxiv.org/abs/1312.6114) |
| 📖 | scikit-learn Clustering | [Docs](https://scikit-learn.org/stable/modules/clustering.html) |
| 🇨🇳 | 无监督学习详解 | [知乎](https://zhuanlan.zhihu.com/p/25110450) |
| 🇨🇳 | 聚类与降维 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88619088) |
| 🇨🇳 | PCA原理 | [B站](https://www.bilibili.com/video/BV1ys411472E) |

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

⬅️ [Back: unsupervised](../)

---

⬅️ [Back: Supervised](../supervised/)

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=80&section=footer" width="100%"/>
</p>
