<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=00C853&height=120&section=header&text=Distance%20Metrics&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-02-00C853?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/distance-metrics-complete.svg" width="100%">

*Caption: Distance metrics quantify how similar or different two data points are. Different metrics capture different notions of similarity.*

---

## 📐 Mathematical Foundations

### Lp Norms and Distances

```
Lp Distance:
d_p(x, y) = ||x - y||_p = (Σᵢ |xᵢ - yᵢ|^p)^(1/p)

Special Cases:
• L1 (Manhattan):  d₁ = Σᵢ |xᵢ - yᵢ|
• L2 (Euclidean):  d₂ = √(Σᵢ (xᵢ - yᵢ)²)
• L∞ (Chebyshev): d∞ = maxᵢ |xᵢ - yᵢ|
```

### Cosine Similarity

```
cos(θ) = (x · y) / (||x|| · ||y||)

Cosine Distance:
d_cos = 1 - cos(θ)

Properties:
• Range: [-1, 1] for similarity, [0, 2] for distance
• Scale invariant (only direction matters)
• Used for: Text embeddings, image similarity
```

### Mahalanobis Distance

```
d_M(x, y) = √((x - y)ᵀ Σ⁻¹ (x - y))

Where Σ = covariance matrix

Properties:
• Accounts for feature correlations
• Scale invariant
• Reduces to Euclidean if Σ = I
```

---

## 🎯 Distance Metrics Comparison

| Metric | Formula | Use Case | Properties |
|--------|---------|----------|------------|
| **Euclidean (L2)** | √Σ(xᵢ-yᵢ)² | General purpose | Sensitive to scale |
| **Manhattan (L1)** | Σ\|xᵢ-yᵢ\| | Sparse data, high-dim | Robust to outliers |
| **Cosine** | 1 - cos(θ) | Text, embeddings | Scale invariant |
| **Mahalanobis** | √((x-y)ᵀΣ⁻¹(x-y)) | Correlated features | Accounts for covariance |
| **Hamming** | Σ(xᵢ ≠ yᵢ) | Binary/categorical | Counts differences |
| **Jaccard** | 1 - \|A∩B\|/\|A∪B\| | Sets, binary vectors | Set similarity |

---

## 💻 Code Examples

```python
import numpy as np
from scipy.spatial.distance import cdist, cosine, mahalanobis
from sklearn.metrics.pairwise import cosine_similarity
import torch

# Sample vectors
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])

# Euclidean distance
d_euclidean = np.linalg.norm(x - y)
print(f"Euclidean: {d_euclidean:.4f}")

# Manhattan distance
d_manhattan = np.sum(np.abs(x - y))
print(f"Manhattan: {d_manhattan}")

# Cosine similarity/distance
cos_sim = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
cos_dist = 1 - cos_sim
print(f"Cosine similarity: {cos_sim:.4f}")

# Mahalanobis distance
X = np.random.randn(100, 3)
cov = np.cov(X.T)
d_mahal = mahalanobis(x, y, np.linalg.inv(cov))
print(f"Mahalanobis: {d_mahal:.4f}")

# Pairwise distances with scipy
X = np.random.randn(5, 3)
D_euclidean = cdist(X, X, metric='euclidean')
D_cosine = cdist(X, X, metric='cosine')

# PyTorch cosine similarity
x_torch = torch.tensor(x, dtype=torch.float32)
y_torch = torch.tensor(y, dtype=torch.float32)
cos_sim_torch = torch.nn.functional.cosine_similarity(
    x_torch.unsqueeze(0), y_torch.unsqueeze(0)
)

# Efficient pairwise L2 in PyTorch
def pairwise_l2(X, Y):
    """Compute pairwise L2 distances"""
    XX = (X ** 2).sum(dim=1, keepdim=True)
    YY = (Y ** 2).sum(dim=1, keepdim=True)
    XY = X @ Y.T
    return torch.sqrt(XX + YY.T - 2 * XY + 1e-8)
```

---

## 🌍 ML Applications

| Application | Distance Metric | Why |
|-------------|-----------------|-----|
| **k-NN** | Euclidean, Manhattan | Nearest neighbor search |
| **k-Means** | Euclidean | Cluster assignment |
| **Text Similarity** | Cosine | Direction matters, not magnitude |
| **Embedding Search** | Cosine, L2 | Semantic similarity |
| **Anomaly Detection** | Mahalanobis | Accounts for correlations |
| **Image Retrieval** | L2, Cosine | Visual similarity |

---

## 📊 When to Use What

```
Use Euclidean (L2) when:
• Features are continuous and comparable scale
• Actual geometric distance matters

Use Manhattan (L1) when:
• Features represent different things
• High-dimensional sparse data
• Grid-like paths matter

Use Cosine when:
• Magnitude doesn't matter (only direction)
• Text/document similarity
• Normalized embeddings

Use Mahalanobis when:
• Features are correlated
• Need scale-invariant distance
• Anomaly detection
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | Similarity Measures Survey | [Paper](https://www.sciencedirect.com/science/article/pii/S0306457309000259) |
| 🎥 | Distance Metrics Explained | [YouTube](https://www.youtube.com/watch?v=7pOSCaOoYNI) |
| 📖 | Scipy Distance Functions | [Docs](https://docs.scipy.org/doc/scipy/reference/spatial.distance.html) |
| 🇨🇳 | 距离度量详解 | [知乎](https://zhuanlan.zhihu.com/p/27305237) |
| 🇨🇳 | 相似度计算方法 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88782888) |

---

## 🔗 Where This Topic Is Used

| Application | How Distance Metrics Are Used |
|-------------|------------------------------|
| **Clustering (k-Means)** | Euclidean distance for centroid assignment |
| **Nearest Neighbors (k-NN)** | Find k closest training examples |
| **Vector Search (FAISS)** | Cosine/L2 for semantic retrieval |
| **Contrastive Learning** | Distance in embedding space |
| **Face Recognition** | Embedding distance for matching |
| **Recommendation Systems** | User/item similarity |

---

⬅️ [Back: 02-Calculus](../02-calculus/) | ➡️ [Next: 03-Optimization](../03-optimization/)

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=00C853&height=80&section=footer" width="100%"/>
</p>
