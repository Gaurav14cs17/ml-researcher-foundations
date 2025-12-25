<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=07 Clustering&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---


# 🎯 Clustering

> **Grouping similar data points together**

---

## 🎯 Visual Overview

<img src="./images/clustering-algorithms-complete.svg" width="100%">

*Caption: Clustering algorithms group similar data points. k-Means finds spherical clusters, DBSCAN finds arbitrary shapes, Hierarchical builds a tree of clusters.*

---

## 📐 Mathematical Foundations

### k-Means

```
Objective: Minimize within-cluster sum of squares

J = Σₖ Σᵢ∈Cₖ ||xᵢ - μₖ||²

Algorithm (Lloyd's):
1. Initialize k centroids μ₁, ..., μₖ
2. Assign each point to nearest centroid:
   Cₖ = {xᵢ : ||xᵢ - μₖ|| ≤ ||xᵢ - μⱼ|| ∀j}
3. Update centroids:
   μₖ = (1/|Cₖ|) Σᵢ∈Cₖ xᵢ
4. Repeat until convergence

Complexity: O(nkdT) where T = iterations
```

### DBSCAN

```
Parameters:
• ε: neighborhood radius
• minPts: minimum points for core point

Definitions:
• Core point: Has ≥ minPts in ε-neighborhood
• Border point: In ε-neighborhood of core point
• Noise: Neither core nor border

Algorithm:
1. Find all core points
2. Connect core points within ε distance
3. Assign border points to nearby clusters
4. Mark remaining as noise
```

### Hierarchical Clustering

```
Agglomerative (bottom-up):
1. Start with each point as cluster
2. Merge closest clusters
3. Repeat until one cluster remains

Linkage methods:
• Single: min distance between clusters
• Complete: max distance
• Average: mean distance
• Ward: minimize variance increase

Creates dendrogram (tree structure)
```

---

## 🎯 Algorithm Comparison

| Algorithm | Cluster Shape | Scalability | Needs k? |
|-----------|---------------|-------------|----------|
| **k-Means** | Spherical | O(nkd) ✅ | Yes |
| **DBSCAN** | Arbitrary | O(n²) or O(n log n) | No |
| **Hierarchical** | Any | O(n²) ❌ | No (cut tree) |
| **GMM** | Elliptical | O(nkd²) | Yes |
| **Spectral** | Any | O(n³) ❌ | Yes |

---

## 💻 Code Examples

```python
import numpy as np
from sklearn.cluster import (
    KMeans, DBSCAN, AgglomerativeClustering, 
    SpectralClustering
)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.datasets import make_blobs

# Create dataset
X, y_true = make_blobs(n_samples=300, centers=4, random_state=42)

# k-Means
kmeans = KMeans(n_clusters=4, random_state=42)
y_kmeans = kmeans.fit_predict(X)
print(f"k-Means silhouette: {silhouette_score(X, y_kmeans):.4f}")

# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
y_dbscan = dbscan.fit_predict(X)
n_clusters = len(set(y_dbscan)) - (1 if -1 in y_dbscan else 0)
print(f"DBSCAN found {n_clusters} clusters")

# Hierarchical
agg = AgglomerativeClustering(n_clusters=4, linkage='ward')
y_agg = agg.fit_predict(X)
print(f"Hierarchical ARI: {adjusted_rand_score(y_true, y_agg):.4f}")

# Gaussian Mixture Model
gmm = GaussianMixture(n_components=4, random_state=42)
y_gmm = gmm.fit_predict(X)
print(f"GMM log-likelihood: {gmm.score(X):.4f}")

# Elbow method for choosing k
inertias = []
K_range = range(1, 10)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X)
    inertias.append(km.inertia_)

# Silhouette analysis
silhouettes = []
for k in range(2, 10):
    km = KMeans(n_clusters=k, random_state=42)
    y_km = km.fit_predict(X)
    silhouettes.append(silhouette_score(X, y_km))

print(f"Best k by silhouette: {np.argmax(silhouettes) + 2}")
```

---

## 🌍 ML Applications

| Application | Algorithm | Why |
|-------------|-----------|-----|
| **Customer Segmentation** | k-Means, GMM | Business segments |
| **Anomaly Detection** | DBSCAN, Isolation Forest | Find outliers |
| **Image Segmentation** | k-Means, Spectral | Pixel grouping |
| **Document Clustering** | k-Means on embeddings | Topic grouping |
| **Gene Expression** | Hierarchical | Biological relationships |

---

## 📊 Evaluation Metrics

| Metric | Needs Labels? | Formula |
|--------|---------------|---------|
| **Silhouette** | No | (b-a)/max(a,b) |
| **Inertia** | No | Within-cluster SS |
| **ARI** | Yes | Adjusted Rand Index |
| **NMI** | Yes | Normalized Mutual Info |
| **Calinski-Harabasz** | No | Ratio of dispersions |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | k-Means++ | [Paper](https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf) |
| 📄 | DBSCAN | [Paper](https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf) |
| 🎥 | Clustering Explained | [YouTube](https://www.youtube.com/watch?v=5I3Ei69I40s) |
| 🇨🇳 | 聚类算法详解 | [知乎](https://zhuanlan.zhihu.com/p/27689464) |
| 🇨🇳 | k-Means原理 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88444444) |

---

## 🔗 Where This Topic Is Used

| Application | How Clustering Is Used |
|-------------|------------------------|
| **Unsupervised Learning** | Core technique |
| **Data Exploration** | Find natural groups |
| **Feature Engineering** | Cluster as feature |
| **Preprocessing** | Data stratification |
| **Semi-supervised** | Pseudo-labeling |

---

⬅️ [Back: 06-Ensemble Methods](../06-ensemble-methods/) | ➡️ [Next: 08-Model Selection](../08-model-selection/)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>

