# 📊 Weight Clustering

> **Grouping weights into clusters**

<img src="./images/clustering-visual.svg" width="100%">

---

## 📐 Mathematical Foundations

### K-Means Clustering
```
Objective: min_C Σᵢ min_j ||wᵢ - cⱼ||²

Where:
• wᵢ = original weight
• cⱼ = cluster centroid (j ∈ {1,...,K})
• K = number of clusters (typically 16-256)
```

### Storage Savings
```
Original: n weights × 32 bits = 32n bits

Clustered:
• K centroids × 32 bits = 32K bits
• n indices × log₂(K) bits = n log₂(K) bits
• Total: 32K + n log₂(K) bits

For n=1M, K=16:
Original: 32 Mbits
Clustered: 512 + 4M = 4.0005 Mbits (8x compression)
```

### With Huffman Coding
```
More frequent clusters → shorter codes
Entropy: H = -Σ pⱼ log₂(pⱼ)

Bits per weight ≈ H (optimal)
Further ~2x compression possible
```

---

## 📂 Topics

| File | Topic | Key Concepts |
|------|-------|--------------|

---

## 🎯 The Core Idea

```
Original weights:
[1.23, 1.19, 1.25, 0.01, 0.03, -0.02, 0.89, 0.91, 0.87]

After K-means clustering (k=3):
Centroids: [1.22, 0.01, 0.89]
Indices:   [0, 0, 0, 1, 1, 1, 2, 2, 2]

Storage:
+-- 3 centroids (FP32): 12 bytes
+-- 9 indices (2 bits each): 3 bytes
+-- Total: 15 bytes vs 36 bytes (2.4x compression)
```

---

## 💻 Code Example

```python
import numpy as np
from sklearn.cluster import KMeans

def cluster_weights(weights, n_clusters=16):
    # Flatten and cluster
    flat = weights.flatten().reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(flat)
    
    # Replace weights with centroids
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_.flatten()
    clustered = centroids[labels].reshape(weights.shape)
    
    return clustered, centroids, labels
```

---

## 🔗 Where This Topic Is Used

| Topic | How Weight Clustering Is Used |
|-------|------------------------------|
| **Deep Compression** | Han et al. compression pipeline |
| **Trained Quantization** | Learn cluster centers |
| **Codebook Networks** | Extreme compression |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | Deep Compression | [arXiv](https://arxiv.org/abs/1510.00149) |
| 📄 | Trained Quantization | [arXiv](https://arxiv.org/abs/1712.05877) |
| 🇨🇳 | 权重聚类详解 | [知乎](https://zhuanlan.zhihu.com/p/628120082) |

---

⬅️ [Back: Pruning](../pruning/) | ➡️ [Next: Weight Sharing](../weight-sharing/)
