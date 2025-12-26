<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=180&section=header&text=Dimensionality%20Reduction&fontSize=38&fontColor=fff&animation=twinkling&fontAlignY=32&desc=PCA%20·%20t-SNE%20·%20UMAP%20·%20Autoencoders&descAlignY=52&descSize=16" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/📚_Section-01.02_Dim_Reduction-00C853?style=for-the-badge" alt="Section"/>
  <img src="https://img.shields.io/badge/📊_Topics-PCA_tSNE_UMAP-blue?style=for-the-badge" alt="Topics"/>
  <img src="https://img.shields.io/badge/✍️_Author-Gaurav_Goswami-purple?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/📅_Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## ⚡ TL;DR

> **Dimensionality reduction compresses high-dimensional data while preserving important structure.** Essential for visualization, noise reduction, and computational efficiency.

- 📊 **PCA**: Linear, preserves variance, fast, interpretable
- 🌀 **t-SNE**: Non-linear, preserves local structure, for visualization
- 🗺️ **UMAP**: Non-linear, preserves global+local structure, faster than t-SNE
- 🧠 **Autoencoders**: Neural network-based, learns nonlinear manifolds

---

## 📑 Table of Contents

1. [Visual Overview](#-visual-overview)
2. [PCA: Complete Theory](#1-pca-principal-component-analysis)
3. [t-SNE](#2-t-sne)
4. [UMAP](#3-umap)
5. [Comparison](#4-comparison)
6. [Code Implementation](#5-code-implementation)
7. [Resources](#-resources)

---

## 🎨 Visual Overview

<img src="./images/pca-tsne-comparison.svg" width="100%">

```
┌─────────────────────────────────────────────────────────────────────────────┐
│               DIMENSIONALITY REDUCTION METHODS COMPARISON                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   HIGH-DIMENSIONAL DATA (e.g., 768D embeddings)                             │
│   ───────────────────────────────────────────────                           │
│                      │                                                       │
│        ┌─────────────┼─────────────┬─────────────┐                          │
│        ▼             ▼             ▼             ▼                          │
│   ┌────────┐   ┌────────┐   ┌────────┐   ┌────────────┐                    │
│   │  PCA   │   │ t-SNE  │   │  UMAP  │   │Autoencoder │                    │
│   └────────┘   └────────┘   └────────┘   └────────────┘                    │
│        │             │             │             │                          │
│   Linear         Non-linear   Non-linear   Non-linear                       │
│   Global         Local        Local+Global  Learned                         │
│   Fast           Slow         Medium        Slow (training)                 │
│   Variance       Similarity   Topology      Reconstruction                  │
│                                                                              │
│   USE CASES:                                                                │
│   • Preprocessing    • Visualization  • Visualization  • Feature learning  │
│   • Noise reduction  • Clustering viz • Clustering     • Generative models │
│   • Feature extract  • 2D/3D plots    • Large datasets • Compression       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. PCA: Principal Component Analysis

### 📌 Goal

Find orthogonal directions (principal components) that maximize variance in the data.

### 📐 Mathematical Formulation

Given centered data $X \in \mathbb{R}^{n \times d}$ (n samples, d features):

$$\text{Covariance matrix: } \Sigma = \frac{1}{n-1}X^TX$$

**Goal**: Find projection $W \in \mathbb{R}^{d \times k}$ that maximizes:
$$\text{Var}(XW) = W^T \Sigma W$$

subject to $W^TW = I$ (orthonormal columns)

### 🔍 Complete Derivation

```
Step 1: First Principal Component
        Find w₁ that maximizes variance of Xw₁:
        max_{w₁} w₁ᵀΣw₁  subject to ‖w₁‖ = 1

Step 2: Lagrangian
        L = w₁ᵀΣw₁ - λ(w₁ᵀw₁ - 1)

Step 3: Take derivative and set to zero
        ∂L/∂w₁ = 2Σw₁ - 2λw₁ = 0
        Σw₁ = λw₁

Step 4: This is an eigenvalue equation!
        w₁ must be an eigenvector of Σ

Step 5: Which eigenvector maximizes variance?
        Variance = w₁ᵀΣw₁ = w₁ᵀ(λw₁) = λw₁ᵀw₁ = λ
        
        Maximum variance when λ is LARGEST eigenvalue!
        w₁ = eigenvector of largest eigenvalue

Step 6: Second Principal Component
        Maximize w₂ᵀΣw₂ subject to ‖w₂‖ = 1 AND w₂ᵀw₁ = 0
        
        Solution: w₂ = eigenvector of second largest eigenvalue
        (Orthogonality follows from Spectral Theorem)

Step 7: General Solution
        Principal components = eigenvectors of Σ sorted by eigenvalue
        PC₁, PC₂, ..., PCₖ = top k eigenvectors
```

### 📐 PCA via SVD (Numerically Stable)

```
Instead of computing Σ = XᵀX and then eigendecomposition,
use SVD of X directly:

X = UΣVᵀ

Then:
  XᵀX = VΣ²Vᵀ  (eigendecomposition of covariance!)

  Principal components = columns of V
  Singular values² = eigenvalues of XᵀX

Projection:
  X_reduced = X·V[:,:k] = U[:,:k]·Σ[:k,:k]
```

### 💡 Examples

**Example 1**: 2D to 1D PCA
```
Data points: (1,2), (2,4), (3,6), (4,8)

Step 1: Center the data
  mean = (2.5, 5)
  centered = (-1.5,-3), (-0.5,-1), (0.5,1), (1.5,3)

Step 2: Covariance matrix
  Σ = [1.67  3.33]
      [3.33  6.67]

Step 3: Eigenvalues and eigenvectors
  λ₁ ≈ 8.33, v₁ ≈ [0.45, 0.89]
  λ₂ ≈ 0,    v₂ ≈ [-0.89, 0.45]

Step 4: First PC explains 8.33/(8.33+0) = 100% of variance

Step 5: Project onto first PC
  The data lies exactly on the line y = 2x!
```

**Example 2**: Explained Variance Ratio
```
Given eigenvalues: [4.0, 2.0, 1.0, 0.5, 0.3, 0.2]

Total variance: 8.0

Explained variance ratios:
  PC1: 4.0/8.0 = 50%
  PC2: 2.0/8.0 = 25%
  PC3: 1.0/8.0 = 12.5%
  ...

Cumulative:
  PC1: 50%
  PC1+PC2: 75%
  PC1+PC2+PC3: 87.5%

To capture 95% variance, need first 5 components.
```

### 💻 Code Implementation

```python
import numpy as np
from sklearn.decomposition import PCA

def pca_from_scratch(X, n_components):
    """
    PCA implementation from scratch.
    
    Steps:
    1. Center the data
    2. Compute covariance matrix
    3. Eigendecomposition
    4. Project onto top eigenvectors
    """
    # Center
    X_centered = X - X.mean(axis=0)
    
    # Covariance matrix
    n = X.shape[0]
    cov = X_centered.T @ X_centered / (n - 1)
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Sort by eigenvalue (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Select top k components
    components = eigenvectors[:, :n_components]
    
    # Project
    X_pca = X_centered @ components
    
    # Explained variance ratio
    explained_var = eigenvalues[:n_components] / eigenvalues.sum()
    
    return X_pca, components, explained_var

def pca_via_svd(X, n_components):
    """
    PCA via SVD (numerically more stable).
    """
    X_centered = X - X.mean(axis=0)
    
    # SVD
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    
    # Components = rows of Vt (or columns of V)
    components = Vt[:n_components].T
    
    # Projected data
    X_pca = U[:, :n_components] * S[:n_components]
    
    # Explained variance
    explained_var = (S[:n_components]**2) / (S**2).sum()
    
    return X_pca, components, explained_var

# Example usage
X = np.random.randn(1000, 100)
X_pca, components, explained_var = pca_from_scratch(X, n_components=10)
print(f"Shape: {X.shape} → {X_pca.shape}")
print(f"Explained variance: {explained_var.sum():.2%}")
```

---

## 2. t-SNE

### 📌 Goal

Preserve **local structure**: similar points in high-D should be similar in low-D.

### 📐 Algorithm

```
Step 1: Compute pairwise similarities in high-D
        pⱼ|ᵢ = exp(-‖xᵢ-xⱼ‖²/2σᵢ²) / Σₖ≠ᵢ exp(-‖xᵢ-xₖ‖²/2σᵢ²)
        pᵢⱼ = (pⱼ|ᵢ + pᵢ|ⱼ) / 2n  (symmetrized)

Step 2: Initialize low-D embedding Y randomly

Step 3: Compute similarities in low-D (Student-t with 1 df)
        qᵢⱼ = (1 + ‖yᵢ-yⱼ‖²)⁻¹ / Σₖ≠ₗ(1 + ‖yₖ-yₗ‖²)⁻¹

Step 4: Minimize KL divergence
        KL(P‖Q) = Σᵢⱼ pᵢⱼ log(pᵢⱼ/qᵢⱼ)

Step 5: Gradient descent on Y
        ∂C/∂yᵢ = 4Σⱼ(pᵢⱼ - qᵢⱼ)(yᵢ - yⱼ)(1 + ‖yᵢ-yⱼ‖²)⁻¹
```

### ⚠️ Key Hyperparameter: Perplexity

```
Perplexity ≈ effective number of neighbors

  Perplexity = 2^H(Pᵢ)  where H is entropy

  Typical values: 5-50
  Low perplexity → tight clusters, may miss global structure
  High perplexity → may merge distinct clusters
```

### 💻 Code

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def tsne_visualization(X, labels=None, perplexity=30):
    """
    t-SNE for visualization.
    """
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=1000,
        random_state=42
    )
    X_tsne = tsne.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    if labels is not None:
        scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='tab10', alpha=0.7)
        plt.colorbar(scatter)
    else:
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.7)
    plt.title(f't-SNE (perplexity={perplexity})')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    
    return X_tsne
```

---

## 3. UMAP

### 📌 Goal

Preserve **both local and global structure** using topological methods.

### 📐 Key Idea

```
UMAP models data as a fuzzy topological structure:

1. Build fuzzy simplicial complex from high-D data
   (weighted graph where edge weights = similarity)

2. Find low-D representation with similar topology

3. Minimize cross-entropy between high-D and low-D graphs
```

### Advantages over t-SNE

| Aspect | t-SNE | UMAP |
|--------|-------|------|
| Speed | O(n²) → O(n log n) with approximations | O(n^1.14) |
| Global structure | Poor | Better preserved |
| Scalability | Struggles > 10K points | Handles millions |
| Theory | Similarity preservation | Topological foundation |

### 💻 Code

```python
import umap

def umap_visualization(X, labels=None, n_neighbors=15, min_dist=0.1):
    """
    UMAP for visualization and clustering.
    """
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,  # Similar to perplexity
        min_dist=min_dist,        # Controls clustering tightness
        random_state=42
    )
    X_umap = reducer.fit_transform(X)
    
    return X_umap
```

---

## 4. Comparison

| Method | Type | Preserves | Speed | Use Case |
|--------|------|-----------|-------|----------|
| **PCA** | Linear | Global variance | Fast O(nd²) | Preprocessing, interpretable |
| **t-SNE** | Non-linear | Local | Slow O(n²) | Visualization |
| **UMAP** | Non-linear | Local + Global | Medium O(n^1.14) | Visualization, clustering |
| **Autoencoder** | Non-linear | Learned | Slow | Feature learning |

### When to Use What

```
Use PCA when:
  ✓ Need interpretability
  ✓ Linear relationships sufficient
  ✓ Preprocessing for other algorithms
  ✓ Very high-D data (d >> 1000)

Use t-SNE when:
  ✓ 2D/3D visualization only
  ✓ Small-medium datasets (<10K)
  ✓ Only care about local clusters

Use UMAP when:
  ✓ Visualization + downstream tasks
  ✓ Large datasets
  ✓ Want global structure preserved
  ✓ Need reproducibility
```

---

## 5. Code Implementation

### Complete Pipeline

```python
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt

class DimensionalityReduction:
    """Complete dimensionality reduction toolkit."""
    
    def __init__(self, X):
        self.X = X
        self.X_centered = X - X.mean(axis=0)
    
    def pca(self, n_components=50):
        """PCA reduction"""
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(self.X)
        
        return {
            'embedding': X_pca,
            'explained_variance': pca.explained_variance_ratio_,
            'components': pca.components_
        }
    
    def tsne(self, n_components=2, perplexity=30, use_pca=True):
        """t-SNE reduction (with optional PCA preprocessing)"""
        X = self.X
        if use_pca and X.shape[1] > 50:
            X = PCA(n_components=50).fit_transform(X)
        
        tsne = TSNE(n_components=n_components, perplexity=perplexity)
        X_tsne = tsne.fit_transform(X)
        
        return {'embedding': X_tsne, 'kl_divergence': tsne.kl_divergence_}
    
    def umap_reduce(self, n_components=2, n_neighbors=15, min_dist=0.1):
        """UMAP reduction"""
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist
        )
        X_umap = reducer.fit_transform(self.X)
        
        return {'embedding': X_umap, 'reducer': reducer}
    
    def plot_comparison(self, labels=None):
        """Compare all methods"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        methods = [
            ('PCA', self.pca(n_components=2)['embedding']),
            ('t-SNE', self.tsne()['embedding']),
            ('UMAP', self.umap_reduce()['embedding'])
        ]
        
        for ax, (name, embedding) in zip(axes, methods):
            if labels is not None:
                ax.scatter(embedding[:, 0], embedding[:, 1], 
                          c=labels, cmap='tab10', alpha=0.7)
            else:
                ax.scatter(embedding[:, 0], embedding[:, 1], alpha=0.7)
            ax.set_title(name)
            ax.set_xlabel('Dim 1')
            ax.set_ylabel('Dim 2')
        
        plt.tight_layout()
        return fig

# Usage
X = np.random.randn(1000, 100)
labels = np.random.randint(0, 5, 1000)

reducer = DimensionalityReduction(X)
fig = reducer.plot_comparison(labels)
```

---

## 📚 Resources

| Type | Resource | Description |
|------|----------|-------------|
| 📄 | [t-SNE Paper](https://www.jmlr.org/papers/v9/vandermaaten08a.html) | Original t-SNE |
| 📄 | [UMAP Paper](https://arxiv.org/abs/1802.03426) | Original UMAP |
| 🎥 | [StatQuest: PCA](https://www.youtube.com/watch?v=FgakZw6K1QQ) | Visual explanation |

---

## 🗺️ Navigation

| ⬅️ Previous | 🏠 Home | ➡️ Next |
|:-----------:|:-------:|:-------:|
| [Decompositions](../01_decompositions/README.md) | [Linear Algebra](../README.md) | [Eigen](../03_eigen/README.md) |

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer&animation=twinkling" width="100%"/>
</p>
