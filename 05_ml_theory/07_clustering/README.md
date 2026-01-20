<!-- Navigation -->
<p align="center">
  <a href="../06_ensemble_methods/">⬅️ Prev: Ensemble Methods</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../">📚 ML Theory</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../08_model_selection/">Next: Model Selection ➡️</a>
</p>

---

<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=120&section=header&text=Clustering&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-05-4ECDC4?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/clustering-algorithms-complete.svg" width="100%">

*Caption: Clustering algorithms group similar data points without supervision.*

---

## 📂 Overview

**Clustering** partitions data into groups where points in the same cluster are more similar to each other than to points in other clusters.

---

## 📐 K-Means

### Objective Function

$$
J = \sum_{i=1}^n \sum_{k=1}^K r_{ik} \|x_i - \mu_k\|^2
$$

where $r_{ik} \in \{0,1\}$ indicates if $x_i$ belongs to cluster $k$.

### Lloyd's Algorithm

**E-step (Assignment):**

$$
r_{ik} = \begin{cases} 1 & \text{if } k = \arg\min_j \|x_i - \mu_j\|^2 \\ 0 & \text{otherwise} \end{cases}
$$

**M-step (Update):**

$$
\mu_k = \frac{\sum_i r_{ik} x_i}{\sum_i r_{ik}}
$$

**Theorem:** K-means converges to a local minimum.

**Proof:** Each step decreases or maintains $J$. Assignment step: each point moves to closest centroid, reducing its contribution. Update step: mean minimizes sum of squared distances. Since $J \geq 0$ and decreases, it converges. $\blacksquare$

### K-Means++ Initialization

$$
P(x_i \text{ as next center}) \propto D(x_i)^2
$$

where $D(x_i)$ = distance to nearest existing center.

**Theorem:** K-means++ achieves $O(\log k)$-approximation in expectation.

---

## 📐 Gaussian Mixture Models

### Model

$$
p(x) = \sum_{k=1}^K \pi_k \mathcal{N}(x | \mu_k, \Sigma_k)
$$

### EM Algorithm

**E-step:** Compute responsibilities:

$$
\gamma_{ik} = \frac{\pi_k \mathcal{N}(x_i | \mu_k, \Sigma_k)}{\sum_j \pi_j \mathcal{N}(x_i | \mu_j, \Sigma_j)}
$$

**M-step:** Update parameters:

$$
N_k = \sum_i \gamma_{ik}
\mu_k = \frac{1}{N_k}\sum_i \gamma_{ik} x_i
\Sigma_k = \frac{1}{N_k}\sum_i \gamma_{ik}(x_i - \mu_k)(x_i - \mu_k)^\top
\pi_k = \frac{N_k}{n}
$$

**Theorem:** EM monotonically increases log-likelihood.

---

## 📐 Spectral Clustering

### Algorithm

1. Construct similarity graph with adjacency $W$
2. Compute normalized Laplacian $L = I - D^{-1/2}WD^{-1/2}$
3. Find $k$ smallest eigenvectors of $L$
4. Run k-means on eigenvector embeddings

### Graph Laplacian

**Unnormalized:** $L = D - W$

**Normalized:** $L_{\text{sym}} = D^{-1/2}LD^{-1/2}$

**Property:** Number of zero eigenvalues = number of connected components.

---

## 📐 DBSCAN

### Definitions

- **Core point:** Has $\geq$ minPts in $\epsilon$-neighborhood
- **Border point:** In $\epsilon$-neighborhood of core point
- **Noise:** Neither core nor border

### Algorithm

1. Find all core points
2. Connect core points within $\epsilon$ distance
3. Assign border points to clusters
4. Mark remaining as noise

**Advantage:** Finds arbitrary-shaped clusters, identifies outliers.

---

## 💻 Code Implementation

```python
import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import eigh

class KMeans:
    """
    K-Means clustering with k-means++ initialization.
    
    min_μ Σᵢ Σₖ rᵢₖ ||xᵢ - μₖ||²
    """
    
    def __init__(self, n_clusters, max_iters=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
    
    def _kmeans_plus_plus(self, X):
        """K-means++ initialization."""
        n = len(X)
        centers = [X[np.random.randint(n)]]
        
        for _ in range(1, self.n_clusters):

            # Distance to nearest center
            distances = np.min(cdist(X, np.array(centers)), axis=1)
            
            # Sample proportional to D²
            probs = distances ** 2
            probs /= probs.sum()
            
            idx = np.random.choice(n, p=probs)
            centers.append(X[idx])
        
        return np.array(centers)
    
    def fit(self, X):
        n = len(X)
        
        # Initialize with k-means++
        self.centroids = self._kmeans_plus_plus(X)
        
        for iteration in range(self.max_iters):
            old_centroids = self.centroids.copy()
            
            # E-step: Assign to nearest centroid
            distances = cdist(X, self.centroids)
            self.labels = np.argmin(distances, axis=1)
            
            # M-step: Update centroids
            for k in range(self.n_clusters):
                mask = self.labels == k
                if np.sum(mask) > 0:
                    self.centroids[k] = X[mask].mean(axis=0)
            
            # Check convergence
            if np.linalg.norm(self.centroids - old_centroids) < self.tol:
                break
        
        # Compute inertia
        self.inertia = np.sum([
            np.sum((X[self.labels == k] - self.centroids[k]) ** 2)
            for k in range(self.n_clusters)
        ])
        
        return self
    
    def predict(self, X):
        distances = cdist(X, self.centroids)
        return np.argmin(distances, axis=1)

class GaussianMixture:
    """
    Gaussian Mixture Model via EM algorithm.
    
    p(x) = Σₖ πₖ N(x | μₖ, Σₖ)
    """
    
    def __init__(self, n_components, max_iters=100, tol=1e-4):
        self.n_components = n_components
        self.max_iters = max_iters
        self.tol = tol
    
    def _gaussian_pdf(self, X, mean, cov):
        """Multivariate Gaussian PDF."""
        d = len(mean)
        diff = X - mean
        
        # Add regularization for numerical stability
        cov_reg = cov + 1e-6 * np.eye(d)
        
        inv_cov = np.linalg.inv(cov_reg)
        det = np.linalg.det(cov_reg)
        
        exponent = -0.5 * np.sum(diff @ inv_cov * diff, axis=1)
        normalization = 1 / np.sqrt((2 * np.pi) ** d * det)
        
        return normalization * np.exp(exponent)
    
    def fit(self, X):
        n, d = X.shape
        K = self.n_components
        
        # Initialize with k-means
        kmeans = KMeans(K)
        kmeans.fit(X)
        
        self.means = kmeans.centroids.copy()
        self.covariances = [np.eye(d) for _ in range(K)]
        self.weights = np.ones(K) / K
        
        prev_log_likelihood = -np.inf
        
        for iteration in range(self.max_iters):

            # E-step: Compute responsibilities
            resp = np.zeros((n, K))
            for k in range(K):
                resp[:, k] = self.weights[k] * self._gaussian_pdf(
                    X, self.means[k], self.covariances[k]
                )
            resp /= resp.sum(axis=1, keepdims=True) + 1e-10
            
            # M-step: Update parameters
            Nk = resp.sum(axis=0)
            
            for k in range(K):
                self.weights[k] = Nk[k] / n
                self.means[k] = (resp[:, k:k+1].T @ X).flatten() / Nk[k]
                
                diff = X - self.means[k]
                self.covariances[k] = (resp[:, k:k+1] * diff).T @ diff / Nk[k]
            
            # Compute log-likelihood
            log_likelihood = np.sum(np.log(resp.sum(axis=1) + 1e-10))
            
            # Check convergence
            if abs(log_likelihood - prev_log_likelihood) < self.tol:
                break
            prev_log_likelihood = log_likelihood
        
        self.labels = np.argmax(resp, axis=1)
        return self
    
    def predict_proba(self, X):
        n = len(X)
        K = self.n_components
        
        resp = np.zeros((n, K))
        for k in range(K):
            resp[:, k] = self.weights[k] * self._gaussian_pdf(
                X, self.means[k], self.covariances[k]
            )
        return resp / resp.sum(axis=1, keepdims=True)
    
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

class SpectralClustering:
    """
    Spectral Clustering using graph Laplacian.
    """
    
    def __init__(self, n_clusters, gamma=1.0):
        self.n_clusters = n_clusters
        self.gamma = gamma
    
    def fit(self, X):
        n = len(X)
        
        # Construct similarity matrix (RBF kernel)
        distances = cdist(X, X, 'sqeuclidean')
        W = np.exp(-self.gamma * distances)
        np.fill_diagonal(W, 0)
        
        # Compute normalized Laplacian
        D = np.diag(W.sum(axis=1))
        D_inv_sqrt = np.diag(1 / np.sqrt(W.sum(axis=1) + 1e-10))
        L = np.eye(n) - D_inv_sqrt @ W @ D_inv_sqrt
        
        # Find k smallest eigenvectors
        eigenvalues, eigenvectors = eigh(L)
        U = eigenvectors[:, :self.n_clusters]
        
        # Normalize rows
        U = U / np.linalg.norm(U, axis=1, keepdims=True)
        
        # Run k-means on eigenvectors
        kmeans = KMeans(self.n_clusters)
        kmeans.fit(U)
        self.labels = kmeans.labels
        
        return self

# Clustering evaluation metrics
def silhouette_score(X, labels):
    """
    Silhouette score: (b - a) / max(a, b)
    where a = intra-cluster distance, b = nearest-cluster distance
    """
    n = len(X)
    unique_labels = np.unique(labels)
    
    scores = []
    for i in range(n):

        # Intra-cluster distance
        same_cluster = labels == labels[i]
        if np.sum(same_cluster) <= 1:
            a = 0
        else:
            a = np.mean(np.linalg.norm(X[same_cluster] - X[i], axis=1))
        
        # Nearest cluster distance
        b = np.inf
        for label in unique_labels:
            if label != labels[i]:
                other_cluster = labels == label
                if np.sum(other_cluster) > 0:
                    b = min(b, np.mean(np.linalg.norm(X[other_cluster] - X[i], axis=1)))
        
        if a == 0 and b == np.inf:
            scores.append(0)
        else:
            scores.append((b - a) / max(a, b))
    
    return np.mean(scores)
```

---

## 📊 Algorithm Comparison

| Algorithm | Cluster Shape | Complexity | Outliers |
|-----------|---------------|------------|----------|
| K-Means | Spherical | $O(nkd)$ | Sensitive |
| GMM | Elliptical | $O(nkd^2)$ | Sensitive |
| Spectral | Arbitrary | $O(n^3)$ | Moderate |
| DBSCAN | Arbitrary | $O(n^2)$ | Robust |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | K-means++ | [Arthur & Vassilvitskii](https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf) |
| 📄 | Spectral Clustering Tutorial | [von Luxburg](https://arxiv.org/abs/0711.0189) |
| 📄 | DBSCAN | [Ester et al.](https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf) |

---

⬅️ [Back: Ensemble Methods](../06_ensemble_methods/) | ➡️ [Next: Model Selection](../08_model_selection/)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<!-- Navigation -->
<p align="center">
  <a href="../06_ensemble_methods/">⬅️ Prev: Ensemble Methods</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../">📚 ML Theory</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../08_model_selection/">Next: Model Selection ➡️</a>
</p>

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=80&section=footer" width="100%"/>
</p>
