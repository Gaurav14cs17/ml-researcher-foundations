<!-- Navigation -->
<p align="center">
  <a href="../02_supervised/">⬅️ Prev: Supervised</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../">📚 Learning Frameworks</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../../02_generalization/">Next: Generalization ➡️</a>
</p>

---

<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=120&section=header&text=Unsupervised%20Learning&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-05-4ECDC4?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/unsupervised.svg" width="100%">

*Caption: Unsupervised learning discovers hidden structure in data without labels.*

---

## 📂 Overview

**Unsupervised learning** finds patterns in data without supervision. Key tasks include clustering, dimensionality reduction, density estimation, and generative modeling.

---

## 📐 Clustering

### K-Means

**Objective:**

$$\min_{\mu_1,...,\mu_K} \sum_{i=1}^n \min_{k=1,...,K} \|x_i - \mu_k\|^2$$

**Algorithm:**

1. **Assign:** \(z_i = \arg\min_k \|x_i - \mu_k\|^2\)

2. **Update:** \(\mu_k = \frac{1}{|C_k|}\sum_{i \in C_k} x_i\)

**Theorem:** K-means converges to a local minimum.

**Proof:** Each step decreases or maintains the objective. Assignment step: each point moves to closest centroid. Update step: centroid is mean of cluster (minimizes sum of squared distances). Objective is bounded below by 0. \(\blacksquare\)

### Gaussian Mixture Models (GMM)

**Model:**

$$p(x) = \sum_{k=1}^K \pi_k \mathcal{N}(x | \mu_k, \Sigma_k)$$

where \(\sum_k \pi_k = 1\) and \(\pi_k \geq 0\).

**EM Algorithm:**

**E-step:** Compute responsibilities:

$$\gamma_{ik} = \frac{\pi_k \mathcal{N}(x_i | \mu_k, \Sigma_k)}{\sum_j \pi_j \mathcal{N}(x_i | \mu_j, \Sigma_j)}$$

**M-step:** Update parameters:

$$\mu_k = \frac{\sum_i \gamma_{ik} x_i}{\sum_i \gamma_{ik}}, \quad \Sigma_k = \frac{\sum_i \gamma_{ik}(x_i - \mu_k)(x_i - \mu_k)^\top}{\sum_i \gamma_{ik}}
\pi_k = \frac{1}{n}\sum_i \gamma_{ik}$$

---

## 📐 Dimensionality Reduction

### Principal Component Analysis (PCA)

**Objective:** Find directions of maximum variance.

$$\max_{w: \|w\|=1} w^\top \Sigma w$$

where \(\Sigma = \frac{1}{n}\sum_i (x_i - \bar{x})(x_i - \bar{x})^\top\) is the covariance matrix.

**Solution:** \(w_1\) = eigenvector of \(\Sigma\) with largest eigenvalue.

**Theorem:** The first \(k\) principal components minimize reconstruction error:

$$\min_{W \in \mathbb{R}^{d \times k}} \sum_{i=1}^n \|x_i - WW^\top x_i\|^2$$

**Proof:** Using SVD of centered data matrix \(X = U\Sigma V^\top\), optimal \(W\) is first \(k\) columns of \(V\). \(\blacksquare\)

### Kernel PCA

For non-linear dimensionality reduction:

1. Compute kernel matrix \(K_{ij} = k(x_i, x_j)\)

2. Center: \(\tilde{K} = K - \frac{1}{n}\mathbf{1}\mathbf{1}^\top K - K\frac{1}{n}\mathbf{1}\mathbf{1}^\top + \frac{1}{n^2}\mathbf{1}\mathbf{1}^\top K \mathbf{1}\mathbf{1}^\top\)

3. Eigendecompose \(\tilde{K}\) and project

---

## 📐 Variational Autoencoders (VAE)

### Evidence Lower Bound (ELBO)

$$\log p(x) \geq \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{\text{KL}}(q(z|x) \| p(z)) = \mathcal{L}_{\text{ELBO}}$$

**Proof:**

$$\log p(x) = \log \int p(x, z) dz = \log \int \frac{p(x, z)}{q(z|x)} q(z|x) dz$$

By Jensen's inequality:

$$\geq \int q(z|x) \log \frac{p(x, z)}{q(z|x)} dz = \mathbb{E}_q[\log p(x|z)] - D_{\text{KL}}(q \| p(z)) \quad \blacksquare$$

### Reparameterization Trick

To enable backpropagation through sampling:

$$z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

### KL Divergence (Gaussian)

$$D_{\text{KL}}(\mathcal{N}(\mu, \sigma^2) \| \mathcal{N}(0, 1)) = \frac{1}{2}\sum_j (\mu_j^2 + \sigma_j^2 - \log \sigma_j^2 - 1)$$

---

## 💻 Code Implementation

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class KMeans:
    """
    K-Means clustering.
    
    min Σᵢ min_k ||xᵢ - μ_k||²
    """
    
    def __init__(self, n_clusters, max_iters=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
    
    def fit(self, X):
        n, d = X.shape
        
        # Initialize centroids randomly
        idx = np.random.choice(n, self.n_clusters, replace=False)
        self.centroids = X[idx].copy()
        
        for _ in range(self.max_iters):
            old_centroids = self.centroids.copy()
            
            # E-step: Assign points to nearest centroid
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            self.labels = np.argmin(distances, axis=1)
            
            # M-step: Update centroids
            for k in range(self.n_clusters):
                mask = self.labels == k
                if np.sum(mask) > 0:
                    self.centroids[k] = X[mask].mean(axis=0)
            
            # Check convergence
            if np.linalg.norm(self.centroids - old_centroids) < self.tol:
                break
        
        return self
    
    def predict(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

class PCA:
    """
    Principal Component Analysis.
    
    Find directions of maximum variance: max_{||w||=1} w'Σw
    """
    
    def __init__(self, n_components):
        self.n_components = n_components
    
    def fit(self, X):
        # Center data
        self.mean = X.mean(axis=0)
        X_centered = X - self.mean
        
        # Compute covariance matrix
        cov = np.cov(X_centered, rowvar=False)
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Sort by decreasing eigenvalue
        idx = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = eigenvalues[idx]
        self.components = eigenvectors[:, idx][:, :self.n_components]
        
        # Explained variance ratio
        self.explained_variance_ratio = self.eigenvalues[:self.n_components] / self.eigenvalues.sum()
        
        return self
    
    def transform(self, X):
        return (X - self.mean) @ self.components
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_reduced):
        return X_reduced @ self.components.T + self.mean

class VAE(nn.Module):
    """
    Variational Autoencoder.
    
    ELBO = E_q[log p(x|z)] - KL(q(z|x) || p(z))
    """
    
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """z = μ + σ·ε where ε ~ N(0, I)"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
    
    def loss(self, x, x_recon, mu, logvar):
        """
        ELBO loss = Reconstruction + KL divergence
        
        Reconstruction: E_q[log p(x|z)] ≈ -||x - x̂||²
        KL: D_KL(N(μ,σ²) || N(0,1)) = ½Σ(μ² + σ² - log(σ²) - 1)
        """
        # Reconstruction loss (binary cross-entropy for images)
        recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return recon_loss + kl_loss

class GMM:
    """
    Gaussian Mixture Model via EM algorithm.
    
    p(x) = Σ_k π_k N(x | μ_k, Σ_k)
    """
    
    def __init__(self, n_components, max_iters=100, tol=1e-4):
        self.n_components = n_components
        self.max_iters = max_iters
        self.tol = tol
    
    def fit(self, X):
        n, d = X.shape
        K = self.n_components
        
        # Initialize with K-means
        kmeans = KMeans(K)
        kmeans.fit(X)
        
        self.means = kmeans.centroids
        self.covariances = np.array([np.eye(d) for _ in range(K)])
        self.weights = np.ones(K) / K
        
        for _ in range(self.max_iters):
            # E-step: compute responsibilities
            resp = self._e_step(X)
            
            # M-step: update parameters
            old_means = self.means.copy()
            self._m_step(X, resp)
            
            # Check convergence
            if np.linalg.norm(self.means - old_means) < self.tol:
                break
        
        return self
    
    def _e_step(self, X):
        """Compute responsibilities γ_ik = p(z_i = k | x_i)"""
        n = len(X)
        K = self.n_components
        
        resp = np.zeros((n, K))
        for k in range(K):
            resp[:, k] = self.weights[k] * self._gaussian_pdf(X, self.means[k], self.covariances[k])
        
        resp /= resp.sum(axis=1, keepdims=True) + 1e-10
        return resp
    
    def _m_step(self, X, resp):
        """Update parameters"""
        n = len(X)
        
        for k in range(self.n_components):
            Nk = resp[:, k].sum()
            self.weights[k] = Nk / n
            self.means[k] = (resp[:, k:k+1].T @ X).flatten() / Nk
            diff = X - self.means[k]
            self.covariances[k] = (resp[:, k:k+1] * diff).T @ diff / Nk
            self.covariances[k] += 1e-6 * np.eye(X.shape[1])  # Regularization
    
    def _gaussian_pdf(self, X, mean, cov):
        """Multivariate Gaussian PDF"""
        d = len(mean)
        diff = X - mean
        inv_cov = np.linalg.inv(cov)
        det = np.linalg.det(cov)
        
        exponent = -0.5 * np.sum(diff @ inv_cov * diff, axis=1)
        return np.exp(exponent) / np.sqrt((2 * np.pi) ** d * det)

```

---

## 📊 Method Comparison

| Method | Task | Assumptions | Output |
|--------|------|-------------|--------|
| **K-Means** | Clustering | Spherical clusters | Hard assignment |
| **GMM** | Clustering | Gaussian clusters | Soft assignment |
| **PCA** | Dimensionality | Linear | Principal components |
| **VAE** | Generative | Latent Gaussian | Decoder network |
| **t-SNE** | Visualization | Local structure | 2D/3D embedding |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | Bishop PRML Ch. 9 | [Book](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) |
| 📄 | VAE Paper | [arXiv](https://arxiv.org/abs/1312.6114) |
| 📄 | t-SNE | [JMLR](https://www.jmlr.org/papers/v9/vandermaaten08a.html) |

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<!-- Navigation -->
<p align="center">
  <a href="../02_supervised/">⬅️ Prev: Supervised</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../">📚 Learning Frameworks</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../../02_generalization/">Next: Generalization ➡️</a>
</p>

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=80&section=footer" width="100%"/>
</p>
