<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=9C27B0&height=120&section=header&text=Multivariate%20Statistics&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-03-9C27B0?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/covariance-matrix.svg" width="100%">

*Caption: The covariance matrix Σ captures relationships between variables. Diagonal elements are variances, off-diagonal elements are covariances. This is central to PCA, VAE, Gaussian Processes, and many other ML methods.*

---

## 📂 Topics

| Folder | Topic | Key Concepts |
|--------|-------|--------------|
| [random-vectors/](./random-vectors/) | Joint distributions | p(x,y) |
| [covariance/](./covariance/) | Covariance structure | Σ matrix |
| [gaussian/](./gaussian/) | 🔥 Multivariate Gaussian | Most important! |
| [exponential-family/](./exponential-family/) | General framework | GLMs |

---

## 📐 Key Concepts

```
Joint distribution:     p(x, y)
Marginal:              p(x) = ∫p(x,y)dy
Conditional:           p(x|y) = p(x,y)/p(y)
Independence:          p(x,y) = p(x)p(y)
```

---

## 🔥 Covariance Matrix

```
Σ = E[(X - μ)(X - μ)ᵀ]

Σᵢⱼ = Cov(Xᵢ, Xⱼ)

Properties:
• Symmetric: Σ = Σᵀ
• Positive semi-definite: xᵀΣx ≥ 0
• Diagonal = variances
```

---

## 🔗 Where This Topic Is Used

| Topic | How Multivariate Stats Is Used |
|-------|-------------------------------|
| **PCA** | Eigendecomposition of covariance matrix |
| **VAE** | Latent space is multivariate Gaussian |
| **Diffusion Models** | Forward process adds Gaussian noise |
| **Gaussian Processes** | Multivariate Gaussian over functions |
| **Bayesian Neural Networks** | Weight distributions |
| **Kalman Filter** | Multivariate Gaussian state estimation |
| **GMM (Gaussian Mixture)** | Clustering with Gaussians |
| **Normalizing Flows** | Transform simple → complex distributions |
| **ELBO** | KL between multivariate distributions |

### Used In These Models

| Model/Method | Multivariate Stats Concept |
|--------------|---------------------------|
| **VAE** | Reparameterization of Gaussian |
| **Diffusion** | Gaussian noise schedule |
| **CLIP** | Covariance in embedding space |
| **Batch Norm** | Running mean and variance |
| **Attention** | Softmax over multivariate scores |

### Prerequisite For

```
Multivariate Stats --> VAE (latent Gaussian)
                  --> Diffusion models
                  --> Gaussian Processes
                  --> PCA / Factor Analysis
                  --> Bayesian inference
```

---

⬅️ [Back: 01-Probability](../01-probability/) | ➡️ [Next: 03-Information Theory](../03-information-theory/)


---

## 📚 References

| Type | Resource | Link |
|------|----------|------|
| 📖 | Textbook | See parent folder |
| 🇨🇳 | 中文资源 | 知乎/B站 |


---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=9C27B0&height=80&section=footer" width="100%"/>
</p>
