<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=9C27B0&height=120&section=header&text=Conditional%20Gaussian&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-03-9C27B0?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 📐 Setup

```
Joint Gaussian:
[X₁]     [μ₁]   [Σ₁₁  Σ₁₂]
[X₂] ~ N([μ₂], [Σ₂₁  Σ₂₂])
```

---

## 📐 Conditioning Formula

```
X₁ | X₂ = x₂ ~ N(μ_{1|2}, Σ_{1|2})

Where:
μ_{1|2} = μ₁ + Σ₁₂ Σ₂₂⁻¹ (x₂ - μ₂)
Σ_{1|2} = Σ₁₁ - Σ₁₂ Σ₂₂⁻¹ Σ₂₁

"Mean shifts linearly, variance shrinks"
```

---

## 🔑 Key Properties

| Property | Value |
|----------|-------|
| Conditional is Gaussian | ✓ Always |
| Conditional mean | Linear in condition |
| Conditional variance | Independent of condition |

---

## 💻 Code

```python
import numpy as np

def conditional_gaussian(mu1, mu2, Sigma11, Sigma12, Sigma22, x2):
    """
    Compute p(x1 | x2) for joint Gaussian
    """
    Sigma22_inv = np.linalg.inv(Sigma22)
    
    # Conditional mean
    mu_cond = mu1 + Sigma12 @ Sigma22_inv @ (x2 - mu2)
    
    # Conditional covariance
    Sigma_cond = Sigma11 - Sigma12 @ Sigma22_inv @ Sigma12.T
    
    return mu_cond, Sigma_cond

# Example: 2D Gaussian
mu = np.array([0, 0])
Sigma = np.array([[1, 0.8], [0.8, 1]])  # Correlated

# Condition on x2 = 1
mu_cond, Sigma_cond = conditional_gaussian(
    mu[:1], mu[1:], 
    Sigma[:1, :1], Sigma[:1, 1:], Sigma[1:, 1:],
    x2=np.array([1])
)
print(f"p(x1|x2=1) ~ N({mu_cond[0]:.2f}, {Sigma_cond[0,0]:.2f})")
```

---

## 🌍 Applications

| Application | How Used |
|-------------|----------|
| Gaussian Processes | Predict at new points |
| Kalman Filter | Update state estimate |
| VAE (Gaussian) | Reparameterization |

---

<- [Back](./README.md)

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=9C27B0&height=80&section=footer" width="100%"/>
</p>
