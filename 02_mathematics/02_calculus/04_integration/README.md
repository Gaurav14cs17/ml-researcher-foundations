<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=00C853&height=120&section=header&text=Integration&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-02-00C853?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/integration.svg" width="100%">

*Caption: Integration computes areas under curves. In ML, this appears in probability (normalization, expectations, marginalization) and is often approximated via Monte Carlo sampling when intractable.*

---

## 📂 Overview

Integration is the inverse of differentiation. In ML, we rarely compute integrals analytically - instead we use Monte Carlo approximation or variational methods. This section covers the mathematical foundations with complete proofs.

---

## 🔑 Key Concepts

| Concept | Formula | ML Usage |
|---------|---------|----------|
| **Expectation** | \(\mathbb{E}[X] = \int x \cdot p(x) \, dx\) | Mean predictions |
| **Variance** | \(\text{Var}[X] = \int (x-\mu)^2 p(x) \, dx\) | Uncertainty quantification |
| **Normalization** | \(\int p(x) \, dx = 1\) | Valid probability distributions |
| **Marginalization** | \(p(x) = \int p(x,z) \, dz\) | Latent variable models |
| **ELBO** | \(\mathcal{L} = \mathbb{E}_{q}[\log p(x|z)] - D_{KL}(q||p)\) | VAE training |

---

## 📐 Formal Definition of Integration

### 1. Riemann Integral

**Definition:** The Riemann integral of $f$ over $[a, b]$ is defined as:

```math
\int_a^b f(x) \, dx = \lim_{n \to \infty} \sum_{i=1}^{n} f(x_i^*) \Delta x
```

where $\Delta x = \frac{b-a}{n}$ and $x_i^* \in [x_{i-1}, x_i]$.

**Intuition:** We approximate the area under the curve by summing up rectangles, then take the limit as rectangles become infinitely thin.

### 2. Fundamental Theorem of Calculus

**Theorem (Part 1):** If $f$ is continuous on $[a, b]$, then:

```math
F(x) = \int_a^x f(t) \, dt
```

is differentiable and \(F'(x) = f(x)\).

**Proof:**
```
By definition of derivative:
F'(x) = lim_{h→0} [F(x+h) - F(x)] / h
      = lim_{h→0} [∫_a^{x+h} f(t)dt - ∫_a^x f(t)dt] / h
      = lim_{h→0} [∫_x^{x+h} f(t)dt] / h
      
By Mean Value Theorem for Integrals:
∫_x^{x+h} f(t)dt = f(c) · h  for some c ∈ [x, x+h]

So: F'(x) = lim_{h→0} f(c) · h / h = lim_{h→0} f(c)

As h → 0, c → x, so by continuity: F'(x) = f(x) ✓
```

**Theorem (Part 2):** If $F$ is an antiderivative of $f$ on $[a, b]$, then:

```math
\int_a^b f(x) \, dx = F(b) - F(a)
```

**Proof:**
```
Let G(x) = ∫_a^x f(t)dt

By Part 1: G'(x) = f(x)
Also given: F'(x) = f(x)

Therefore: G(x) = F(x) + C for some constant C

At x = a: G(a) = ∫_a^a f(t)dt = 0
So: 0 = F(a) + C, which gives C = -F(a)

At x = b: G(b) = F(b) + C = F(b) - F(a)

Therefore: ∫_a^b f(x)dx = G(b) = F(b) - F(a) ✓
```

---

## 📊 Integration Techniques

### 1. Substitution (Change of Variables)

**Theorem:** If \(u = g(x)\) is differentiable and $f$ is continuous, then:

```math
\int f(g(x)) \cdot g'(x) \, dx = \int f(u) \, du
```

**Proof:**
```
Let F be an antiderivative of f, so F'(u) = f(u)

By chain rule:
d/dx[F(g(x))] = F'(g(x)) · g'(x) = f(g(x)) · g'(x)

Therefore F(g(x)) is an antiderivative of f(g(x))·g'(x)

∫f(g(x))·g'(x)dx = F(g(x)) + C = F(u) + C = ∫f(u)du ✓
```

**Example (Gaussian normalization):**
```
∫_{-∞}^{∞} e^{-x²/2} dx

Let I = ∫_{-∞}^{∞} e^{-x²/2} dx

Then: I² = ∫∫ e^{-(x²+y²)/2} dx dy

Convert to polar: x = r cos θ, y = r sin θ
Jacobian: dx dy = r dr dθ

I² = ∫_0^{2π} ∫_0^{∞} e^{-r²/2} r dr dθ
   = 2π ∫_0^{∞} r e^{-r²/2} dr

Let u = r²/2, du = r dr:
I² = 2π ∫_0^{∞} e^{-u} du = 2π[-e^{-u}]_0^{∞} = 2π

Therefore: I = √(2π)

This proves: ∫_{-∞}^{∞} e^{-x²/2} dx = √(2π) ✓
```

### 2. Integration by Parts

**Theorem:** 

```math
\int u \, dv = uv - \int v \, du
```

**Proof:**
```
From product rule: d(uv) = u dv + v du
Rearranging: u dv = d(uv) - v du
Integrating both sides: ∫u dv = uv - ∫v du ✓
```

**Example (Computing E[X] for exponential distribution):**
```
For X ~ Exp(λ): p(x) = λe^{-λx} for x ≥ 0

E[X] = ∫_0^∞ x · λe^{-λx} dx

Let u = x, dv = λe^{-λx}dx
Then du = dx, v = -e^{-λx}

E[X] = [-xe^{-λx}]_0^∞ + ∫_0^∞ e^{-λx} dx
     = 0 + [-1/λ · e^{-λx}]_0^∞
     = 0 - (-1/λ)
     = 1/λ ✓
```

---

## 🎯 Integration in Machine Learning

### 1. Expectation and Variance

**Definition:** For a random variable $X$ with PDF \(p(x)\):

```math
\mathbb{E}[f(X)] = \int_{-\infty}^{\infty} f(x) \cdot p(x) \, dx
```

**Variance derivation:**
```
Var[X] = E[(X - μ)²]
       = E[X² - 2μX + μ²]
       = E[X²] - 2μE[X] + μ²
       = E[X²] - 2μ² + μ²
       = E[X²] - μ²
       = E[X²] - (E[X])²

This is the computational formula for variance.
```

### 2. Marginalization

**Theorem:** For joint distribution \(p(x, z)\), the marginal is:

```math
p(x) = \int p(x, z) \, dz = \int p(x|z) p(z) \, dz
```

**Proof:**
```
By definition of conditional probability:
p(x, z) = p(x|z) · p(z)

Therefore:
p(x) = ∫ p(x, z) dz = ∫ p(x|z) · p(z) dz ✓

This is the law of total probability (continuous form).
```

**ML Application:** In VAEs, the marginal likelihood is:
```
p(x) = ∫ p(x|z) p(z) dz

This integral is intractable for complex decoders,
so we use variational inference with ELBO.
```

### 3. ELBO Derivation (Complete Proof)

**Problem:** We want to maximize \(\log p(x)\) but the integral is intractable.

**Solution:** Introduce variational distribution \(q(z|x)\) and derive a lower bound.

**Complete Derivation:**
```
log p(x) = log ∫ p(x, z) dz
         = log ∫ p(x, z) · q(z|x)/q(z|x) dz
         = log E_{q(z|x)}[p(x, z)/q(z|x)]

By Jensen's inequality (log is concave):
log E[Y] ≥ E[log Y]

Therefore:
log p(x) ≥ E_{q(z|x)}[log(p(x, z)/q(z|x))]
         = E_{q(z|x)}[log p(x, z) - log q(z|x)]
         = E_{q(z|x)}[log p(x|z) + log p(z) - log q(z|x)]
         = E_{q(z|x)}[log p(x|z)] - E_{q(z|x)}[log q(z|x) - log p(z)]
         = E_{q(z|x)}[log p(x|z)] - D_{KL}(q(z|x) || p(z))
         = ELBO

This is the Evidence Lower BOund used in VAE training!
```

**Why it's a bound:**
```
Gap = log p(x) - ELBO = D_{KL}(q(z|x) || p(z|x)) ≥ 0

The bound is tight when q(z|x) = p(z|x) (true posterior).
```

---

## 📊 Monte Carlo Integration

### Theory

**Problem:** Computing \(\int f(x) p(x) dx\) analytically is often intractable.

**Solution:** Use random sampling to approximate.

**Theorem (Monte Carlo Estimator):**

```math
\mathbb{E}[f(X)] \approx \frac{1}{N} \sum_{i=1}^{N} f(x_i) \quad \text{where } x_i \sim p(x)
```

**Proof of Unbiasedness:**
```
Let μ̂ = (1/N) Σᵢ f(xᵢ)

E[μ̂] = E[(1/N) Σᵢ f(xᵢ)]
      = (1/N) Σᵢ E[f(xᵢ)]
      = (1/N) · N · E[f(X)]
      = E[f(X)]
      = ∫ f(x) p(x) dx ✓

The estimator is unbiased.
```

**Variance of Estimator:**
```
Var[μ̂] = Var[(1/N) Σᵢ f(xᵢ)]
       = (1/N²) Σᵢ Var[f(xᵢ)]   (independence)
       = (1/N²) · N · Var[f(X)]
       = Var[f(X)] / N

Standard error = σ / √N

This shows convergence rate is O(1/√N).
```

### Importance Sampling

**Problem:** What if we can't sample from \(p(x)\) directly?

**Solution:** Sample from a proposal distribution \(q(x)\) and reweight.

**Theorem:**

```math
\mathbb{E}_p[f(X)] = \mathbb{E}_q\left[f(X) \cdot \frac{p(X)}{q(X)}\right] \approx \frac{1}{N} \sum_{i=1}^{N} f(x_i) \cdot \frac{p(x_i)}{q(x_i)}
```

**Proof:**
```
E_q[f(X) · p(X)/q(X)] = ∫ f(x) · p(x)/q(x) · q(x) dx
                      = ∫ f(x) · p(x) dx
                      = E_p[f(X)] ✓
```

**Optimal proposal:**
```
Var is minimized when q(x) ∝ |f(x)| · p(x)
In practice, choose q to have heavier tails than p.
```

---

## 📐 Multidimensional Integration

### Jacobian Change of Variables

**Theorem:** For transformation \(y = g(x)\) where $x \in \mathbb{R}^n$:

```math
\int f(y) \, dy = \int f(g(x)) \cdot |\det(J_g(x))| \, dx
```

where $J_g$ is the Jacobian matrix.

**Proof sketch:**
```
The Jacobian determinant |det(J)| measures how volumes scale
under the transformation. Each small volume element transforms as:

dy₁ dy₂ ... dyₙ = |det(J)| dx₁ dx₂ ... dxₙ
```

**Example (Normalizing Flows):**
```
In normalizing flows, we transform z ~ p_z(z) to x = f(z)

p_x(x) = p_z(f⁻¹(x)) · |det(∂f⁻¹/∂x)|
       = p_z(z) · |det(∂f/∂z)|⁻¹

Log-likelihood:
log p_x(x) = log p_z(z) - log|det(∂f/∂z)|

This is tractable if the Jacobian has special structure.
```

---

## 💻 Code Examples

### Basic Monte Carlo Integration

```python
import numpy as np
import torch

# Monte Carlo estimate of E[f(X)] where X ~ N(0,1)
def monte_carlo_expectation(f, n_samples=100000):
    """
    Estimate E[f(X)] where X ~ N(0,1)
    """
    samples = np.random.normal(0, 1, n_samples)
    return np.mean(f(samples)), np.std(f(samples)) / np.sqrt(n_samples)

# Example 1: E[X²] where X ~ N(0,1) should be 1 (variance)
mean, std_err = monte_carlo_expectation(lambda x: x**2)
print(f"E[X²] = {mean:.4f} ± {std_err:.4f}")  # ≈ 1.0

# Example 2: E[e^X] where X ~ N(0,1)
# Analytical: E[e^X] = e^{μ + σ²/2} = e^{0.5} ≈ 1.6487
mean, std_err = monte_carlo_expectation(lambda x: np.exp(x))
print(f"E[e^X] = {mean:.4f} ± {std_err:.4f}")  # ≈ 1.6487
```

### Importance Sampling

```python
import numpy as np
from scipy import stats

def importance_sampling(f, p, q, n_samples=100000):
    """
    Estimate E_p[f(X)] using samples from q(x)
    
    Args:
        f: function to evaluate
        p: target distribution (scipy.stats object)
        q: proposal distribution (scipy.stats object)
    """

    # Sample from proposal
    samples = q.rvs(n_samples)
    
    # Compute importance weights
    weights = p.pdf(samples) / q.pdf(samples)
    
    # Weighted average
    estimate = np.mean(f(samples) * weights)
    return estimate

# Example: Estimate P(X > 3) where X ~ N(0,1)
# This is a rare event, so importance sampling helps

# Target: N(0, 1)
p = stats.norm(0, 1)

# Proposal: N(3, 1) - centered at the rare event region
q = stats.norm(3, 1)

# Function: indicator for X > 3
f = lambda x: (x > 3).astype(float)

# Standard MC (inefficient for rare events)
samples_standard = p.rvs(100000)
estimate_standard = np.mean(f(samples_standard))

# Importance sampling
estimate_is = importance_sampling(f, p, q)

print(f"True P(X > 3) = {1 - p.cdf(3):.6f}")
print(f"Standard MC estimate = {estimate_standard:.6f}")
print(f"Importance sampling estimate = {estimate_is:.6f}")
```

### Reparameterization Trick (VAE)

```python
import torch
import torch.nn as nn

class VAEEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
    
    def forward(self, x):
        h = torch.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = μ + σ * ε
        where ε ~ N(0, I)
        
        This allows gradients to flow through the sampling operation.
        
        Mathematical justification:
        If ε ~ N(0, I), then z = μ + σ*ε ~ N(μ, σ²I)
        
        The gradient ∂z/∂μ = 1 and ∂z/∂σ = ε
        Both are well-defined, unlike ∂/∂μ of sampling operation.
        """
        std = torch.exp(0.5 * logvar)  # σ = exp(log(σ²)/2)
        eps = torch.randn_like(std)     # ε ~ N(0, I)
        z = mu + eps * std              # z ~ N(μ, σ²I)
        return z

# ELBO loss computation
def vae_loss(x, x_recon, mu, logvar):
    """
    ELBO = E_q[log p(x|z)] - D_KL(q(z|x) || p(z))
    
    Reconstruction loss (negative log likelihood):
    -E_q[log p(x|z)] ≈ ||x - x_recon||²  (for Gaussian decoder)
    
    KL divergence (analytical for Gaussians):
    D_KL(N(μ, σ²) || N(0, 1)) = 0.5 * Σ(μ² + σ² - log(σ²) - 1)
    """

    # Reconstruction loss
    recon_loss = nn.functional.mse_loss(x_recon, x, reduction='sum')
    
    # KL divergence: D_KL(q(z|x) || p(z))
    # For q(z|x) = N(μ, σ²) and p(z) = N(0, 1):
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + kl_loss
```

### Numerical Integration (Simpson's Rule)

```python
import numpy as np

def simpsons_rule(f, a, b, n=1000):
    """
    Approximate ∫_a^b f(x) dx using Simpson's rule
    
    Formula: ∫f(x)dx ≈ (h/3)[f(x₀) + 4f(x₁) + 2f(x₂) + 4f(x₃) + ... + f(xₙ)]
    
    Error: O(h⁴) where h = (b-a)/n
    """
    if n % 2 == 1:
        n += 1  # Simpson's rule requires even n
    
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    
    # Simpson's formula
    result = y[0] + y[-1]  # endpoints
    result += 4 * np.sum(y[1:-1:2])  # odd indices
    result += 2 * np.sum(y[2:-1:2])  # even indices
    result *= h / 3
    
    return result

# Example: ∫_0^π sin(x) dx = 2
result = simpsons_rule(np.sin, 0, np.pi, n=100)
print(f"∫sin(x)dx from 0 to π = {result:.10f}")  # Should be 2.0

# Example: Gaussian integral ∫_{-10}^{10} e^{-x²} dx ≈ √π
result = simpsons_rule(lambda x: np.exp(-x**2), -10, 10, n=1000)
print(f"∫e^(-x²)dx ≈ {result:.10f}, √π = {np.sqrt(np.pi):.10f}")
```

---

## 🔗 Where This Topic Is Used

| Application | Usage | Mathematical Form |
|-------------|-------|-------------------|
| **VAE** | ELBO optimization | \(\mathbb{E}_q[\log p(x|z)] - D_{KL}\) |
| **Policy Gradient** | Expected reward | $\nabla_\theta \mathbb{E}[\sum r_t]$ |
| **Bayesian Inference** | Posterior normalization | \(p(\theta|D) = \frac{p(D|\theta)p(\theta)}{\int p(D|\theta)p(\theta)d\theta}\) |
| **Normalizing Flows** | Change of variables | \(\log p_x(x) = \log p_z(z) - \log|\det J|\) |
| **Diffusion Models** | Denoising objective | \(\mathbb{E}_{t,\epsilon}[\|\epsilon - \epsilon_\theta(x_t, t)\|^2]\) |

---

## 📚 Key Theorems Summary

| Theorem | Statement | ML Application |
|---------|-----------|----------------|
| **Fundamental Theorem** | \(\frac{d}{dx}\int_a^x f(t)dt = f(x)\) | Gradient computation |
| **Jensen's Inequality** | $\log \mathbb{E}[X] \geq \mathbb{E}[\log X]$ | ELBO derivation |
| **Law of Large Numbers** | $\bar{X}_n \to \mu$ as $n \to \infty$ | Monte Carlo convergence |
| **CLT** | \(\sqrt{n}(\bar{X}_n - \mu) \to N(0, \sigma^2)\) | Error bounds |

---

## 📚 References

| Type | Resource | Link |
|------|----------|------|
| 📖 | Calculus (Stewart) | Standard reference |
| 📖 | Pattern Recognition and ML | Bishop, Ch. 2 |
| 🎥 | 3Blue1Brown: Essence of Calculus | [YouTube](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr) |
| 📄 | Auto-Encoding Variational Bayes | Kingma & Welling, 2013 |

---

## 🗺️ Navigation

| ⬅️ Previous | 🏠 Home | ➡️ Next |
|:-----------:|:-------:|:-------:|
| [Gradients](../03_gradients/README.md) | [Calculus](../README.md) | [Limits & Continuity](../05_limits_continuity/README.md) |

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=00C853&height=80&section=footer" width="100%"/>
</p>
