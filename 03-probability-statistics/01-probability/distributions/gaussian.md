<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=Gaussian%20Distribution&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---

# Gaussian Distribution

> **The most important distribution in ML**

---

## 📐 Definition

```
Univariate:
p(x) = (1/√(2πσ²)) exp(-(x-μ)²/(2σ²))

Parameters:
• μ = mean
• σ² = variance
```

---

## 🔑 Properties

| Property | Value |
|----------|-------|
| Mean | μ |
| Variance | σ² |
| Mode | μ |
| Entropy | ½log(2πeσ²) |

---

## 🌍 Why So Common?

```
1. Central Limit Theorem: Sum of many RVs → Gaussian
2. Maximum entropy: Most "uncertain" for given mean/variance
3. Conjugate prior: Posterior is also Gaussian
4. Mathematical convenience: Closed-form operations
```

---

## 💻 Code

```python
import numpy as np
from scipy import stats

# Create distribution
dist = stats.norm(loc=0, scale=1)  # N(0,1)

# Sample
samples = dist.rvs(size=1000)

# Probability
dist.pdf(0)      # p(x=0)
dist.cdf(1.96)   # P(X ≤ 1.96) ≈ 0.975

# PyTorch
import torch
dist = torch.distributions.Normal(0, 1)
samples = dist.sample((1000,))
log_prob = dist.log_prob(samples)
```

---

## 📐 DETAILED MATHEMATICAL THEORY

### 1. Gaussian Distribution: Complete Derivation

**Probability Density Function:**

```
Univariate:
  p(x; μ, σ²) = (1/√(2πσ²)) exp(-(x-μ)²/(2σ²))

Multivariate (d dimensions):
  p(x; μ, Σ) = (1/√((2π)^d|Σ|)) exp(-(1/2)(x-μ)ᵀΣ⁻¹(x-μ))

where:
  μ ∈ ℝ^d         = mean vector
  Σ ∈ ℝ^{d×d}     = covariance matrix (symmetric, positive definite)
  |Σ|             = determinant of Σ
  Σ⁻¹             = inverse (precision matrix)
```

---

### 2. Maximum Entropy Property

**Theorem: Gaussian is Maximum Entropy Distribution**

```
Among all distributions with fixed mean μ and variance σ²:
  The Gaussian has maximum entropy H(X)

Entropy of Gaussian:
  H(X) = (1/2)log(2πeσ²)  nats
       = (1/2)log₂(2πeσ²)  bits
```

**Proof:**

```
Step 1: Entropy definition
  H(X) = -∫ p(x)log p(x) dx

Step 2: Optimization problem
  maximize: H(X) = -∫ p(x)log p(x) dx
  subject to:
    ∫ p(x) dx = 1              (normalization)
    ∫ x·p(x) dx = μ            (fixed mean)
    ∫ (x-μ)²·p(x) dx = σ²      (fixed variance)

Step 3: Lagrangian
  L[p] = -∫ p log p dx + λ₀(∫ p dx - 1) + λ₁(∫ xp dx - μ) + λ₂(∫ (x-μ)²p dx - σ²)

Step 4: Functional derivative (Calculus of Variations)
  δL/δp = 0
  
  -log p - 1 + λ₀ + λ₁x + λ₂(x-μ)² = 0
  
  log p = -1 + λ₀ + λ₁x + λ₂(x-μ)²
  
  p(x) = exp(-1 + λ₀ + λ₁x + λ₂(x-μ)²)

Step 5: Impose constraints
  From fixed mean: λ₁ = 0
  From fixed variance: λ₂ = -1/(2σ²)
  From normalization: λ₀ = 1 - (1/2)log(2πσ²)
  
  p(x) = (1/√(2πσ²))·exp(-(x-μ)²/(2σ²))
  
  This is the Gaussian distribution! ✓  QED

Multivariate case: Same proof with vector calculus
  Result: N(μ, Σ) has entropy H = (d/2)log(2πe) + (1/2)log|Σ|
```

**Why This Matters:**

```
Maximum entropy principle:
  "Given constraints, choose distribution with maximum uncertainty"
  
In ML:
  • Don't overcommit to specific structure without evidence
  • Gaussian is the "least informative" given mean & variance
  • Natural choice when only moments are known
```

---

### 3. Central Limit Theorem: Why Gaussians Appear Everywhere

**Theorem (Central Limit Theorem):**

```
Let X₁, X₂, ..., Xₙ be i.i.d. random variables with:
  • E[Xᵢ] = μ
  • Var[Xᵢ] = σ² < ∞
  
Define sample mean: X̄ₙ = (1/n)Σᵢ₌₁ⁿ Xᵢ

Then as n → ∞:
  √n(X̄ₙ - μ) →_d N(0, σ²)
  
Or equivalently:
  X̄ₙ →_d N(μ, σ²/n)
```

**Intuition:**

```
Sum of many small independent effects → Gaussian

Examples in ML:
  1. Gradient noise in SGD: Sum of per-sample gradients
  2. Neural network activations: Sum of many weighted inputs
  3. Measurement errors: Sum of many error sources
  4. Random initialization: Sum of many small random weights
```

**Proof Sketch (via Characteristic Functions):**

```
Step 1: Standardize
  Zᵢ = (Xᵢ - μ)/σ  (mean 0, variance 1)
  
  Sₙ = Σᵢ₌₁ⁿ Zᵢ/√n  (scaled sum)

Step 2: Characteristic function
  φ_Sₙ(t) = E[exp(itSₙ)]
          = E[exp(it·Z₁/√n)]ⁿ  (by independence)
          = [φ_Z(t/√n)]ⁿ

Step 3: Taylor expansion
  φ_Z(t/√n) = 1 + i(t/√n)E[Z] - (t²/2n)E[Z²] + O(t³/n^{3/2})
             = 1 - t²/(2n) + O(t³/n^{3/2})  (since E[Z]=0, E[Z²]=1)

Step 4: Take limit
  φ_Sₙ(t) = [1 - t²/(2n)]ⁿ → exp(-t²/2)  as n→∞
  
  exp(-t²/2) is the characteristic function of N(0,1)! ✓  QED
```

---

### 4. Conjugate Prior Property

**Theorem: Gaussian-Gaussian Conjugacy**

```
Likelihood: X|μ ~ N(μ, σ²)  (σ² known)
Prior:      μ ~ N(μ₀, σ₀²)

Then posterior is also Gaussian:
  μ|X ~ N(μₙ, σₙ²)

where:
  σₙ² = 1/(1/σ₀² + n/σ²)
  
  μₙ = σₙ²·(μ₀/σ₀² + nX̄/σ²)
     = (τ₀·μ₀ + τ·nX̄)/(τ₀ + τ·n)
     
  with τ = 1/σ², τ₀ = 1/σ₀² (precisions)
```

**Proof:**

```
Step 1: Write posterior (Bayes' rule)
  p(μ|X) ∝ p(X|μ)·p(μ)
         = [Πᵢ₌₁ⁿ p(xᵢ|μ)]·p(μ)
         ∝ exp(-Σᵢ(xᵢ-μ)²/(2σ²))·exp(-(μ-μ₀)²/(2σ₀²))

Step 2: Expand exponents
  -Σᵢ(xᵢ-μ)²/(2σ²) - (μ-μ₀)²/(2σ₀²)
  = -[n/σ²·Σᵢ(xᵢ²-2xᵢμ+μ²) + (μ²-2μμ₀+μ₀²)/σ₀²]/(2)
  = -[(n/σ²+1/σ₀²)μ² - 2(nX̄/σ²+μ₀/σ₀²)μ + ...]/2

Step 3: Complete the square
  a·μ² - 2b·μ + c = a(μ - b/a)² + (c - b²/a)
  
  where:
    a = n/σ² + 1/σ₀² = 1/σₙ²  (precision of posterior)
    b = nX̄/σ² + μ₀/σ₀²
  
  Therefore:
    σₙ² = 1/(n/σ² + 1/σ₀²)
    μₙ = b/a = σₙ²·(nX̄/σ² + μ₀/σ₀²) ✓  QED
```

**Intuition:**

```
Posterior mean μₙ is precision-weighted average:
  
  μₙ = (τ₀·μ₀ + τ·nX̄)/(τ₀ + τ·n)
  
  • Prior precision τ₀ = 1/σ₀²: How confident in prior
  • Data precision τ·n = n/σ²: How confident in data
  
  More data → μₙ approaches X̄ (data dominates)
  Strong prior → μₙ stays near μ₀ (prior dominates)
  
Posterior variance decreases:
  σₙ² = 1/(1/σ₀² + n/σ²) < min(σ₀², σ²/n)
  
  More data or stronger prior → less uncertainty
```

---

### 5. Gaussian Identities for ML

**Marginal and Conditional Gaussians:**

```
Joint distribution:
  [x]     [[Σ_xx  Σ_xy]]
  [y] ~ N([μ_x], [Σ_yx  Σ_yy])
            [μ_y]

Marginal:
  x ~ N(μ_x, Σ_xx)  (just drop y!)

Conditional:
  x|y ~ N(μ_x + Σ_xy Σ_yy⁻¹(y-μ_y), Σ_xx - Σ_xy Σ_yy⁻¹ Σ_yx)
  
  = N(μ_{x|y}, Σ_{x|y})

where:
  μ_{x|y} = μ_x + Σ_xy Σ_yy⁻¹(y - μ_y)  (regression formula!)
  Σ_{x|y} = Σ_xx - Σ_xy Σ_yy⁻¹ Σ_yx     (Schur complement)
```

**Linear Transformation:**

```
If X ~ N(μ, Σ), then:
  Y = AX + b ~ N(Aμ + b, AΣAᵀ)

Proof:
  Characteristic function:
    φ_Y(t) = E[exp(itᵀY)]
           = E[exp(itᵀ(AX+b))]
           = exp(itᵀb)·E[exp(i(Aᵀt)ᵀX)]
           = exp(itᵀb)·φ_X(Aᵀt)
           = exp(itᵀb)·exp(i(Aᵀt)ᵀμ - (1/2)(Aᵀt)ᵀΣ(Aᵀt))
           = exp(itᵀ(Aμ+b) - (1/2)tᵀ(AΣAᵀ)t)
           
  This is the CF of N(Aμ+b, AΣAᵀ) ✓

Applications:
  • PCA: Project X onto principal directions
  • Whitening: Y = Σ^{-1/2}X ~ N(0, I)
  • Neural network layer: Y = WX + b (if X Gaussian)
```

**Sum of Gaussians:**

```
If X ~ N(μ_x, Σ_x) and Y ~ N(μ_y, Σ_y) are independent:
  X + Y ~ N(μ_x + μ_y, Σ_x + Σ_y)

Variances add! (independent case only)
```

**Product of Gaussian PDFs (Unnormalized):**

```
p₁(x) = N(x; μ₁, Σ₁)
p₂(x) = N(x; μ₂, Σ₂)

p₁(x)·p₂(x) ∝ N(x; μ, Σ)

where:
  Σ = (Σ₁⁻¹ + Σ₂⁻¹)⁻¹
  μ = Σ(Σ₁⁻¹μ₁ + Σ₂⁻¹μ₂)

Used in: Kalman filtering, sensor fusion, Bayesian inference
```

---

### 6. Kullback-Leibler Divergence for Gaussians

**Univariate:**

```
D_KL(N(μ₁,σ₁²) || N(μ₂,σ₂²)) = log(σ₂/σ₁) + (σ₁² + (μ₁-μ₂)²)/(2σ₂²) - 1/2

Special case (KL to standard normal):
D_KL(N(μ,σ²) || N(0,1)) = (μ² + σ² - 1 - log(σ²))/2

This is the VAE regularization term!
```

**Multivariate:**

```
D_KL(N(μ₁,Σ₁) || N(μ₂,Σ₂)) = 
  (1/2)[tr(Σ₂⁻¹Σ₁) + (μ₂-μ₁)ᵀΣ₂⁻¹(μ₂-μ₁) - d + log(|Σ₂|/|Σ₁|)]
```

---

### 7. Sampling from Multivariate Gaussian

**Box-Muller Transform (Univariate):**

```
Generate X ~ N(0,1) from uniform U₁, U₂ ~ Unif(0,1):

  X = √(-2ln(U₁))·cos(2πU₂)
  Y = √(-2ln(U₁))·sin(2πU₂)

X, Y are independent N(0,1)

Proof uses transformation of variables:
  Polar coordinates + exponential distribution
```

**Cholesky Decomposition (Multivariate):**

```
To sample X ~ N(μ, Σ):

  1. Compute Cholesky: Σ = LLᵀ (L lower triangular)
  2. Sample Z ~ N(0, I) (d independent standard normals)
  3. Set X = μ + LZ
  
Then X ~ N(μ, Σ) because:
  E[X] = μ + L·E[Z] = μ
  Cov[X] = L·Cov[Z]·Lᵀ = L·I·Lᵀ = LLᵀ = Σ ✓

Cost: O(d³) for Cholesky, O(d²) per sample
```

---

### 8. Gaussian Processes: Function-Space View

**Gaussian Process as Infinite-Dimensional Gaussian:**

```
A Gaussian Process is a collection of random variables, 
any finite subset of which has a joint Gaussian distribution.

GP(m, k): Distribution over functions
  • m(x): Mean function
  • k(x,x'): Covariance function (kernel)

For any finite set X = {x₁,...,xₙ}:
  f(X) = [f(x₁),...,f(xₙ)]ᵀ ~ N(m(X), K(X,X))
  
  where K_ij = k(xᵢ, xⱼ)
```

**Posterior Inference (GP Regression):**

```
Prior: f ~ GP(0, k)
Likelihood: y|f ~ N(f, σ²I)

Observed: (X, y)
Query: f* at X*

Posterior:
  f*|X,y,X* ~ N(μ*, Σ*)
  
  μ* = K(X*,X)[K(X,X) + σ²I]⁻¹y
  Σ* = K(X*,X*) - K(X*,X)[K(X,X) + σ²I]⁻¹K(X,X*)
  
This is Gaussian conditioning!
```

---

### 9. Numerical Stability for Gaussians in Code

**Computing Log-Likelihood:**

```
Bad (numerically unstable):
  L = -0.5·(x-μ)ᵀΣ⁻¹(x-μ) - 0.5·log|Σ| - (d/2)·log(2π)
  
  Problem: Computing Σ⁻¹ and |Σ| prone to overflow/underflow

Good (via Cholesky):
  1. Compute L = cholesky(Σ)  (Σ = LLᵀ)
  2. Solve Lα = (x-μ)  via forward substitution
  3. Compute: log p = -0.5·||α||² - Σᵢlog(Lᵢᵢ) - (d/2)·log(2π)
  
  where log|Σ| = 2·Σᵢlog(Lᵢᵢ) (diagonal of L)
  
  Benefits:
    • No matrix inversion
    • Numerically stable
    • O(d³) Cholesky + O(d²) per evaluation
```

**PyTorch Implementation:**

```python
import torch
from torch.distributions import MultivariateNormal

# Direct way
mu = torch.zeros(d)
Sigma = torch.eye(d)
dist = MultivariateNormal(mu, Sigma)

# More stable: Provide Cholesky factor directly
L = torch.linalg.cholesky(Sigma)
dist = MultivariateNormal(mu, scale_tril=L)

# Sample
samples = dist.sample((1000,))

# Log probability (numerically stable)
log_prob = dist.log_prob(samples)
```

---

### 10. Gaussian in Deep Learning

**Gaussian Initialization (Xavier/He):**

```
Xavier (Glorot): W ~ N(0, 2/(n_in + n_out))
He: W ~ N(0, 2/n_in)

Reason: Preserve variance through layers
  If X ~ N(0, σ_x²) and W ~ N(0, σ_w²):
    Y = WX has Var[Y] = n·σ_w²·σ_x²
    
  To keep Var[Y] = Var[X]:
    Set σ_w² = 1/n
```

**Gaussian Noise for Robustness:**

```
Training with noise:
  X_noisy = X + ε, where ε ~ N(0, σ²I)
  
  Effect: Implicit regularization
  Equivalent to: L₂ penalty on gradients
```

**Variational Autoencoders:**

```
Encoder: q(z|x) = N(μ(x), σ²(x)I)
Decoder: p(x|z) = N(f(z), σ²I)
Prior: p(z) = N(0, I)

Reparameterization trick:
  z = μ(x) + σ(x)·ε, where ε ~ N(0,I)
  
  Allows backprop through sampling!
```

**Gaussian Mixture Models:**

```
p(x) = Σₖ₌₁ᴷ πₖ·N(x; μₖ, Σₖ)

where:
  πₖ = mixing coefficient (Σₖπₖ = 1)
  
EM algorithm for learning parameters
Used in: Clustering, density estimation
```

---

### 11. Gaussian vs Other Distributions

```
                    Gaussian          Laplace           Student-t
Mean:               μ                 μ                 μ
Variance:           σ²                2b²               ν/(ν-2)σ²
Tail:               Light (e^{-x²})   Medium (e^{-|x|}) Heavy (x^{-ν})
ML use:             Most common       L1 regularization Robust regression
Outliers:           Sensitive         More robust       Very robust
```

---

⬅️ [Back: Bernoulli](./bernoulli.md)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>
