<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=FF6B6B&height=120&section=header&text=Convex%20Optimization&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-04-FF6B6B?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 📂 Subtopics

| Folder | Topic | Why Important |
|--------|-------|---------------|
| [01_elbo/](./01_elbo/) | ELBO & Variational Inference | VAE, Diffusion! |

---

## 🎯 What is Convex Optimization?

A convex optimization problem has:
1. **Convex objective** function (bowl-shaped)
2. **Convex constraint** set (no holes, connected)

```
+---------------------------------------------------------+
|                                                         |
|   CONVEX OPTIMIZATION PROBLEM:                          |
|                                                         |
|   minimize   f(x)      where f is convex               |
|   subject to gᵢ(x) ≤ 0  where gᵢ are convex            |
|              Ax = b                                     |
|                                                         |
|   KEY PROPERTY: Any local minimum = global minimum!    |
|                                                         |
+---------------------------------------------------------+
```

---

## 🎯 Why Convexity Matters

```
NON-CONVEX (typical DL):           CONVEX:

        •  local min                    
       ╱ ╲                              
      ╱   ╲     •  local min             ╲     ╱
     ╱     ╲   ╱ ╲                        ╲   ╱
    ╱       ╲-╱   ╲                        ╲ ╱
                   ╲____• global            •
                                        global = local!

• Many local minima              • Only one minimum
• SGD might get stuck            • Any method finds it
• Need good initialization       • Initialization irrelevant
```

---

## 📐 Definition of Convex Function

### Formal Definition

A function f is **CONVEX** if for all x, y and θ ∈ [0,1]:

\[f(\theta x + (1-\theta)y) \leq \theta f(x) + (1-\theta)f(y)\]

Visually: The chord is ABOVE the function

```
            • f(x)
           ╱|
          ╱ |   chord
         ╱  |
        ╱   |
       ╱    • f(y)
      ╱------------
     ╱   function
    •  below chord ✓
```

### Second-Order Conditions

```
For twice-differentiable f:

f is convex ⟺ ∇²f(x) ≽ 0 (Hessian is positive semidefinite)
f is strictly convex ⟺ ∇²f(x) ≻ 0 (Hessian is positive definite)

Example: f(x) = ½xᵀQx + bᵀx
∇f(x) = Qx + b
∇²f(x) = Q
Convex ⟺ Q ≽ 0
```

### First-Order Conditions

```
For differentiable convex f:

f(y) ≥ f(x) + ∇f(x)ᵀ(y - x)

The tangent line/plane is BELOW the function!
Used in gradient descent convergence proofs.
```

---

## 📐 Strong Convexity

```
f is μ-strongly convex if:

f(y) ≥ f(x) + ∇f(x)ᵀ(y - x) + (μ/2)‖y - x‖²

Equivalently: ∇²f(x) ≽ μI

Benefits:
• Unique global minimum
• Faster convergence: O((1-μ/L)^k) vs O(1/k)
• More stable optimization
```

---

## 📐 Lipschitz Smoothness

```
f is L-smooth if gradient is Lipschitz:

‖∇f(x) - ∇f(y)‖ ≤ L‖x - y‖

Equivalently: ∇²f(x) ≼ LI

Used for: Learning rate bounds (α ≤ 1/L)
```

---

## 📐 Convergence Rates

```
Gradient Descent: θₜ₊₁ = θₜ - α∇f(θₜ)

For L-smooth convex f with step α ≤ 1/L:
f(θₜ) - f(θ*) ≤ O(1/t)

For L-smooth μ-strongly convex:
f(θₜ) - f(θ*) ≤ O((1 - μ/L)^t)

Condition number κ = L/μ determines speed
```

---

## 🌍 Real-World Convex Problems

| Problem | Convex? | Why? |
|---------|---------|------|
| **Linear Regression** | ✅ Yes | Quadratic loss |
| **Logistic Regression** | ✅ Yes | Log-loss is convex |
| **SVM** | ✅ Yes | Hinge loss + L2 |
| **LASSO** | ✅ Yes | L1 regularized |
| **Portfolio Optimization** | ✅ Yes | Mean-variance |
| **Deep Learning** | ❌ No | Non-linear activations |
| **Matrix Factorization** | ❌ No | Product of unknowns |

---

## 📊 Convex Examples

| Function | Formula | Convex? |
|----------|---------|---------|
| Linear | f(x) = aᵀx + b | ✅ Yes (and concave!) |
| Quadratic | f(x) = xᵀQx (Q≻0) | ✅ Yes |
| Norm | f(x) = ‖x‖ | ✅ Yes |
| Log-sum-exp | f(x) = log(Σeˣⁱ) | ✅ Yes |
| Negative entropy | f(x) = Σxᵢlog(xᵢ) | ✅ Yes |
| Exponential | f(x) = eˣ | ✅ Yes |
| x³ | f(x) = x³ | ❌ No |

---

## 💻 Code: Check Convexity

```python
import numpy as np
from scipy.linalg import eigh

def is_convex_quadratic(Q):
    """Check if f(x) = x'Qx is convex by checking eigenvalues"""
    eigenvalues = eigh(Q, eigvals_only=True)
    return all(eigenvalues >= 0)

# Example: f(x,y) = x² + y² 
Q = np.array([[1, 0], [0, 1]])
print(f"Is f(x,y)=x²+y² convex? {is_convex_quadratic(Q)}")  # True

# Example: f(x,y) = x² - y²
Q = np.array([[1, 0], [0, -1]])
print(f"Is f(x,y)=x²-y² convex? {is_convex_quadratic(Q)}")  # False
```

---

# Part 2: ELBO (Evidence Lower Bound)

## 🎯 What is ELBO?

```
Problem: We want to maximize log p(x) (log-likelihood)
         But it's intractable!

Solution: Maximize a lower bound instead = ELBO

+-----------------------------------------------------+
|                                                     |
|   log p(x) = ELBO + KL(q || p)                     |
|                                                     |
|   Since KL ≥ 0, we have:                           |
|                                                     |
|   log p(x) ≥ ELBO                                  |
|                                                     |
|   Maximizing ELBO ≈ Maximizing log p(x)            |
|                                                     |
+-----------------------------------------------------+
```

---

## 📐 ELBO Formula

```
ELBO = E_q(z|x)[log p(x|z)] - KL(q(z|x) || p(z))
       ---------------------   ------------------
       Reconstruction term      Regularization term
       
       "How well can we         "Stay close to 
        reconstruct x?"          the prior p(z)"
```

---

## 📐 ELBO Decomposition (Derivation)

```
Start with log-likelihood:
  log p(x) = log ∫ p(x,z) dz

Step 1: Introduce variational distribution q(z|x)
  log p(x) = log ∫ q(z|x) [p(x,z)/q(z|x)] dz

Step 2: Apply Jensen's inequality (log is concave)
  log p(x) ≥ ∫ q(z|x) log[p(x,z)/q(z|x)] dz  = ELBO

Step 3: Expand the bound
  ELBO = E_q[log p(x,z)] - E_q[log q(z|x)]
       = E_q[log p(x|z) + log p(z)] - E_q[log q(z|x)]
       = E_q[log p(x|z)] - KL(q(z|x) || p(z))

Step 4: Exact relationship
  log p(x) = ELBO + KL(q(z|x) || p(z|x))

Since KL ≥ 0, ELBO is always a lower bound.
Equality when q(z|x) = p(z|x) (true posterior).
```

---

## 📊 Three Ways to Write ELBO

```
1. ELBO = E_q[log p(x,z)] - E_q[log q(z)]

2. ELBO = E_q[log p(x|z)] - KL(q(z|x) || p(z))

3. ELBO = log p(x) - KL(q(z|x) || p(z|x))
```

---

## 🌍 Where ELBO is Used

| Model | How ELBO is Used | Paper |
|-------|------------------|-------|
| **VAE** | Main training objective | [Kingma 2013](https://arxiv.org/abs/1312.6114) |
| **Diffusion Models** | Variational bound on likelihood | [DDPM 2020](https://arxiv.org/abs/2006.11239) |
| **Bayesian NN** | Approximate posterior | [Weight Uncertainty](https://arxiv.org/abs/1505.05424) |
| **LLM Fine-tuning** | RLHF uses variational methods | [InstructGPT](https://arxiv.org/abs/2203.02155) |
| **Normalizing Flows** | Tighter ELBO with flows | [Rezende 2015](https://arxiv.org/abs/1505.05770) |

---

## 🎨 ELBO in Diffusion Models

```
Forward Process (Add Noise):
x_0 --> x_1 --> x_2 --> ... --> x_T
 |       |       |              |
 v       v       v              v
Clean   Noisy   Noisier    Pure Noise

Reverse Process (Denoise):
x_T --> x_{T-1} --> ... --> x_1 --> x_0
 |         |               |       |
 v         v               v       v
Noise   Less Noisy      Cleaner  Clean!
```

### ELBO for Diffusion

```
+---------------------------------------------------------+
|                                                         |
|  log p(x_0) ≥ ELBO = E_q[ log p(x_T)                   |
|                          + Σ log p(x_{t-1}|x_t)        |
|                          - Σ log q(x_t|x_{t-1}) ]      |
|                                                         |
+---------------------------------------------------------+

Simplified Training Objective (DDPM):

L_simple = E_{t,x_0,ε}[ ||ε - ε_θ(x_t, t)||² ]

• t = random timestep
• ε = noise added at step t  
• ε_θ = neural network predicting noise
```

### Connection to ELBO

```
Full ELBO decomposition:

L = L_0 + L_1 + ... + L_{T-1} + L_T

where each L_t is a KL divergence:

L_t = KL( q(x_{t-1}|x_t,x_0) || p_θ(x_{t-1}|x_t) )

Key insight:
• q(x_{t-1}|x_t,x_0) is Gaussian (tractable!)
• p_θ(x_{t-1}|x_t) is also Gaussian
• KL between Gaussians has closed form
• Reduces to ||ε - ε_θ||² loss!
```

---

## 💻 Training Code (Simplified)

```python

# Diffusion Model Training with ELBO-based Loss
import torch
import torch.nn as nn

def train_step(model, x_0, noise_schedule):

    # Sample random timestep
    t = torch.randint(0, T, (batch_size,))
    
    # Sample noise
    epsilon = torch.randn_like(x_0)
    
    # Create noisy image: x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε
    alpha_bar = noise_schedule.alpha_bar[t]
    x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * epsilon
    
    # Predict noise
    epsilon_pred = model(x_t, t)
    
    # ELBO-derived loss: ||ε - ε_θ(x_t, t)||²
    loss = nn.MSELoss()(epsilon_pred, epsilon)
    
    return loss
```

---

## 📚 Resources

| Type | Title | Link |
|------|-------|------|
| 📖 | Boyd & Vandenberghe | [Free PDF](https://web.stanford.edu/~boyd/cvxbook/) |
| 🎥 | Stanford CVX101 | [YouTube](https://www.youtube.com/playlist?list=PL3940DD956CDF0622) |
| 📖 | CMU 10-725 | [Course](https://www.stat.cmu.edu/~ryantibs/convexopt/) |
| 💻 | CVXPY | [Docs](https://www.cvxpy.org/) |
| 📄 | VAE Original | [arXiv:1312.6114](https://arxiv.org/abs/1312.6114) |
| 📄 | DDPM (Diffusion) | [arXiv:2006.11239](https://arxiv.org/abs/2006.11239) |
| 🇨🇳 | 凸优化入门 | [知乎](https://zhuanlan.zhihu.com/p/25383715) |
| 🇨🇳 | ELBO推导详解 | [知乎](https://zhuanlan.zhihu.com/p/22464760) |

---

## 🔗 Where This Topic Is Used

| Topic | How Convexity Is Used |
|-------|----------------------|
| **SVM** | Dual is convex QP |
| **Logistic Regression** | Convex log-loss |
| **LASSO** | Convex L1 regularization |
| **Convergence Proofs** | Strong convexity bounds |
| **Neural Network Theory** | Local convexity analysis |
| **ELBO/VAE** | Variational lower bound |

---

⬅️ [Back: Advanced Methods](../03_advanced_methods/) | ➡️ [Next: Constrained Optimization](../05_constrained_optimization/)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=FF6B6B&height=80&section=footer" width="100%"/>
</p>
