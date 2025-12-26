<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=FF6B6B&height=120&section=header&text=Stochastic%20Gradient%20Descent&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-04-FF6B6B?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 📐 Mathematical Foundations

### Vanilla SGD
```
θₜ₊₁ = θₜ - η ∇L_B(θₜ)

Where B is a mini-batch (random subset)
E[∇L_B] = ∇L (unbiased estimator)
```

### SGD with Momentum
```
vₜ₊₁ = β vₜ + ∇L_B(θₜ)
θₜ₊₁ = θₜ - η vₜ₊₁

β typically 0.9 (exponential moving average of gradients)
```

### Nesterov Accelerated Gradient
```
θ_lookahead = θₜ - β vₜ
vₜ₊₁ = β vₜ + ∇L_B(θ_lookahead)
θₜ₊₁ = θₜ - η vₜ₊₁

Evaluates gradient at "lookahead" position
```

### Convergence Rate
```
For convex functions with σ² gradient variance:
E[f(θₜ) - f*] ≤ O(1/√t) + O(σ²/η)

Learning rate schedule:
ηₜ = η₀ / √t or step decay
```

---

## 🌍 Where SGD is Used

| Application | How | Paper/Reference |
|-------------|-----|-----------------|
| **GPT/LLM Training** | Mini-batch SGD on billions of tokens | [GPT-3 Paper](https://arxiv.org/abs/2005.14165) |
| **Diffusion Models** | Denoising score matching with SGD | [DDPM](https://arxiv.org/abs/2006.11239) |
| **ResNet/ImageNet** | SGD with momentum, batch norm | [ResNet](https://arxiv.org/abs/1512.03385) |
| **Recommendation Systems** | Matrix factorization with SGD | Netflix Prize |
| **Reinforcement Learning** | Policy gradient (a form of SGD) | [PPO](https://arxiv.org/abs/1707.06347) |

---

## 🔗 Dependency Graph

```
foundations/linear-algebra
         |
         v
    basic-methods/gradient-descent
         |
         v
+--------+--------+
|   SGD Variants  |
+-----------------+
| • vanilla-sgd   |
| • momentum      |
| • nesterov      |
| • learning-rates|
+--------+--------+
         |
         v
    machine-learning/adam
```

---

# Part 1: Vanilla SGD

## 📐 Formula

```
+-------------------------------------------------+
|                                                 |
|   θ_{t+1} = θ_t - α · ∇L(θ_t; x_i, y_i)        |
|                                                 |
|   where:                                        |
|   • θ = parameters                              |
|   • α = learning rate                           |
|   • (x_i, y_i) = random sample from dataset    |
|                                                 |
+-------------------------------------------------+
```

---

## 🎯 Key Insight

| Full Batch GD | Mini-Batch SGD |
|---------------|----------------|
| ∇L = (1/N) Σ ∇L_i | ∇L ≈ (1/B) Σ ∇L_i |
| N = all data | B = batch size (32-512) |
| Exact gradient | Noisy estimate |
| Slow per step | Fast per step |
| Smooth path | Noisy path |

---

## 🌍 Real-World Applications

### 1. **Language Model Training (GPT, BERT)**
```
Dataset: Trillions of tokens
Batch size: 512 - 4096
Why SGD: Impossible to fit full dataset in memory
```

### 2. **Image Classification (ResNet on ImageNet)**
```
Dataset: 1.2M images
Batch size: 256
Why SGD: Memory efficient, good generalization
Paper: "Deep Residual Learning" (2015)
```

### 3. **Diffusion Models (Stable Diffusion)**
```
Training: Predict noise at each timestep
Loss: ||ε - ε_θ(x_t, t)||²
SGD variant: Adam (covered later)
Paper: "Denoising Diffusion Probabilistic Models"
```

---

## ⚠️ Noise is a Feature, Not a Bug

```
Why noise helps:

1. Escapes local minima
   -----•-----     With noise:    -----•→→→-----
        ╲_╱   -------------->          \_→•_/
   Stuck here!                    Escapes!

2. Finds flatter minima (better generalization)
   
   Sharp minimum:     Flat minimum:
       |╲             --------
       | ╲            ╲      ╱
       |  •           ╲•----╱
   Overfits!          Generalizes!
```

---

## 💻 PyTorch Example

```python
import torch
import torch.nn as nn

# Model and data
model = nn.Linear(784, 10)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(100):
    for x_batch, y_batch in dataloader:
        # Forward pass
        output = model(x_batch)
        loss = criterion(output, y_batch)
        
        # Backward pass (compute ∇L)
        optimizer.zero_grad()
        loss.backward()
        
        # SGD update: θ = θ - α∇L
        optimizer.step()
```

---

## 📊 Convergence Analysis

| Assumption | Rate | Notes |
|------------|------|-------|
| Convex, L-smooth | O(1/√T) | Sublinear |
| Strongly convex | O(1/T) | Linear |
| Non-convex | O(1/√T) to stationary | Finds local min |

---

## 📐 DETAILED MATHEMATICAL THEORY

### 1. SGD Algorithm: From Full Batch to Stochastic

**Full Batch Gradient Descent:**
```
Loss: L(θ) = (1/N) Σᵢ₌₁ⁿ ℓ(θ; xᵢ, yᵢ)

Update:
  θₜ₊₁ = θₜ - α·∇L(θₜ)
       = θₜ - α·(1/N) Σᵢ₌₁ⁿ ∇ℓ(θₜ; xᵢ, yᵢ)

Cost per iteration: O(N) gradient computations
```

**Stochastic Gradient Descent (SGD):**
```
Sample: Pick i uniformly at random from {1,...,N}

Stochastic gradient:
  g̃ₜ = ∇ℓ(θₜ; xᵢ, yᵢ)  (single sample!)

Update:
  θₜ₊₁ = θₜ - α·g̃ₜ

Cost per iteration: O(1) gradient computation

Key property: E[g̃ₜ | θₜ] = ∇L(θₜ)  (unbiased!)
```

**Mini-Batch SGD (Practical):**
```
Sample: Pick batch B ⊂ {1,...,N} of size b

Mini-batch gradient:
  g̃ₜ = (1/b) Σᵢ∈B ∇ℓ(θₜ; xᵢ, yᵢ)

Update:
  θₜ₊₁ = θₜ - α·g̃ₜ

Cost per iteration: O(b) gradient computations

Variance reduction: Var[g̃ₜ] ∝ 1/b
```

---

### 2. Convergence Theory: Convex Case

**Theorem 1: Sublinear Convergence (Convex + Smooth)**

```
Assumptions:
  1. L is convex: L(y) ≥ L(x) + ∇L(x)ᵀ(y-x)
  2. L is L-smooth: ||∇L(x) - ∇L(y)|| ≤ L||x-y||
  3. Bounded gradients: E[||g̃ₜ||²] ≤ G²
  4. Unbiased: E[g̃ₜ] = ∇L(θₜ)
  5. Constant step size: αₜ = α = 1/(2L)

Then:
  E[L(θ̄_T)] - L(θ*) ≤ (2L||θ₀ - θ*||² + αG²T)/(2T)
                     = O(1/√T)  when α = O(1/√T)

where θ̄_T = (1/T) Σₜ₌₁ᵀ θₜ (average iterate)
```

**Proof:**

```
Step 1: Descent lemma for stochastic update
  E[||θₜ₊₁ - θ*||²]
    = E[||θₜ - α·g̃ₜ - θ*||²]
    = E[||θₜ - θ*||²] - 2α·E[g̃ₜᵀ(θₜ - θ*)] + α²E[||g̃ₜ||²]
    
  By unbiasedness: E[g̃ₜᵀ(θₜ - θ*)] = ∇L(θₜ)ᵀ(θₜ - θ*)
  
  By convexity: ∇L(θₜ)ᵀ(θₜ - θ*) ≥ L(θₜ) - L(θ*)
  
  Therefore:
  E[||θₜ₊₁ - θ*||²] ≤ ||θₜ - θ*||² - 2α(L(θₜ) - L(θ*)) + α²G²

Step 2: Rearrange
  2α(L(θₜ) - L(θ*)) ≤ ||θₜ - θ*||² - E[||θₜ₊₁ - θ*||²] + α²G²

Step 3: Sum from t=0 to T-1
  2α Σₜ(L(θₜ) - L(θ*)) ≤ ||θ₀ - θ*||² + Tα²G²

Step 4: Average and apply Jensen's inequality
  By convexity: L(θ̄_T) ≤ (1/T) Σₜ L(θₜ)
  
  Therefore:
  2αT(L(θ̄_T) - L(θ*)) ≤ ||θ₀ - θ*||² + Tα²G²
  
  L(θ̄_T) - L(θ*) ≤ (||θ₀ - θ*||²)/(2αT) + (αG²)/2

Step 5: Optimize step size
  Set α = ||θ₀ - θ*||/(G√T) to balance terms
  
  L(θ̄_T) - L(θ*) ≤ (G||θ₀ - θ*||)/√T = O(1/√T) ✓  QED
```

**Key Insight:**
```
SGD converges O(1/√T) vs GD's O(1/T)
  
BUT: SGD's cost per iteration is O(1) vs GD's O(N)

Total cost to reach ε-accuracy:
  GD:  O(N/ε) gradient evaluations
  SGD: O(1/ε²) gradient evaluations

SGD wins when N > 1/ε (almost always in ML!)
```

---

### 3. Strongly Convex Case: Faster Convergence

**Theorem 2: Linear Convergence (Strongly Convex)**

```
Additional assumption:
  L is μ-strongly convex: L(y) ≥ L(x) + ∇L(x)ᵀ(y-x) + (μ/2)||y-x||²

With decreasing step size αₜ = α₀/(1 + μα₀t):

  E[L(θₜ) - L(θ*)] ≤ C/(μα₀t) = O(1/t)

Much faster than O(1/√t)!
```

---

### 4. Non-Convex Case: Stationary Points

**Theorem 3: Finding Stationary Points**

```
For non-convex L (no convexity assumption!):

With constant step size α = 1/(2L):
  (1/T) Σₜ₌₀ᵀ⁻¹ E[||∇L(θₜ)||²] ≤ (2L(L(θ₀) - L*))/T + LσG²/T

where σG² = E[||g̃ₜ - ∇L(θₜ)||²] (gradient variance)

To find ε-stationary point (||∇L|| ≤ ε):
  T = O(1/ε²) iterations
```

**Why This Matters for Deep Learning:**

```
Neural networks are non-convex, yet SGD works!

Empirical observations:
  1. Local minima are nearly as good as global minima
  2. High dimensionality → saddle points, not local mins
  3. SGD noise helps escape saddle points
  4. Wide networks → loss landscape becomes "nicer"
```

---

### 5. Variance Reduction: The Mini-Batch Effect

**Gradient Variance:**

```
Single sample (b=1):
  Var[g̃] = E[||g̃ - ∇L||²] = σG²

Mini-batch (size b):
  Var[ḡ] = E[||(1/b)Σᵢ₌₁ᵇ g̃ᵢ - ∇L||²]
         = σG²/b  (variance decreases!)

Trade-off:
  • Larger b → Less variance, smoother convergence
  • Smaller b → More noise, better exploration
  • Optimal b depends on problem (typically 32-512)
```

---

### 6. Learning Rate Schedules: Theory

**Robbins-Monro Conditions (Theoretical):**

```
For convergence, need:
  1. Σₜ αₜ = ∞         (go far enough)
  2. Σₜ αₜ² < ∞        (noise decreases)

Examples:
  • αₜ = α₀/t          ✓ (satisfies both)
  • αₜ = α₀/√t         ✓
  • αₜ = constant      ✗ (violates condition 2)
```

**Practical Schedules:**

```
1. Constant:
   α(t) = α₀
   
   Pros: Simple, works if α₀ well-tuned
   Cons: Never fully converges

2. Step decay:
   α(t) = α₀ · γ^⌊t/s⌋
   
   Example: Divide by 10 every 30 epochs
   Used in: ResNet ImageNet training

3. Cosine annealing:
   α(t) = α_min + (α_max - α_min)(1 + cos(πt/T))/2
   
   Popular for transformers

4. Warmup + decay:
   α(t) = α_max · min(t/t_warmup, (t/t_warmup)^{-0.5})
   
   Critical for transformer training
```

---

### 7. Noise and Generalization: The Implicit Bias

**Why SGD Generalizes Better Than GD:**

```
Theory (Simplified):
  SGD introduces noise → implicit regularization
  
  SGD tends to find flatter minima (better generalization)

Mathematical intuition:
  SDE approximation of SGD:
    dθ_t = -∇L(θ_t)dt + √(2αB)·dW_t
    
  where:
    B = covariance of gradient noise
    dW_t = Brownian motion
  
  Effect: SGD explores around minimum
  → Finds wider valleys (flatter minima)
  → Better generalization!
```

---

# Part 2: SGD with Momentum

## 📐 Formula

```
+-------------------------------------------------+
|                                                 |
|   v_t = β · v_{t-1} + ∇L(θ_t)                  |
|   θ_{t+1} = θ_t - α · v_t                      |
|                                                 |
|   where:                                        |
|   • v = velocity (accumulated gradient)         |
|   • β = momentum coefficient (typically 0.9)    |
|   • α = learning rate                           |
|                                                 |
+-------------------------------------------------+
```

---

## 🎯 Visual Intuition

```
Without Momentum:              With Momentum:
                               
    •                              •
    |╲                             ╲
    | ╲                             ╲
    |  ╲                             ╲
    |   •                             ╲
    |  ╱                               •
    | ╱                              (faster!)
    •
  Oscillates                   Smooth path
```

---

## 🔬 Physics Analogy

```
Ball rolling down a hill:

              Start
                •
               ╱ velocity builds up
              ╱
             •   
            ╱
           ╱
          ╱
         •  
        ╱     
       ╱
      •------•  overshoots slightly
             ╱  then settles
            •   
         Minimum

β = 0.9 means: "Remember 90% of previous velocity"
```

---

## 📊 Why Momentum Helps

| Problem | Without Momentum | With Momentum |
|---------|------------------|---------------|
| Ravines | Oscillates | Accelerates through |
| Saddle points | Stuck | Escapes faster |
| Noise | Noisy path | Smoothed |
| Convergence | Slow | 2-10x faster |

---

## 📐 DETAILED MATHEMATICAL THEORY

### 1. Momentum: Mathematical Derivation

**Standard (Polyak) Momentum:**

```
Algorithm:
  v₀ = 0
  For t = 0, 1, 2, ...:
    vₜ₊₁ = β·vₜ + gₜ
    θₜ₊₁ = θₜ - α·vₜ₊₁

where:
  vₜ = velocity (accumulated gradient)
  β ∈ [0,1) = momentum coefficient (typically 0.9)
  gₜ = ∇L(θₜ) or stochastic gradient
  α = learning rate
```

**Exponential Moving Average Interpretation:**

```
Expand vₜ recursively:
  vₜ = β·vₜ₋₁ + gₜ
     = β(β·vₜ₋₂ + gₜ₋₁) + gₜ
     = β²·vₜ₋₂ + β·gₜ₋₁ + gₜ
     = ...
     = Σᵢ₌₀^∞ β^i · gₜ₋ᵢ

Effective averaging window:
  w_eff = Σᵢ₌₀^∞ β^i = 1/(1-β)
  
  β = 0.9 → w_eff = 10 gradients
  β = 0.99 → w_eff = 100 gradients
  β = 0.999 → w_eff = 1000 gradients
```

---

### 2. Acceleration: Why Momentum Converges Faster

**Without Momentum (GD on Strongly Convex):**

```
Convergence rate for condition number κ = L/μ:
  ||θₜ - θ*|| ≤ ((κ-1)/(κ+1))^t · ||θ₀ - θ*||
  
  Rate: ρ_GD = (κ-1)/(κ+1) ≈ 1 - 2/κ  (when κ large)

Number of iterations to reach ε-accuracy:
  T_GD = O(κ log(1/ε))
```

**With Optimal Momentum:**

```
Optimal β: β* = (√κ - 1)/(√κ + 1)

Convergence rate:
  ||θₜ - θ*|| ≤ ((√κ-1)/(√κ+1))^t · ||θ₀ - θ*||
  
  Rate: ρ_Mom = (√κ-1)/(√κ+1) ≈ 1 - 2/√κ  (when κ large)

Number of iterations:
  T_Mom = O(√κ log(1/ε))

Speedup factor:
  T_GD/T_Mom = √κ
  
  κ = 100 → 10× fewer iterations!
  κ = 10000 → 100× fewer iterations!
```

**Proof Sketch (Strongly Convex Quadratics):**

```
Consider quadratic: f(θ) = (1/2)θᵀAθ - bᵀθ
  where A is positive definite with eigenvalues λ₁,...,λₙ
  
  Condition number: κ = λ_max/λ_min

Step 1: Momentum update in matrix form
  [θₜ₊₁]   [I - αA    βI] [θₜ]     [0]
  [vₜ₊₁] = [  -αA   βI] [vₜ] + [αb]

Step 2: Spectral analysis
  Convergence determined by spectral radius ρ(M) of update matrix M
  
  ρ_GD = max|(1 - α·λᵢ)| = (κ-1)/(κ+1)  (for α = 2/(λ_max+λ_min))
  
  ρ_Mom = ((√κ-1)/(√κ+1))  (with optimal β)

Step 3: General convex case
  For general smooth strongly convex f:
    Similar analysis via Polyak-Lojasiewicz condition
    Result: O(√κ) acceleration holds ✓  QED
```

---

### 3. Nesterov Accelerated Gradient (NAG)

**Nesterov Momentum (1983):**

```
Algorithm:
  v₀ = 0
  For t = 0, 1, 2, ...:
    θ_lookahead = θₜ - α·β·vₜ       (lookahead!)
    vₜ₊₁ = β·vₜ + ∇f(θ_lookahead)   (gradient at lookahead)
    θₜ₊₁ = θₜ - α·vₜ₊₁

Key difference: Evaluate gradient at lookahead position!
```

**Why Lookahead Helps:**

```
Standard momentum: "Blind momentum"
  1. Apply momentum: θ_new = θ - α·v
  2. Compute gradient at θ_new
  3. Update velocity
  
  Problem: May overshoot, then have to correct

Nesterov momentum: "Informed momentum"
  1. Look ahead: θ_look = θ - α·β·v
  2. Compute gradient at θ_look (future position!)
  3. Correct velocity based on future gradient
  
  Benefit: Better anticipation of future gradient
```

**Convergence Guarantee:**

```
For smooth convex f:
  f(θₜ) - f(θ*) ≤ (2L||θ₀ - θ*||²)/(t+1)²
  
  Rate: O(1/t²) vs O(1/t) for GD

For smooth strongly convex f:
  Same O(√κ) as standard momentum, but better constants
```

---

### 4. Momentum in Non-Convex Optimization

**Escaping Saddle Points:**

```
Saddle point: ∇f = 0, but Hessian has negative eigenvalues

Problem for GD:
  • Attracted to saddle from most directions
  • Can get stuck for many iterations
  • Escape time: exponential in dimension!

Momentum helps:
  • Kinetic energy carries through flat region
  • Escapes faster than GD
  • Escape time: polynomial in dimension

Mathematical intuition:
  At saddle with Hessian eigenvalue λ < 0:
    Momentum amplifies motion in negative curvature direction
    Escape time: O(log(1/|λ|)) vs O(1/|λ|) for GD
```

---

### 5. Practical Hyperparameter Selection

**Momentum Coefficient β:**

```
Common values:
  β = 0.9   (default, works well most cases)
  β = 0.99  (slower but steadier convergence)
  β = 0.999 (very smooth, for noisy objectives)

Heuristic rule:
  β ≈ 1 - 1/√κ  (where κ = condition number)
  
  Well-conditioned (κ ≈ 10): β = 0.68
  Ill-conditioned (κ ≈ 100): β = 0.90
  Very ill-conditioned (κ ≈ 10000): β = 0.99

In practice: Just use β = 0.9 as starting point
```

**Learning Rate with Momentum:**

```
Rule of thumb:
  α_momentum ≈ (1-β)·α_no_momentum
  
  Reason: Velocity accumulates, so effective step is larger

Example:
  No momentum: α = 0.1
  With β = 0.9: α = 0.01
  
  Effective step: (1/(1-β))·α = 10·0.01 = 0.1 (same!)
```

---

### 6. Code Implementation

```python
import numpy as np

class MomentumSGD:
    def __init__(self, params, lr=0.01, momentum=0.9, nesterov=False):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.velocities = [np.zeros_like(p) for p in params]
    
    def step(self, gradients):
        for i, (param, grad) in enumerate(zip(self.params, gradients)):
            v = self.velocities[i]
            
            if self.nesterov:
                # Nesterov momentum
                v = self.momentum * v + grad
                param -= self.lr * (grad + self.momentum * v)
            else:
                # Standard (Polyak) momentum
                v = self.momentum * v + grad
                param -= self.lr * v
            
            self.velocities[i] = v

# PyTorch equivalent
import torch

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    nesterov=True  # Nesterov momentum
)

for epoch in range(100):
    for x, y in dataloader:
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
```

---

### 7. SGD Variants Summary

```
Vanilla SGD:
  θₜ₊₁ = θₜ - α·g̃ₜ
  
  Pros: Simple, unbiased
  Cons: High variance, slow

SGD with Momentum:
  vₜ = β·vₜ₋₁ + g̃ₜ
  θₜ₊₁ = θₜ - α·vₜ
  
  Effect: Smooths gradients, accelerates
  Convergence: O(1/t) with β = 1-O(1/√κ)

Nesterov Momentum:
  vₜ = β·vₜ₋₁ + ∇L(θₜ - α·β·vₜ₋₁)
  θₜ₊₁ = θₜ - α·vₜ
  
  Better: Looks ahead before stepping

RMSprop (adaptive):
  vₜ = β·vₜ₋₁ + (1-β)·g̃ₜ²
  θₜ₊₁ = θₜ - α·g̃ₜ/√(vₜ + ε)
  
  Effect: Per-parameter learning rates
```

---

## 📚 References

| Type | Resource | Link |
|------|----------|------|
| 📄 | Robbins & Monro (1951) | Original stochastic approximation |
| 📄 | SGD Convergence | [Bottou et al., 2018](https://arxiv.org/abs/1606.04838) |
| 📄 | Polyak Momentum (1964) | Classic |
| 📄 | Sutskever - Momentum Importance | [Paper](https://www.cs.toronto.edu/~hinton/absps/momentum.pdf) |
| 📖 | Goodfellow Ch. 8 | [Deep Learning Book](https://www.deeplearningbook.org/) |
| 🎥 | Stanford CS231n | [Optimization Lecture](http://cs231n.stanford.edu/) |
| 🇨🇳 | SGD优化详解 | [知乎](https://zhuanlan.zhihu.com/p/22252270) |
| 🇨🇳 | 动量法原理 | [机器之心](https://www.jiqizhixin.com/articles/2017-07-12-8) |

---

⬅️ [Back: Adam](../01_adam/) | ⬆️ [Up: Machine Learning](../)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=FF6B6B&height=80&section=footer" width="100%"/>
</p>
