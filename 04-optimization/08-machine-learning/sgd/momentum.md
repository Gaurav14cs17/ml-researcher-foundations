<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=FF6B6B&height=120&section=header&text=SGD%20with%20Momentum&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-04-FF6B6B?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

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

## 🌍 Real-World Applications

### 1. **ResNet Training (ImageNet)**
```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1,
    momentum=0.9,     # <-- Key!
    weight_decay=1e-4
)
```
- Paper: "Deep Residual Learning" - Kaiming He
- Result: Won ImageNet 2015

### 2. **Diffusion Model Training**
```
# Training diffusion models (like Stable Diffusion)
# Uses momentum-based optimizers

Score matching loss:
L = E_t,x,ε[||ε - ε_θ(x_t, t)||²]

Momentum helps smooth noisy gradients from:
• Random timestep t
• Random noise ε  
• Random data x
```

### 3. **BERT Pre-training**
```
# Original BERT used Adam (momentum + adaptive)
# But SGD+Momentum works too with careful tuning
Batch: 256
LR: 1e-4 with warmup
Momentum: 0.9
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

## 💻 Code Example

```python
# PyTorch - SGD with Momentum
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9
)

# Manual implementation
v = 0
for epoch in range(100):
    g = compute_gradient(theta)
    v = beta * v + g           # Accumulate
    theta = theta - lr * v     # Update
```

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

**Practical Note:**

```
PyTorch's SGD with momentum uses standard Polyak momentum, not Nesterov
  
To get Nesterov:
  optimizer = torch.optim.SGD(..., momentum=0.9, nesterov=True)

Difference usually small in practice, but Nesterov often slightly better
```

---

### 4. Heavy Ball vs Nesterov: Detailed Comparison

**Heavy Ball (Polyak) Formulation:**

```
Standard form:
  vₜ₊₁ = β·vₜ + gₜ
  θₜ₊₁ = θₜ - α·vₜ₊₁

Equivalent form (after algebraic manipulation):
  θₜ₊₁ = θₜ - α·gₜ + β(θₜ - θₜ₋₁)
  
  Interpretation: Current gradient + momentum from previous step
```

**Nesterov Formulation:**

```
Standard form:
  vₜ₊₁ = β·vₜ + ∇f(θₜ - α·β·vₜ)
  θₜ₊₁ = θₜ - α·vₜ₊₁

Sutskever's equivalent form (easier to implement):
  θₜ₊₁ = θₜ - α·gₜ + β(θₜ - θₜ₋₁) + β·α·(gₜ - gₜ₋₁)
  
  Extra term: β·α·(gₜ - gₜ₋₁) (gradient correction!)
```

---

### 5. Momentum for Ill-Conditioned Problems

**Ravine Problem:**

```
Consider f(x,y) = (1/2)(x²/ε² + y²) where ε << 1
  
  • Steep in x-direction (curvature 1/ε²)
  • Shallow in y-direction (curvature 1)
  • Condition number: κ = 1/ε²

Gradient descent oscillates:
  Step in x: Too aggressive (steep)
  Step in y: Too conservative (shallow)
  
  ∇f = [x/ε², y]
  
  Update: [x,y] → [x - α·x/ε², y - α·y]
  
  Oscillation when α·1/ε² > 2 → α < 2ε²
  But then progress in y is O(ε²) - very slow!
```

**Momentum Saves the Day:**

```
With momentum:
  • x-oscillations damped by averaging
  • y-momentum builds up consistently
  • Net effect: Smooth diagonal path

Convergence improves from O(1/ε²) to O(1/ε) iterations!
```

---

### 6. Stochastic Momentum: Variance Reduction

**SGD Momentum with Noise:**

```
Stochastic gradient: g̃ₜ with variance σG²

Variance of velocity:
  Var[vₜ] = σG²/(1-β²)
  
  β = 0 (no momentum): Var[v] = σG²
  β = 0.9: Var[v] = σG²/(1-0.81) = 5.26·σG²
  β = 0.99: Var[v] = σG²/(1-0.9801) ≈ 50·σG²

Wait, momentum increases variance!?

YES, but update has different variance:
  θₜ₊₁ = θₜ - α·vₜ
  
  Total displacement over T steps:
    Δθ = α·Σₜ₌₀ᵀ⁻¹ vₜ
  
  Variance of displacement:
    Var[Δθ] ∝ σG²·T/(1-β)  (less than no momentum!)
  
  Reason: Consecutive velocities correlated → cancellation
```

---

### 7. Momentum in Non-Convex Optimization

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

### 8. Practical Hyperparameter Selection

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

### 9. Momentum Variations in Deep Learning

**PyTorch SGD with Momentum:**

```python
# Standard implementation
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    dampening=0,  # Reduces momentum (usually 0)
    nesterov=False
)

# Update rule:
# v = momentum * v + g
# θ = θ - lr * v
```

**Weight Decay with Momentum:**

```
Two implementations:

1. L2 regularization (standard):
   gₜ = ∇L(θₜ) + λ·θₜ  (add weight decay to gradient)
   vₜ₊₁ = β·vₜ + gₜ
   θₜ₊₁ = θₜ - α·vₜ₊₁

2. Decoupled weight decay (better):
   vₜ₊₁ = β·vₜ + ∇L(θₜ)
   θₜ₊₁ = (1 - α·λ)·θₜ - α·vₜ₊₁
   
   Effect: Decouple weight decay from learning rate
   Used in: AdamW, SGDW

Difference matters for momentum!
  Standard: Weight decay affects accumulated velocity
  Decoupled: Weight decay independent of velocity
```

---

### 10. Convergence Comparison: Numbers

**Example: Logistic Regression on MNIST**

```
Setup:
  • Loss: Logistic loss (smooth, strongly convex)
  • Dimension: 784 × 10 = 7840 parameters
  • Condition number: κ ≈ 1000 (estimated)

Theoretical iterations to ε = 10⁻⁶:

GD (no momentum):
  T ≈ κ·log(1/ε) ≈ 1000·14 = 14,000 iterations

GD with optimal momentum (β ≈ 0.97):
  T ≈ √κ·log(1/ε) ≈ 32·14 = 448 iterations
  
  Speedup: 31× faster!

SGD (mini-batch 32):
  T ≈ (1/ε)·√(N/b) ≈ 10⁶·√(60000/32) ≈ 4.3×10⁷ samples
  Epochs: 720

SGD with momentum:
  T ≈ 10⁷ samples  (empirical)
  Epochs: 170
  
  Speedup: 4× faster
```

---

### 11. Common Pitfalls and Solutions

**1. Momentum Overshoot:**
```
Problem: Too much momentum → oscillation
  
Symptoms:
  • Loss goes up after going down
  • Training unstable
  
Solution:
  • Reduce β (0.9 → 0.8)
  • Reduce learning rate
  • Add gradient clipping
```

**2. Initial Velocity:**
```
Standard: v₀ = 0

Alternative: "Warm start"
  v₀ = -∇L(θ₀)  (initialize with first gradient)
  
  Benefit: Faster initial progress
  Risk: Less stable
```

**3. Momentum with LR Decay:**
```
Question: Should we decay momentum too?

Answer: Usually no!
  • LR decay: α_t = α₀/√t
  • Momentum: Keep β = 0.9 constant
  
  Reason: Momentum provides smoothing, independent of step size
```

---

### 12. Code: Manual Implementation

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

# Usage
model_params = [np.random.randn(10, 5), np.random.randn(5)]
optimizer = MomentumSGD(model_params, lr=0.01, momentum=0.9)

for epoch in range(100):
    grads = compute_gradients(model_params)
    optimizer.step(grads)
```

---

## 📚 Resources

| Type | Title | Link |
|------|-------|------|
| 📄 Paper | Polyak Momentum (1964) | Classic |
| 📄 Paper | Sutskever - Momentum Importance | [Paper](https://www.cs.toronto.edu/~hinton/absps/momentum.pdf) |
| 🎥 Video | Momentum Visualized | [YouTube](https://www.youtube.com/watch?v=k8fTYJPd3_I) |
| 🇨🇳 知乎 | 动量法详解 | [知乎](https://zhuanlan.zhihu.com/p/21486826) |

---

---

➡️ [Next: Vanilla Sgd](./vanilla-sgd.md)

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=FF6B6B&height=80&section=footer" width="100%"/>
</p>
