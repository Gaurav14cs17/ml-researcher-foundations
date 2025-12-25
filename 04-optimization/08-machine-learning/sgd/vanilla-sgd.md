<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=FF6B6B&height=120&section=header&text=Vanilla%20SGD&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
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

**Proof Sketch:**

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

**Batch Size vs Convergence:**

```
Let C = computational budget (total gradient evaluations)

Small batch (b=32):
  • Iterations: T = C/32
  • Convergence: O(1/√(C/32)) = O(√32/√C)
  • Progress per time: More updates

Large batch (b=1024):
  • Iterations: T = C/1024
  • Convergence: O(1/√(C/1024)) = O(√1024/√C)
  • Progress per time: Fewer but more accurate updates

Sweet spot: b ∈ [32, 512] for most problems
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

3. Exponential:
   α(t) = α₀ · e^{-λt}
   
   Smooth decay

4. Polynomial (1/t):
   α(t) = α₀/(1 + t/T)^p
   
   p=1 satisfies Robbins-Monro
   
5. Cosine annealing:
   α(t) = α_min + (α_max - α_min)(1 + cos(πt/T))/2
   
   Popular for transformers

6. Warmup + decay:
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

**Flat Minima vs Sharp Minima:**

```
Sharp minimum:
  • High curvature (large eigenvalues of Hessian)
  • Small perturbations → large loss increase
  • Poor generalization

Flat minimum:
  • Low curvature (small eigenvalues)
  • Robust to perturbations
  • Good generalization

SGD naturally finds flat minima due to noise!

Mathematical connection:
  Generalization gap ∝ (α·B)/(μ)
  
  where:
    α = learning rate
    B = gradient noise covariance
    μ = sharpness (Hessian eigenvalues)
```

---

### 8. Practical Guidelines from Theory

**Batch Size Selection:**

```
Linear scaling rule (Goyal et al., 2017):
  When increasing batch size b → k·b:
    Scale learning rate α → k·α
  
  Intuition: Larger batch = more accurate gradient
           → Can take larger steps

Caveat: Works up to b ≈ 512-1024, then breaks down

Example (ImageNet):
  b = 256, α = 0.1  (baseline)
  b = 1024, α = 0.4  (4× batch, 4× LR)
```

**Warmup for Large Batch:**

```
Problem: Large initial LR can destabilize training

Solution: Gradual warmup
  For first T_warmup steps:
    α(t) = α_target · (t/T_warmup)
  
  Then use normal schedule

Why it works:
  • Early training: High gradient variance
  • Warmup: Gives time to stabilize
  • After warmup: Can use full LR
```

**Gradient Clipping:**

```
When ||g̃ₜ|| > threshold:
  g̃ₜ ← g̃ₜ · (threshold/||g̃ₜ||)

Effect: Bounds ||g̃ₜ|| ≤ threshold

Useful when:
  • Training RNNs (exploding gradients)
  • Training GANs (unstable dynamics)
  • Very deep networks

Theory: Still converges if clip threshold reasonable
```

---

### 9. SGD Variants: Mathematical Comparison

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

Adam (combines both):
  Covered in adam/ folder
```

---

## 📚 Resources

| Type | Title | Link |
|------|-------|------|
| 📄 Paper | Bottou - Large-Scale ML | [Paper](https://leon.bottou.org/publications/pdf/compstat-2010.pdf) |
| 📖 Book | Deep Learning Ch.8 | [Book](https://www.deeplearningbook.org/) |
| 🇨🇳 知乎 | SGD原理详解 | [知乎](https://zhuanlan.zhihu.com/p/22252270) |
| 🇨🇳 CSDN | 随机梯度下降 | [CSDN](https://blog.csdn.net/google19890102/article/details/69942970) |

---

---

⬅️ [Back: Momentum](./momentum.md)

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=FF6B6B&height=80&section=footer" width="100%"/>
</p>
