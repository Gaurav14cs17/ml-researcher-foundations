<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=FF6B6B&height=120&section=header&text=Basic%20Optimization%20Methods&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-04-FF6B6B?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 📂 Subtopics

| Folder | Topic | Order | Used In |
|--------|-------|-------|---------|
| [01_gradient_descent/](./01_gradient_descent/) | Gradient Descent | First-order | All DL |
| [02_newton/](./02_newton/) | Newton's Method | Second-order | Fast optimization |

---

## 🎯 First-Order vs Second-Order

```
+---------------------------------------------------------+
|                                                         |
|   FIRST-ORDER (uses gradient)                           |
|   -------------------------                             |
|   x_{k+1} = x_k - α∇f(x_k)                              |
|                                                         |
|   Pros: Cheap, scalable to billions of params           |
|   Cons: Can be slow, needs LR tuning                    |
|   Used: Neural networks (SGD, Adam, etc.)               |
|                                                         |
+---------------------------------------------------------+
|                                                         |
|   SECOND-ORDER (uses Hessian)                           |
|   --------------------------                            |
|   x_{k+1} = x_k - H⁻¹∇f(x_k)                            |
|                                                         |
|   Pros: Very fast convergence                           |
|   Cons: O(n³) per step, memory O(n²)                    |
|   Used: Small problems, L-BFGS                          |
|                                                         |
+---------------------------------------------------------+
```

---

# Part 1: Gradient Descent

## 🎯 The Core Idea

```
+---------------------------------------------------------+
|                                                         |
|   Want to minimize f(x)?                                |
|                                                         |
|   1. Compute gradient ∇f(x)                            |
|      (Direction of steepest ASCENT)                    |
|                                                         |
|   2. Move in OPPOSITE direction                        |
|      x_{new} = x_{old} - α∇f(x_{old})                  |
|                                                         |
|   3. Repeat until convergence                          |
|                                                         |
|   That's it! Simple but powerful.                      |
|                                                         |
+---------------------------------------------------------+
```

---

## 📐 The Algorithm

```python

# Gradient Descent in 5 lines
x = initial_guess
for i in range(max_iterations):
    gradient = compute_gradient(f, x)
    x = x - learning_rate * gradient
    if converged(gradient):
        break
```

---

## 🎯 Visual Understanding

```
       Loss Surface (Mountain)
       
            Start here
               ●╲
              ╱  ╲
             ╱    ╲
            ╱      ●  Step 1
           ╱        ╲
          ╱          ╲
         ╱            ●  Step 2
        ╱              ╲
       ╱                ●  Step 3
      ╱                  ╲
     ╱____________________●  Minimum!
     
   Each step: Move opposite to gradient (downhill)
```

---

## 📐 DETAILED MATHEMATICAL THEORY

### 1. Gradient Descent: Complete Convergence Analysis

**Algorithm:**
```
Input: f: ℝⁿ → ℝ, starting point x₀, learning rate α
Output: x* ≈ argmin f

For k = 0, 1, 2, ...:
  1. Compute gradient: g_k = ∇f(x_k)
  2. Update: x_{k+1} = x_k - α·g_k
  3. Check convergence: ||g_k|| < ε

Return x_k
```

---

### 2. Convergence for Convex + L-Smooth Functions

**Theorem 1: Sublinear Convergence**

**Assumptions:**
1. f is convex
2. f is L-smooth: \(\|\nabla f(x) - \nabla f(y)\| \leq L\|x - y\|\)
3. Step size: α = 1/L

**Conclusion:** \(f(x_k) - f(x^*) \leq \frac{2L\|x_0 - x^*\|^2}{k}\)

**Convergence rate:** O(1/k) iterations

**Proof:**

```
Step 1: L-smoothness implies quadratic upper bound
  For any x, y:
    f(y) ≤ f(x) + ∇f(x)ᵀ(y-x) + (L/2)||y-x||²

Step 2: Apply to GD update with α = 1/L
  Let y = x_{k+1} = x_k - (1/L)∇f(x_k)
  
  f(x_{k+1}) ≤ f(x_k) + ∇f(x_k)ᵀ(x_{k+1} - x_k) + (L/2)||x_{k+1} - x_k||²
             = f(x_k) - (1/L)||∇f(x_k)||² + (L/2)·(1/L²)||∇f(x_k)||²
             = f(x_k) - (1/2L)||∇f(x_k)||²

Step 3: Descent lemma
  f(x_{k+1}) ≤ f(x_k) - (1/2L)||∇f(x_k)||²  ... (*)

Step 4: By convexity
  f(x_k) - f(x*) ≤ ∇f(x_k)ᵀ(x_k - x*)  (first-order condition)
                 ≤ ||∇f(x_k)||·||x_k - x*||  (Cauchy-Schwarz)

  Therefore: ||∇f(x_k)||² ≥ (f(x_k) - f(x*))²/||x_k - x*||²

Step 5: Substitute into (*)
  f(x_{k+1}) ≤ f(x_k) - (1/2L)·(f(x_k) - f(x*))²/||x_k - x*||²

Step 6: Track distance to optimum
  ||x_{k+1} - x*||² = ||x_k - (1/L)∇f(x_k) - x*||²
                    = ||x_k - x*||² - (2/L)∇f(x_k)ᵀ(x_k - x*) + (1/L²)||∇f(x_k)||²

  By convexity: ∇f(x_k)ᵀ(x_k - x*) ≥ f(x_k) - f(x*)
  
  Therefore:
  ||x_{k+1} - x*||² ≤ ||x_k - x*||² - (2/L)(f(x_k) - f(x*)) + (1/L²)||∇f(x_k)||²

Step 7: From Step 3, we have ||∇f(x_k)||² ≤ 2L(f(x_k) - f(x_{k+1}))
  
  Substituting:
  ||x_{k+1} - x*||² ≤ ||x_k - x*||² - (2/L)(f(x_k) - f(x*)) + (2/L)(f(x_k) - f(x_{k+1}))
                    = ||x_k - x*||² - (2/L)(f(x_{k+1}) - f(x*))

Step 8: Rearrange
  f(x_{k+1}) - f(x*) ≤ (L/2)(||x_k - x*||² - ||x_{k+1} - x*||²)

Step 9: Sum telescoping series from 0 to k-1
  Σᵢ₌₀^{k-1} (f(x_{i+1}) - f(x*)) ≤ (L/2)||x_0 - x*||²

  Since f(x_i) is decreasing:
  k·(f(x_k) - f(x*)) ≤ Σᵢ₌₀^{k-1} (f(x_{i+1}) - f(x*)) ≤ (L/2)||x_0 - x*||²

  Therefore: f(x_k) - f(x*) ≤ (L||x_0 - x*||²)/(2k) ✓  QED
```

---

### 3. Strongly Convex Case: Linear Convergence

**Theorem 2: Exponential Convergence**

**Additional assumption:** f is μ-strongly convex: \(f(y) \geq f(x) + \nabla f(x)^T(y-x) + \frac{\mu}{2}\|y-x\|^2\)

**Conclusion with α = 1/L:**
\[f(x_k) - f(x^*) \leq (1 - \mu/L)^k (f(x_0) - f(x^*))\]

**Key Quantity: Condition Number**
```
κ = L/μ  (condition number)

Convergence rate: ρ = 1 - 1/κ = (κ-1)/κ

Examples:
  κ = 2:   ρ = 0.5    → Half distance each step
  κ = 10:  ρ = 0.9    → 10% improvement per step
  κ = 100: ρ = 0.99   → 1% improvement per step (slow!)
  κ = ∞:   ρ = 1      → No improvement (ill-conditioned)

Number of iterations to reach ε-accuracy:
  k ≥ κ·log(1/ε)
```

**Proof Sketch:**

```
Step 1: Combine descent lemma with strong convexity
  From L-smoothness:
    f(x_{k+1}) ≤ f(x_k) - (1/2L)||∇f(x_k)||²
  
  From strong convexity:
    ||∇f(x_k)||² ≥ 2μ(f(x_k) - f(x*))
  
  Therefore:
    f(x_{k+1}) - f(x*) ≤ f(x_k) - f(x*) - (μ/L)(f(x_k) - f(x*))
                       = (1 - μ/L)(f(x_k) - f(x*))

Step 2: Apply recursively
  f(x_k) - f(x*) ≤ (1 - μ/L)^k (f(x_0) - f(x*)) ✓  QED
```

---

### 4. Non-Convex Case: Stationary Points

**Theorem 3: First-Order Stationary Point**

For non-convex f (L-smooth):

GD with α = 1/L satisfies:
\[\min_{0 \leq k \leq K-1} \|\nabla f(x_k)\|^2 \leq \frac{2L(f(x_0) - f_{inf})}{K}\]

where \(f_{inf} = \inf_x f(x)\)

**Interpretation:** Find ε-stationary point ($\|\nabla f\| \leq \epsilon$) in O(1/ε²) iterations

---

### 5. Learning Rate Selection: Theory vs Practice

**Practical Schedules:**

```
1. Constant (simplest):
   α_k = α_0
   
   Pros: Simple
   Cons: May oscillate near minimum

2. Step decay:
   α_k = α_0 · γ^⌊k/s⌋
   
   Example: Divide by 10 every 30 epochs
   Used in: ResNet training

3. Exponential decay:
   α_k = α_0 · e^{-λk}
   
   Smooth decay

4. 1/k schedule (theoretical):
   α_k = α_0 / k
   
   Satisfies Robbins-Monro conditions:
   • Σ_k α_k = ∞  (go far enough)
   • Σ_k α_k² < ∞  (noise decreases)

5. Cosine annealing:
   α_k = α_min + (α_max - α_min) · (1 + cos(πk/K))/2
   
   Smooth, popular for transformers

6. Warmup + decay:
   α_k = α_max · min(k/k_warmup, (k/k_warmup)^{-0.5})
   
   Used in: BERT, GPT training
```

---

### 6. Momentum: Accelerated Gradient Descent

**Standard Momentum (Polyak 1964):**

```
Algorithm:
  v₀ = 0
  For k = 0, 1, 2, ...:
    v_{k+1} = β·v_k + ∇f(x_k)
    x_{k+1} = x_k - α·v_{k+1}

where β ∈ [0,1) is momentum coefficient (typically 0.9)
```

**Convergence Improvement:**

```
Without momentum:
  k ≥ κ·log(1/ε)  iterations

With momentum (optimal β):
  k ≥ √κ·log(1/ε)  iterations

Speedup: √κ
  κ = 100 → 10× fewer iterations!
  κ = 10000 → 100× fewer iterations!
```

**Nesterov Momentum (Nesterov 1983):**

```
Algorithm:
  v₀ = 0
  For k = 0, 1, 2, ...:
    x_lookahead = x_k - α·β·v_k  (lookahead position)
    v_{k+1} = β·v_k + ∇f(x_lookahead)  (gradient at lookahead!)
    x_{k+1} = x_k - α·v_{k+1}

Key difference: Evaluate gradient at lookahead position
```

---

## 📊 Convergence Rates Summary

| Function Type | Rate | Meaning |
|--------------|------|---------|
| Strongly Convex | O(e^{-kt}) | Exponential! Fast |
| Convex, Smooth | O(1/k) | Linear |
| Non-convex | O(1/√k) | Sublinear (slow) |

---

## 💻 Implementation

### NumPy
```python
import numpy as np

def gradient_descent(f, grad_f, x0, lr=0.01, max_iter=1000, tol=1e-6):
    x = x0.copy()
    history = [x.copy()]
    
    for i in range(max_iter):
        g = grad_f(x)
        x = x - lr * g
        history.append(x.copy())
        
        if np.linalg.norm(g) < tol:
            print(f"Converged in {i+1} iterations")
            break
    
    return x, history

# Example: f(x,y) = x² + y²
def f(x):
    return x[0]**2 + x[1]**2

def grad_f(x):
    return np.array([2*x[0], 2*x[1]])

x_opt, history = gradient_descent(f, grad_f, np.array([5.0, 3.0]))
print(f"Optimal: {x_opt}")  # Close to [0, 0]
```

### PyTorch
```python
import torch

x = torch.tensor([5.0, 3.0], requires_grad=True)
optimizer = torch.optim.SGD([x], lr=0.1)

for i in range(100):
    loss = x[0]**2 + x[1]**2
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
print(f"Optimal: {x.data}")  # Close to [0, 0]
```

---

# Part 2: Newton's Method

## 🎯 The Core Idea

```
+---------------------------------------------------------+
|                                                         |
|   Gradient Descent: Linear approximation               |
|   f(x + Δx) ≈ f(x) + ∇f(x)ᵀΔx                          |
|                                                         |
|   Newton's Method: Quadratic approximation             |
|   f(x + Δx) ≈ f(x) + ∇f(x)ᵀΔx + ½ΔxᵀHΔx               |
|                                                         |
|   Why better? Captures curvature!                       |
|                                                         |
+---------------------------------------------------------+
```

---

## 📐 The Algorithm

```
Newton Step:

+---------------------------------------------------------+
|                                                         |
|   x_{k+1} = x_k - H(x_k)⁻¹ ∇f(x_k)                     |
|                                                         |
|   where:                                                |
|   • H = Hessian (matrix of second derivatives)          |
|   • ∇f = gradient (vector of first derivatives)        |
|                                                         |
|   No learning rate needed!                              |
|   (The Hessian provides natural step size)              |
|                                                         |
+---------------------------------------------------------+
```

---

## 🎯 Visual: Why It's Faster

```
Gradient Descent:              Newton's Method:
(Linear approx)                (Quadratic approx)

    ╲                               ╲
     ╲   function                    ╲   function
      ╲_____•_____                    ╲__•__╱
           ╱╲                            |
          ╱  ╲ tangent line              | perfect step!
         ╱    ╲                          ↓
                                         • minimum

Takes many small steps           Takes one big accurate step
```

---

## 📐 Mathematical Foundations

### Newton Update Rule

```
Standard Newton Step:
x_{k+1} = x_k - H(x_k)⁻¹∇f(x_k)

Where:
• H(x) = ∇²f(x) is the Hessian matrix
• ∇f(x) is the gradient
• H⁻¹∇f is the Newton direction
```

### Derivation from Taylor Expansion

```
Second-order Taylor approximation:
f(x + Δx) ≈ f(x) + ∇f(x)ᵀΔx + ½Δxᵀ H Δx

Setting derivative to zero:
∇f(x) + H·Δx = 0

Solving for optimal step:
Δx* = -H⁻¹∇f(x)
```

### Newton Decrement

```
λ² = ∇f(x)ᵀ H(x)⁻¹ ∇f(x)

Interpretation:
• λ² ≈ f(x) - f(x*)  (approximate suboptimality)
• Stopping criterion: λ² < ε
```

### Convergence Analysis

```
Near optimum (local convergence):
‖x_{k+1} - x*‖ ≤ C · ‖x_k - x*‖²

Quadratic convergence means:
• Error squares each iteration
• 10⁻² → 10⁻⁴ → 10⁻⁸ → 10⁻¹⁶
• Very fast once "close enough"

Global convergence (with damping):
x_{k+1} = x_k - α_k · H⁻¹∇f

where α_k found by line search
```

---

## 📊 Convergence Comparison

| Method | Convergence | Per-Step Cost | Memory |
|--------|-------------|---------------|--------|
| **Gradient Descent** | O(1/k) | O(n) | O(n) |
| **Newton** | O(log log(1/ε)) | O(n³) | O(n²) |
| **L-BFGS** | Superlinear | O(n) | O(mn) |

```
Newton converges QUADRATICALLY:

If error at step k is ε,
error at step k+1 is ε²!

Example:
Step 1: error = 0.1
Step 2: error = 0.01
Step 3: error = 0.0001
Step 4: error = 0.00000001

4 steps to machine precision!
```

---

## 💻 Algorithm Implementation

```python
def newton_method(f, grad_f, hess_f, x0, tol=1e-8, max_iter=100):
    x = x0
    for k in range(max_iter):
        g = grad_f(x)        # Gradient
        H = hess_f(x)        # Hessian
        
        # Newton direction: solve H·d = -g
        d = np.linalg.solve(H, -g)
        
        # Newton decrement (stopping criterion)
        lambda_sq = -g @ d
        if lambda_sq / 2 < tol:
            break
        
        # Damped Newton with backtracking
        alpha = backtracking_line_search(f, x, g, d)
        x = x + alpha * d
    
    return x

def backtracking_line_search(f, x, g, d, alpha=1.0, beta=0.5, c=0.1):
    """Armijo backtracking line search"""
    while f(x + alpha * d) > f(x) + c * alpha * (g @ d):
        alpha *= beta
    return alpha
```

---

## ⚠️ Challenges and Solutions

```
1. Hessian computation: O(n²) storage, O(n³) solve
2. Non-positive definite H: Direction may not be descent
3. Far from optimum: May diverge without damping
4. Saddle points: H singular or indefinite

Solutions:
• Regularization: H + λI (Levenberg-Marquardt)
• Modified Newton: Use |H| eigenvalues
• Trust region: Constrain step size
• Line search: Ensure descent
```

---

## 📐 Proof: Quadratic Convergence

**Theorem:** Under suitable conditions (f twice continuously differentiable, H positive definite near x*, H Lipschitz continuous), Newton's method converges quadratically:

\[\|x_{k+1} - x^*\| \leq C \|x_k - x^*\|^2\]

**Proof Sketch:**

```
Step 1: Taylor expand gradient at x*
  ∇f(x_k) = ∇f(x*) + H(x*)(x_k - x*) + O(||x_k - x*||²)
          = H(x*)(x_k - x*) + O(||x_k - x*||²)  (since ∇f(x*) = 0)

Step 2: Newton step
  x_{k+1} = x_k - H(x_k)⁻¹∇f(x_k)

Step 3: Error analysis
  x_{k+1} - x* = x_k - x* - H(x_k)⁻¹∇f(x_k)
               = x_k - x* - H(x_k)⁻¹[H(x*)(x_k - x*) + O(||x_k - x*||²)]
               = [I - H(x_k)⁻¹H(x*)](x_k - x*) + O(||x_k - x*||²)

Step 4: Since H is continuous, H(x_k) → H(x*) as x_k → x*
  H(x_k)⁻¹H(x*) → I
  
  Therefore: ||x_{k+1} - x*|| = O(||x_k - x*||²)  ∎
```

---

## 🌍 Where Newton's Method Is Used

| Application | Why Newton? | Details |
|-------------|-------------|---------|
| **L-BFGS** | Approximates Newton | Scipy default |
| **Logistic Regression** | Small scale, convex | Sklearn uses Newton |
| **Scientific Computing** | Need precision | Physics simulations |
| **Trust Region (RL)** | TRPO uses Newton | Constrained optimization |
| **Interior Point** | LP/QP solvers | Gurobi, MOSEK |

---

## 📊 When to Use What

| Scenario | Method | Why |
|----------|--------|-----|
| n < 1000 | Newton | Fast, accurate |
| n < 100000 | L-BFGS | Practical Newton |
| n > 100000 | SGD/Adam | Only option |
| Non-convex DL | Adam | Handles saddles |
| Convex ML | L-BFGS | Guaranteed optimal |

---

## 📚 Resources

| Type | Title | Link |
|------|-------|------|
| 📖 | Nocedal & Wright | [Springer](https://link.springer.com/book/10.1007/978-0-387-40065-5) |
| 📄 | L-BFGS Paper | [ACM](https://dl.acm.org/doi/10.1145/279232.279236) |
| 🎥 | Newton's Method | [YouTube](https://www.youtube.com/watch?v=sDv4f4s2SB8) |
| 🎥 | GD Visualization | [YouTube](https://www.youtube.com/watch?v=IHZwWFHWa-w) |
| 🇨🇳 | 知乎 梯度下降 | [知乎](https://zhuanlan.zhihu.com/p/25202034) |
| 🇨🇳 | 知乎 牛顿法 | [知乎](https://zhuanlan.zhihu.com/p/37588590) |

---

## 🔗 Where This Topic Is Used

| Application | Usage |
|-------------|-------|
| **Machine Learning** | Core concept for ML systems |
| **Deep Learning** | Foundation for neural networks |
| **Research** | Important for understanding papers |

---

⬅️ [Back: Foundations](../01_foundations/) | ➡️ [Next: Advanced Methods](../03_advanced_methods/)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=FF6B6B&height=80&section=footer" width="100%"/>
</p>
