<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=FF6B6B&height=120&section=header&text=Lagrange%20Multipliers&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-04-FF6B6B?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Core Concept

**Lagrange multipliers** provide a method to find local extrema of a function subject to equality constraints. The key insight: at the optimum, the gradient of the objective is parallel to the gradient of the constraint.

---

## 📐 Mathematical Formulation

### Problem

```
minimize   f(x)
subject to g(x) = 0
```

### Lagrangian Function

```
L(x, λ) = f(x) + λᵀg(x)

Where:
  x: decision variables
  λ: Lagrange multipliers (one per constraint)
```

### Necessary Conditions

At optimum x*, λ*:

```
1. Stationarity:  ∇ₓL = ∇f(x*) + λᵀ∇g(x*) = 0
2. Feasibility:   g(x*) = 0
```

---

## 🔑 Geometric Intuition

```
At the optimum:
∇f(x*) ∥ ∇g(x*)  (parallel)

Why?
If ∇f not parallel to ∇g, we could move along
the constraint surface to decrease f.

Therefore: ∇f(x*) = -λ∇g(x*)
```

**Visual:**
```
        ∇f ↗
           ╱
    ━━━━━━●━━━━━  (constraint surface g(x)=0)
           ╲
        ∇g ↘
        
At optimum: both gradients perpendicular to surface
```

---

## 💡 Simple Example

**Problem:**
```
Minimize: f(x, y) = x² + y²
Subject to: g(x, y) = x + y - 1 = 0
```

**Solution:**
```python
# Lagrangian: L = x² + y² + λ(x + y - 1)

# Conditions:
# ∂L/∂x = 2x + λ = 0  →  x = -λ/2
# ∂L/∂y = 2y + λ = 0  →  y = -λ/2
# ∂L/∂λ = x + y - 1 = 0

# From first two: x = y
# From third: 2x = 1  →  x = y = 0.5

# Solution: (x*, y*) = (0.5, 0.5)
# Optimal value: f* = 0.5
# Multiplier: λ* = -1
```

---

## 💻 Implementation

```python
import numpy as np
from scipy.optimize import minimize

def objective(X):
    """f(x,y) = x² + y²"""
    return X[0]**2 + X[1]**2

def constraint(X):
    """g(x,y) = x + y - 1"""
    return X[0] + X[1] - 1

# Solve with scipy
result = minimize(
    objective,
    x0=[0, 0],
    constraints={'type': 'eq', 'fun': constraint},
    method='SLSQP'
)

print(f"Optimal point: {result.x}")           # [0.5, 0.5]
print(f"Optimal value: {result.fun}")         # 0.5
print(f"Multiplier: {result.get('lagrange')}")  # -1.0
```

---

## 🌍 Applications in ML

| Application | Objective f(x) | Constraint g(x) |
|-------------|----------------|-----------------|
| **Max Entropy** | H(p) = -Σp·log(p) | Σpᵢ = 1 |
| **SVM (Hard)** | ½\|\|w\|\|² | yᵢ(wᵀxᵢ + b) = 1 |
| **PCA** | maximize variance | \|\|w\|\| = 1 |
| **GMM** | log-likelihood | Σπₖ = 1 |

---

## 🔄 Extension to Multiple Constraints

For m constraints gᵢ(x) = 0:

```
L(x, λ) = f(x) + Σᵢλᵢgᵢ(x)

Conditions:
∇ₓL = ∇f + Σᵢλᵢ∇gᵢ = 0
gᵢ(x) = 0, ∀i
```

**Interpretation:** ∇f is in the span of {∇g₁, ..., ∇gₘ}

---

## 📚 Theory

### Second-Order Conditions

For x* to be a local minimum:

```
∇²ₓₓL(x*, λ*) is positive definite on the tangent space
T = {v : ∇g(x*)ᵀv = 0}
```

### Regularity Condition

**LICQ (Linear Independence Constraint Qualification):**
```
{∇g₁(x*), ..., ∇gₘ(x*)} are linearly independent
```

Ensures uniqueness of λ*.

---

## 📐 DETAILED MATHEMATICAL DERIVATIONS

### 1. Why Lagrange Multipliers Work: Complete Proof

**Theorem (First-Order Necessary Conditions):** Let x* be a local minimum of f(x) subject to g(x) = 0, where f and g are continuously differentiable, and ∇g(x*) ≠ 0. Then there exists λ* such that:

```
∇f(x*) + λ*∇g(x*) = 0
```

**Proof:**

```
Step 1: Define feasible directions
A direction d is feasible if there exists α > 0 such that:
x* + αd satisfies g(x* + αd) ≈ 0 for small α

Taylor expansion:
g(x* + αd) ≈ g(x*) + α∇g(x*)ᵀd
           = α∇g(x*)ᵀd           (since g(x*) = 0)

For feasibility: ∇g(x*)ᵀd = 0

So: Feasible directions lie in tangent space T = {d : ∇g(x*)ᵀd = 0}

Step 2: Optimality implies no descent direction
Since x* is a local minimum, f cannot decrease along any feasible direction:
∇f(x*)ᵀd ≥ 0  for all d ∈ T

Step 3: Characterize T⊥ (orthogonal complement)
T = {d : ∇g(x*)ᵀd = 0}
T⊥ = span(∇g(x*))

Step 4: ∇f must be in T⊥
If ∇f(x*) ∉ T⊥, then ∇f has a component in T.
Let d = -projection of ∇f onto T
Then ∇f(x*)ᵀd < 0 (descent direction in T)
Contradiction! (x* wouldn't be optimal)

Therefore: ∇f(x*) ∈ T⊥ = span(∇g(x*))

Step 5: Conclusion
∇f(x*) = -λ*∇g(x*) for some λ* ∈ ℝ
⟹ ∇f(x*) + λ*∇g(x*) = 0  ∎
```

**Geometric Interpretation:**
```
At optimum, ∇f and ∇g are parallel:

   Level curve g(x) = 0
   ──────●──────  (constraint)
        /|\
       / | \
      ∇f ∇g

If not parallel, we could move along constraint
to decrease f → not optimal!
```

---

### 2. Second-Order Sufficient Conditions

**Theorem:** If at (x*, λ*):
1. First-order conditions hold: ∇L = 0
2. LICQ holds: ∇g(x*) ≠ 0
3. ∇²ₓₓL(x*, λ*) is positive definite on T

Then x* is a strict local minimum.

**Proof Sketch:**

```
Step 1: Taylor expansion of f along constraint
For d ∈ T (feasible direction):
f(x* + d) ≈ f(x*) + ∇f(x*)ᵀd + (1/2)dᵀ∇²f(x*)d

Step 2: Use first-order conditions
∇f(x*)ᵀd = -λ*∇g(x*)ᵀd = 0  (since d ∈ T)

So: f(x* + d) ≈ f(x*) + (1/2)dᵀ∇²f(x*)d

Step 3: Hessian of Lagrangian
∇²L = ∇²f + λ*∇²g

On T, ∇²g contribution vanishes (constrained Hessian):
dᵀ∇²L d = dᵀ∇²f d + λ*dᵀ∇²g d ≈ dᵀ∇²f d

Step 4: Positive definiteness
If ∇²L ≻ 0 on T:
f(x* + d) > f(x*) for all d ∈ T, d ≠ 0
⟹ x* is strict local minimum  ∎
```

---

### 3. Multiple Constraints: General Case

**Problem:**
```
minimize   f(x)
subject to g₁(x) = 0
           g₂(x) = 0
           ...
           gₘ(x) = 0
```

**Lagrangian:**
```
L(x, λ) = f(x) + Σᵢ λᵢgᵢ(x)
        = f(x) + λᵀg(x)
```

**First-Order Necessary Conditions (KKT for equality):**
```
∇ₓL = ∇f(x*) + Σᵢ λᵢ*∇gᵢ(x*) = 0    (Stationarity)
gᵢ(x*) = 0, ∀i                      (Feasibility)
```

**Geometric interpretation:**
```
∇f(x*) ∈ span{∇g₁(x*), ..., ∇gₘ(x*)}

That is:
∇f(x*) = -Σᵢ λᵢ*∇gᵢ(x*)

Tangent space: T = {d : ∇gᵢ(x*)ᵀd = 0, ∀i}
Normal space: N = span{∇g₁, ..., ∇gₘ}

At optimum: ∇f ∈ N
```

---

### 4. Worked Example: Constrained Least Squares

**Problem:**
```
minimize   f(x) = (1/2)||Ax - b||²
subject to Cx = d
```

**Solution:**

```
Step 1: Lagrangian
L(x, λ) = (1/2)||Ax - b||² + λᵀ(Cx - d)
        = (1/2)(Ax - b)ᵀ(Ax - b) + λᵀ(Cx - d)

Step 2: Compute gradients
∂L/∂x = Aᵀ(Ax - b) + Cᵀλ = 0
∂L/∂λ = Cx - d = 0

Step 3: Solve system
From first equation: AᵀAx + Cᵀλ = Aᵀb
From second: Cx = d

In matrix form:
[AᵀA  Cᵀ] [x]   [Aᵀb]
[C    0 ] [λ] = [d  ]

This is the KKT system!

Step 4: Solution (if invertible)
[x*]   [AᵀA  Cᵀ]⁻¹ [Aᵀb]
[λ*] = [C    0 ]   [d  ]
```

**Implementation:**
```python
import numpy as np

def constrained_least_squares(A, b, C, d):
    """
    Solve: min ||Ax - b||²  s.t. Cx = d
    """
    m, n = A.shape
    p = C.shape[0]
    
    # Build KKT system
    K = np.block([[A.T @ A,  C.T    ],
                  [C,        np.zeros((p, p))]])
    
    rhs = np.concatenate([A.T @ b, d])
    
    # Solve
    sol = np.linalg.solve(K, rhs)
    
    x_opt = sol[:n]
    lambda_opt = sol[n:]
    
    return x_opt, lambda_opt

# Example
np.random.seed(42)
m, n, p = 10, 5, 2

A = np.random.randn(m, n)
b = np.random.randn(m)
C = np.random.randn(p, n)
d = np.random.randn(p)

x_opt, lambda_opt = constrained_least_squares(A, b, C, d)

print(f"Optimal x: {x_opt}")
print(f"Constraint satisfied: {np.allclose(C @ x_opt, d)}")
print(f"Objective: {0.5 * np.linalg.norm(A @ x_opt - b)**2:.6f}")
```

---

### 5. Connection to SVM (Support Vector Machines)

**Primal Problem (Hard-margin SVM):**
```
minimize   (1/2)||w||²
subject to yᵢ(wᵀxᵢ + b) ≥ 1,  ∀i
```

**Lagrangian:**
```
L(w, b, α) = (1/2)||w||² - Σᵢ αᵢ[yᵢ(wᵀxᵢ + b) - 1]

Where αᵢ ≥ 0 are Lagrange multipliers
```

**KKT Conditions:**
```
∂L/∂w = w - Σᵢ αᵢyᵢxᵢ = 0  ⟹  w = Σᵢ αᵢyᵢxᵢ
∂L/∂b = -Σᵢ αᵢyᵢ = 0      ⟹  Σᵢ αᵢyᵢ = 0

Complementary slackness:
  αᵢ[yᵢ(wᵀxᵢ + b) - 1] = 0,  ∀i
```

**Dual Problem (substituting w = Σᵢ αᵢyᵢxᵢ):**
```
maximize   Σᵢ αᵢ - (1/2)ΣᵢΣⱼ αᵢαⱼyᵢyⱼxᵢᵀxⱼ
subject to αᵢ ≥ 0, Σᵢ αᵢyᵢ = 0
```

**Support vectors:** Points where αᵢ > 0 (constraint is active)

---

### 6. Lagrange Multipliers Interpretation

**Economic interpretation:**

```
λ* = Shadow price = Marginal value of relaxing constraint

If constraint changes to g(x) = ε (small ε):
f(x*(ε)) ≈ f(x*(0)) - λ*ε

λ* tells you how much f would improve if constraint relaxed!
```

**Example:**
```
Maximize profit f(x) subject to budget g(x) = 0

λ* > 0: Increasing budget by $1 increases profit by $λ*
λ* = 0: Constraint not binding (slack in budget)
```

---

### 7. Research Paper Applications

| Paper/Method | Optimization Problem | Constraint | λ Interpretation |
|--------------|---------------------|------------|------------------|
| **PCA** | max wᵀΣw | wᵀw = 1 | Eigenvalue! |
| **Fisher LDA** | max wᵀS_Bw/wᵀS_Ww | wᵀS_Ww = 1 | Generalized eigenvalue |
| **SVM** | min ½\|\|w\|\|² | y(wx + b) ≥ 1 | Support vector weights |
| **Max Entropy** | max -Σp log p | Σp = 1, 𝔼[f] = μ | Moment constraints |
| **CCA** | max uᵀXYᵀv | uᵀu = vᵀv = 1 | Canonical correlation |

**PCA Derivation:**
```
Problem: max_w wᵀΣw  s.t. ||w|| = 1

Lagrangian: L = wᵀΣw - λ(wᵀw - 1)

∂L/∂w = 2Σw - 2λw = 0
⟹ Σw = λw

This is eigenvalue equation!
w* = eigenvector of Σ
λ* = eigenvalue (variance captured)
```

---

### 8. Practical Tips for Research Papers

**When you see Lagrange multipliers in papers:**

1. **Look for the primal-dual formulation**
   ```
   Primal: min f(x) s.t. g(x) = 0
   Dual:   max_λ min_x L(x, λ)
   ```

2. **Check for duality gap**
   ```
   Strong duality: Primal opt = Dual opt
   Holds for convex problems with constraints qualification
   ```

3. **Complementary slackness** (for inequalities):
   ```
   αᵢ gᵢ(x) = 0
   
   Either αᵢ = 0 (constraint inactive) or gᵢ(x) = 0 (binding)
   ```

4. **Augmented Lagrangian method** (ADMM):
   ```
   L_ρ(x, λ) = f(x) + λᵀg(x) + (ρ/2)||g(x)||²
   
   Add quadratic penalty for robustness
   ```

---

### 9. Connection to Modern ML

**Neural Network Training with Constraints:**

```python
# Example: Train NN with orthogonal weights
# Constraint: WᵀW = I

def augmented_lagrangian_step(W, lambda_mat, rho):
    """
    One step of augmented Lagrangian optimization
    """
    # Standard training loss
    loss = compute_loss(W)
    
    # Constraint: WᵀW - I = 0
    constraint = W.T @ W - torch.eye(W.shape[1])
    
    # Augmented Lagrangian
    aug_loss = loss + torch.trace(lambda_mat @ constraint) + \
               (rho/2) * torch.norm(constraint, 'fro')**2
    
    # Gradient descent
    aug_loss.backward()
    optimizer.step()
    
    # Update multiplier (dual ascent)
    lambda_mat = lambda_mat + rho * constraint.detach()
    
    return lambda_mat
```

---

### 10. Advanced: Constrained Optimization in Transformers

**LayerNorm as constrained optimization:**

```
Problem: Normalize activations to zero mean, unit variance

Can be viewed as:
minimize   ||h - h_norm||²
subject to mean(h_norm) = 0, var(h_norm) = 1

Using Lagrange multipliers:
h_norm = (h - μ) / σ

where μ, σ are the multipliers (learned scale/shift applied after)
```

**Attention with constraints:**

```
Attention weights: softmax(scores) implicitly solves:
maximize   Σᵢ αᵢ · scoreᵢ
subject to Σᵢ αᵢ = 1, αᵢ ≥ 0

Lagrangian: L = Σᵢ αᵢ · scoreᵢ - λ(Σᵢ αᵢ - 1)

Solution: αᵢ* = exp(scoreᵢ) / Σⱼ exp(scoreⱼ) = softmax(scores)ᵢ
```

---

##

## 🎓 Historical Note

Developed by **Joseph-Louis Lagrange** (1736-1813) in his work on celestial mechanics. The method revolutionized constrained optimization and remains fundamental 250+ years later!

---

## 📚 Resources

### Books
- **Convex Optimization** - Boyd & Vandenberghe (§5.1)
- **Numerical Optimization** - Nocedal & Wright (Ch 12)

### Papers
- Lagrange (1788) - Mécanique Analytique

### Online
- 3Blue1Brown: Lagrange Multipliers
- Khan Academy: Constrained Optimization

---

⬅️ [Back: KKT](../kkt/)


---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=FF6B6B&height=80&section=footer" width="100%"/>
</p>
