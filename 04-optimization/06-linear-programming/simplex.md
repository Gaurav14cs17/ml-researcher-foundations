# Simplex Method

> **Classic algorithm for Linear Programming**

---

## 🎯 Visual Overview

<img src="./simplex/images/simplex.svg" width="100%">

*Caption: The Simplex method moves along edges of the feasible polytope, improving the objective at each vertex until reaching the optimal corner.*

---

## 📐 Mathematical Foundations

### Standard Form LP

```
Minimize:    cᵀx
Subject to:  Ax = b
             x ≥ 0

Where:
• x ∈ ℝⁿ (decision variables)
• c ∈ ℝⁿ (cost vector)
• A ∈ ℝᵐˣⁿ (constraint matrix, m < n)
• b ∈ ℝᵐ (right-hand side, b ≥ 0)
```

### Basic Feasible Solutions

```
Partition: x = [x_B, x_N]  (basic and non-basic)
           A = [B, N]       (basis matrix and non-basic)

Basic solution:
x_B = B⁻¹b,  x_N = 0

Feasible if: x_B ≥ 0

Number of basic solutions: C(n, m) = n!/(m!(n-m)!)
```

### Reduced Costs

```
Reduced cost for non-basic variable j:

c̄_j = c_j - c_Bᵀ B⁻¹ A_j
    = c_j - πᵀ A_j

where π = B⁻ᵀ c_B (simplex multipliers / dual variables)

Optimality condition: c̄_j ≥ 0 for all j ∈ N
```

### Simplex Tableau

```
Initial tableau:
┌────────────────────────────────────┐
│   | x_B | x_N |  b  │
│───|─────|─────|─────│
│ A |  I  | B⁻¹N| B⁻¹b│
│───|─────|─────|─────│
│ c̄ |  0  | c̄_N | -z  │
└────────────────────────────────────┘

After row operations:
• Basic variables have identity columns
• c̄_B = 0 (reduced costs of basic vars)
• Last entry = negative of objective value
```

---

## 🔄 Simplex Algorithm

### Step-by-Step

```
1. Initialize: Find basic feasible solution (BFS)

2. Optimality test: 
   If c̄_j ≥ 0 for all j, STOP (optimal)

3. Select entering variable (pricing):
   Choose j with c̄_j < 0 (most negative or Dantzig's rule)

4. Compute direction:
   d = B⁻¹ A_j (column of entering variable)

5. Ratio test (select leaving variable):
   θ* = min { (B⁻¹b)_i / d_i : d_i > 0 }
   Leaving variable = argmin of above
   
   If all d_i ≤ 0: Problem is UNBOUNDED

6. Pivot: Update basis, tableau, and repeat
```

### Pivot Operation

```
If variable x_s enters and x_r leaves:

New tableau element:
ā_ij = a_ij - (a_is × a_rj) / a_rs

Pivot element a_rs becomes 1
Pivot column becomes unit vector e_r
```

---

## 💻 Code Example

```python
import numpy as np

def simplex(c, A, b):
    """
    Solve: min cᵀx s.t. Ax = b, x ≥ 0
    Assumes initial BFS with identity basis in last m columns
    """
    m, n = A.shape
    
    # Initial basis (last m columns)
    basis = list(range(n - m, n))
    
    while True:
        # Extract basis matrix
        B = A[:, basis]
        B_inv = np.linalg.inv(B)
        
        # Basic solution
        x_B = B_inv @ b
        
        # Compute reduced costs
        c_B = c[basis]
        pi = B_inv.T @ c_B  # Dual variables
        
        # Find entering variable (most negative reduced cost)
        reduced_costs = c - A.T @ pi
        reduced_costs[basis] = 0  # Set basic to 0
        
        j = np.argmin(reduced_costs)
        if reduced_costs[j] >= -1e-10:
            # Optimal!
            x = np.zeros(n)
            x[basis] = x_B
            return x, c @ x
        
        # Direction
        d = B_inv @ A[:, j]
        
        if np.all(d <= 0):
            raise ValueError("Problem is unbounded")
        
        # Ratio test
        ratios = np.full(m, np.inf)
        positive_d = d > 1e-10
        ratios[positive_d] = x_B[positive_d] / d[positive_d]
        
        r = np.argmin(ratios)
        
        # Pivot: swap variables
        basis[r] = j

# Two-phase simplex for finding initial BFS
def two_phase_simplex(c, A, b):
    """Handle problems without obvious initial BFS"""
    m, n = A.shape
    
    # Phase 1: Minimize sum of artificial variables
    A_aug = np.hstack([A, np.eye(m)])
    c_phase1 = np.concatenate([np.zeros(n), np.ones(m)])
    
    x_phase1, obj = simplex(c_phase1, A_aug, b)
    
    if obj > 1e-10:
        raise ValueError("Problem is infeasible")
    
    # Phase 2: Solve original problem
    # (Implementation depends on basis from phase 1)
    pass
```

---

## 📊 Complexity Analysis

```
Worst case: Exponential O(2ⁿ) pivots (rare in practice)
Average case: O(m) pivots (polynomial in practice)

Klee-Minty cube: Exponential example
• n-dimensional hypercube
• 2ⁿ - 1 pivots with Dantzig's rule
• Solved by randomized pivot rules

Per-pivot cost: O(m²) for tableau update
```

---

## 🔧 Pivot Rules

| Rule | Description | Behavior |
|------|-------------|----------|
| **Dantzig** | Most negative c̄_j | Classic, can cycle |
| **Bland** | Smallest index j with c̄_j < 0 | Anti-cycling |
| **Steepest edge** | Maximize improvement per unit | Practical |
| **Random** | Random j with c̄_j < 0 | Polynomial expected |

---

## 📚 References

| Type | Resource | Link |
|------|----------|------|
| 📖 | Bertsimas & Tsitsiklis | Introduction to Linear Optimization |
| 📖 | Chvátal | Linear Programming |
| 🎥 | MIT OCW 15.053 | [Simplex Lecture](https://ocw.mit.edu/courses/15-053-optimization-methods-in-management-science-spring-2013/) |
| 🇨🇳 | 单纯形法详解 | [知乎](https://zhuanlan.zhihu.com/p/31644892) |
| 🇨🇳 | 线性规划入门 | [CSDN](https://blog.csdn.net/golden1314521/article/details/44282917) |

---

---

⬅️ [Back: Interior Point](./interior-point.md)
