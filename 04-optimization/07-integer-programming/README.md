<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=07 Integer Programming&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---


# 📂 Integer Programming

> **Discrete optimization with integer constraints (NP-hard)**

---

## 🎯 Visual Overview

<img src="./images/integer-prog.svg" width="100%">

*Caption: Integer programming deals with optimization where variables must be integers. Solved via branch & bound, cutting planes, and branch & cut methods.*

---

## 📐 Mathematical Foundations

### Integer Linear Program (ILP)

```
minimize    cᵀx
subject to  Ax ≤ b
            x ∈ ℤⁿ  (integer constraint!)

Complexity: NP-hard in general
Special cases: Some polynomial (e.g., totally unimodular A)
```

### Mixed Integer LP (MIP/MILP)

```
minimize    cᵀx + dᵀy
subject to  Ax + By ≤ b
            x ∈ ℤⁿ     (integer variables)
            y ∈ ℝᵐ     (continuous variables)

Applications:
• Facility location
• Vehicle routing
• Scheduling
• Network design
```

### Binary Integer Program (BIP)

```
minimize    cᵀx
subject to  Ax ≤ b
            x ∈ {0,1}ⁿ  (binary decisions)

Special case of ILP
Models: selection, assignment, knapsack
```

### LP Relaxation

```
Relax integer constraint x ∈ ℤⁿ to x ∈ ℝⁿ:

LP relaxation provides:
• Lower bound on ILP optimal (for minimization)
• Upper bound for maximization
• Fractional solutions guide branching

Integrality gap = (ILP optimal) / (LP relaxation optimal)
```

### Branch and Bound Algorithm

```
Algorithm:
1. INITIALIZE: 
   - Solve LP relaxation → z_LP, x_LP
   - Best integer solution z* = ∞
   - Active nodes = {root}

2. While active nodes non-empty:
   a. SELECT: Pick node from active list
   
   b. BOUND: Solve LP relaxation
      - If infeasible: prune
      - If z_LP ≥ z*: prune (can't improve)
   
   c. Check if solution is integer:
      - If yes and z_LP < z*: update z* = z_LP
      - If no: BRANCH on fractional variable
        Create two children: xᵢ ≤ ⌊x̄ᵢ⌋ and xᵢ ≥ ⌈x̄ᵢ⌉

3. Return z* and corresponding solution

Branching strategies:
• Most fractional: Branch on variable closest to 0.5
• Strong branching: Evaluate LP bounds of potential branches
• Pseudocost: Use historical branching information
```

### Cutting Planes

```
Strengthen LP relaxation by adding valid inequalities:

Gomory cuts:
From optimal simplex tableau row:
Σⱼ fⱼ xⱼ ≥ f₀

where fⱼ = fractional part of coefficient

Properties:
• Cut off fractional solution
• Do not cut off any integer feasible point
• Finite convergence (Gomory's algorithm)
```

### Branch and Cut

```
Combine branch & bound with cutting planes:

1. At each node:
   - Solve LP relaxation
   - Generate violated cuts
   - Add cuts and re-solve
   
2. Branch when no more cuts found

Most successful approach for modern MIP solvers
```

---

## 💻 Code Example

```python
from scipy.optimize import milp, LinearConstraint, Bounds
import numpy as np

def solve_knapsack_milp(values, weights, capacity):
    """
    0-1 Knapsack as MILP
    
    max  Σ vᵢxᵢ
    s.t. Σ wᵢxᵢ ≤ W
         xᵢ ∈ {0,1}
    """
    n = len(values)
    
    # Objective: maximize values (negate for milp which minimizes)
    c = -np.array(values)
    
    # Constraint: weights ≤ capacity
    A = np.array([weights])
    b_u = np.array([capacity])
    b_l = np.array([-np.inf])
    
    constraints = LinearConstraint(A, b_l, b_u)
    
    # Variable bounds and integrality
    bounds = Bounds(lb=0, ub=1)
    integrality = np.ones(n)  # All binary
    
    result = milp(c, constraints=constraints, bounds=bounds, 
                  integrality=integrality)
    
    return -result.fun, result.x

# Example
values = [60, 100, 120]
weights = [10, 20, 30]
capacity = 50

opt_value, selection = solve_knapsack_milp(values, weights, capacity)
print(f"Optimal value: {opt_value}")
print(f"Selected items: {np.where(selection > 0.5)[0]}")
```

---

## 📊 Complexity and Solvers

| Problem Type | Complexity | Solver Examples |
|--------------|------------|-----------------|
| ILP | NP-hard | Gurobi, CPLEX, SCIP |
| 0-1 Knapsack | Pseudo-polynomial | Dynamic programming |
| TSP | NP-hard | Concorde, OR-Tools |
| Set Cover | NP-hard | Greedy + LP rounding |

---

## 📁 Topics

| File | Topic | Key Concept |
|------|-------|-------------|
| [milp.md](./milp.md) | Mixed Integer LP | Continuous + integer variables |
| [branch-bound.md](./branch-bound.md) | Branch & Bound | Tree search with LP bounds |

---

## 📚 References

| Type | Resource | Link |
|------|----------|------|
| 📖 | Wolsey | Integer Programming |
| 📖 | Korte & Vygen | Combinatorial Optimization |
| 📄 | Gomory (1958) | Cutting plane method |
| 🛠️ | Gurobi | [gurobi.com](https://www.gurobi.com/) |
| 🛠️ | SCIP | [scipopt.org](https://www.scipopt.org/) |
| 🇨🇳 | 整数规划基础 | [知乎](https://zhuanlan.zhihu.com/p/25110450) |
| 🇨🇳 | 分支定界法 | [CSDN](https://blog.csdn.net/u011285477/article/details/80274246) |

---

⬅️ [Back: Linear Programming](../06-linear-programming/) | ➡️ [Next: Machine Learning](../08-machine-learning/)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>
