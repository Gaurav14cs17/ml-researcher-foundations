<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=FF6B6B&height=120&section=header&text=Simplex%20Algorithm&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-04-FF6B6B?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Core Concept

The **simplex algorithm** solves linear programming problems by walking along the edges of the feasible polytope from vertex to vertex, always improving the objective, until reaching the optimum.

---

## 📐 Linear Programming Problem

```
minimize    cᵀx
subject to  Ax = b
            x ≥ 0
            
Where:
  x ∈ ℝⁿ: decision variables
  A ∈ ℝᵐˣⁿ: constraint matrix
  b ∈ ℝᵐ: right-hand side
  c ∈ ℝⁿ: objective coefficients

```

**Standard form:** All constraints are equalities, all variables non-negative

---

## 🔑 Key Insight

### Fundamental Theorem of LP

```
If an optimal solution exists, then:
• At least one optimal solution is a vertex (basic feasible solution)
• The optimum occurs at a vertex of the feasible polytope

Therefore: Only check vertices, not interior points!

```

### Why Simplex Works

```
Feasible region: Polytope (convex polygon in n dimensions)
Vertices: Finite number
Edges: Connect adjacent vertices

Simplex:
1. Start at a vertex
2. Move to adjacent vertex with better objective
3. Repeat until no improvement possible

```

---

## 📐 Algorithm

### Basic Steps

```
1. Find initial basic feasible solution (vertex)
2. While exists improving direction:
   a. Choose entering variable (pivot column)
   b. Choose leaving variable (pivot row)
   c. Perform pivot operation
   d. Update tableau
3. Return optimal solution

```

### Tableau Form

```
+------------------------------------+

|  z  |  -cᵀ  | 0 |  →  objective    |
+------------------------------------+

|  0  |   A   | b |  →  constraints  |
+------------------------------------+
    ↑      ↑     ↑
    z    vars  RHS

```

---

## 💻 Example

**Problem:**

```
maximize   3x₁ + 2x₂
subject to x₁ + x₂ ≤ 4
          2x₁ + x₂ ≤ 5
          x₁, x₂ ≥ 0

```

**Convert to standard form:**

```
minimize   -3x₁ - 2x₂
subject to x₁ + x₂ + s₁ = 4
          2x₁ + x₂ + s₂ = 5
          x₁, x₂, s₁, s₂ ≥ 0

```

**Initial tableau:**

```
+---------------------------------+

| z | 3  2  0  0 | 0 |
+---------------------------------+

|s₁ | 1  1  1  0 | 4 |
|s₂ | 2  1  0  1 | 5 |
+---------------------------------+

```

**After pivots:**

```
Optimal: x₁ = 1, x₂ = 3
Objective: 3(1) + 2(3) = 9

```

---

## 💻 Implementation

```python
import numpy as np
from scipy.optimize import linprog

# Problem: max 3x₁ + 2x₂
#          s.t. x₁ + x₂ ≤ 4
#               2x₁ + x₂ ≤ 5

c = [-3, -2]  # Minimize negative = maximize
A_ub = [[1, 1], [2, 1]]  # Upper bound constraints
b_ub = [4, 5]

result = linprog(
    c,
    A_ub=A_ub,
    b_ub=b_ub,
    bounds=[(0, None), (0, None)],
    method='simplex'
)

print(f"Optimal solution: {result.x}")      # [1, 3]
print(f"Optimal value: {-result.fun}")      # 9

```

---

## 📊 Complexity

### Worst Case

```
Exponential: O(2ⁿ) iterations possible
(Klee-Minty cube example)

```

### Average Case

```
Polynomial in practice: O(m) to O(3m) pivots
where m = number of constraints

Empirical: Very fast on real problems

```

### Why It Works Well

```
Despite exponential worst-case:
• Typical problems reach optimum quickly
• Heuristics (Dantzig's rule, steepest edge) help
• Warm-starting enables fast updates

```

---

## 🔄 Variants

### Revised Simplex

```
• More efficient: O(m³) per iteration vs O(mn)
• Works with basis matrix directly
• Standard in modern solvers

```

### Dual Simplex

```
• Maintains dual feasibility
• Useful for re-optimization
• Used in branch-and-bound

```

### Primal-Dual Methods

```
• Maintain both primal and dual feasibility
• Interior-point methods
• Polynomial worst-case complexity

```

---

## 🌍 Applications

| Domain | Application | Variables |
|--------|-------------|-----------|
| **Manufacturing** | Production planning | 1000s |
| **Transportation** | Route optimization | 10,000s |
| **Finance** | Portfolio optimization | 100s-1000s |
| **ML** | SVM training (dual) | = # samples |
| **Networks** | Max flow | = # edges |

---

## 🎓 Historical Note

Invented by **George Dantzig** in 1947 while working for the U.S. Air Force. Named one of the top 10 algorithms of the 20th century!

**Impact:**
- Enabled operations research as a field
- Used in $100B+ decisions annually
- Foundation for integer programming

---

## 📚 Resources

### Books
- **Linear Programming** - Dantzig (1963) - The original!
- **Introduction to Linear Optimization** - Bertsimas & Tsitsiklis
- **Linear Programming and Network Flows** - Bazaraa et al.

### Software
- **CPLEX** (IBM): Commercial, very fast
- **Gurobi**: Commercial, industry standard
- **GLPK**: Open-source
- **SciPy**: `scipy.optimize.linprog`

### Papers
- Dantzig (1951) - The simplex method
- Klee & Minty (1972) - Exponential example

---

⬅️ [Back: Linear Programming](../) | ➡️ [Next: Integer Programming](../../07_integer_programming/)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=FF6B6B&height=80&section=footer" width="100%"/>
</p>
