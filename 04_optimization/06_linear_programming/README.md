<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=FF6B6B&height=120&section=header&text=Linear%20Programming&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-04-FF6B6B?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 📂 Subtopics

| Folder | Topic | Applications |
|--------|-------|--------------|
| [01_simplex/](./01_simplex/) | Simplex Method | Classic LP solver |

---

## 🎯 What is Linear Programming?

```
+---------------------------------------------------------+
|                                                         |
|   STANDARD FORM:                                        |
|                                                         |
|   minimize    cᵀx         (linear objective)            |
|   subject to  Ax = b      (linear constraints)          |
|               x ≥ 0       (non-negativity)              |
|                                                         |
|   where:                                                |
|   • x ∈ ℝⁿ = decision variables                        |
|   • c ∈ ℝⁿ = cost coefficients                         |
|   • A ∈ ℝᵐˣⁿ = constraint matrix                       |
|   • b ∈ ℝᵐ = right-hand side                           |
|                                                         |
+---------------------------------------------------------+
```

---

## 🌍 Real-World Applications

| Industry | Problem | LP Formulation |
|----------|---------|----------------|
| **Airlines** | Crew scheduling | Minimize cost, cover all flights |
| **Logistics** | Route optimization | Minimize distance, capacity constraints |
| **Manufacturing** | Production planning | Maximize profit, resource limits |
| **Finance** | Portfolio optimization | Minimize risk, return constraints |
| **Energy** | Power grid dispatch | Minimize cost, meet demand |
| **Telecom** | Network flow | Maximize throughput, capacity |

---

## 📐 Simple Example

### Problem: Factory Production

```
A factory makes chairs (x₁) and tables (x₂)

Profit: $20 per chair, $30 per table

Constraints:
• Wood: 4 units/chair, 3 units/table, 120 units available
• Labor: 2 hours/chair, 4 hours/table, 80 hours available

Formulation:
maximize    20x₁ + 30x₂           (profit)
subject to  4x₁ + 3x₂ ≤ 120       (wood)
            2x₁ + 4x₂ ≤ 80        (labor)
            x₁, x₂ ≥ 0
```

---

## 🔗 Three Methods

```
+---------------------------------------------------------+
|                                                         |
|                    LP SOLVERS                           |
|                                                         |
+-------------+-----------------+-------------------------+
|   SIMPLEX   |   DUAL SIMPLEX  |   INTERIOR POINT        |
+-------------+-----------------+-------------------------+
| Walk on     | Dual space      | Walk through            |
| vertices    | vertices        | interior                |
|             |                 |                         |
| Fast in     | Good for        | Polynomial              |
| practice    | re-optimization | worst-case              |
|             |                 |                         |
| Exponential | Exponential     | O(n³·⁵)                 |
| worst-case  | worst-case      | guaranteed              |
+-------------+-----------------+-------------------------+
```

---

## 💻 Code Examples

### Python (PuLP)
```python
from pulp import *

# Create problem
prob = LpProblem("Factory", LpMaximize)

# Decision variables
x1 = LpVariable("chairs", lowBound=0)
x2 = LpVariable("tables", lowBound=0)

# Objective
prob += 20*x1 + 30*x2, "Profit"

# Constraints
prob += 4*x1 + 3*x2 <= 120, "Wood"
prob += 2*x1 + 4*x2 <= 80, "Labor"

# Solve
prob.solve()

print(f"Chairs: {x1.value()}")
print(f"Tables: {x2.value()}")
print(f"Profit: ${value(prob.objective)}")
```

### Python (SciPy)
```python
from scipy.optimize import linprog

# minimize -c'x (negate for maximization)
c = [-20, -30]  
A = [[4, 3], [2, 4]]  
b = [120, 80]

result = linprog(c, A_ub=A, b_ub=b)
print(f"Optimal x: {result.x}")
print(f"Profit: ${-result.fun}")
```

### Gurobi (Commercial)
```python
import gurobipy as gp

m = gp.Model()
x1 = m.addVar(name="chairs")
x2 = m.addVar(name="tables")

m.setObjective(20*x1 + 30*x2, gp.GRB.MAXIMIZE)
m.addConstr(4*x1 + 3*x2 <= 120, "Wood")
m.addConstr(2*x1 + 4*x2 <= 80, "Labor")

m.optimize()
```

---

## 📊 Comparison of Methods

| Method | Best For | Complexity |
|--------|----------|------------|
| **Simplex** | Most practical problems | Exp worst, fast avg |
| **Dual Simplex** | Re-optimization, warm start | Same as simplex |
| **Interior Point** | Very large problems | O(n^3.5 log(1/ε)) |

---

# Part 1: Simplex Method

## 🎯 Visual Overview

*The Simplex method moves along edges of the feasible polytope, improving the objective at each vertex until reaching the optimal corner.*

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

## 📐 Proof: Simplex Optimality

**Theorem:** If all reduced costs are non-negative (c̄ ≥ 0), then the current basic feasible solution is optimal.

**Proof:**

```
Let x^B be current BFS with objective z* = c_B^T x_B

For any feasible x:
z = c^T x = c_B^T x_B + c_N^T x_N

Using Ax = b:
B x_B + N x_N = b
x_B = B^{-1}b - B^{-1}N x_N

Substituting:
z = c_B^T (B^{-1}b - B^{-1}N x_N) + c_N^T x_N
  = c_B^T B^{-1}b + (c_N - N^T B^{-T} c_B)^T x_N
  = z* + c̄_N^T x_N

Since c̄_N ≥ 0 and x_N ≥ 0:
z = z* + c̄_N^T x_N ≥ z*

Therefore x^B is optimal. ∎
```

---

## 💻 Simplex Code Example

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

# Part 2: LP Duality

## 📐 Mathematical Foundations

### Primal-Dual Pair

```
PRIMAL (P):                      DUAL (D):
─────────────                    ─────────────
min  cᵀx                         max  bᵀy
s.t. Ax ≥ b                      s.t. Aᵀy ≤ c
     x ≥ 0                            y ≥ 0

Relationship:
• n primal variables ↔ n dual constraints
• m primal constraints ↔ m dual variables
• Primal min ↔ Dual max
```

### Converting Between Forms

```
Standard conversions:
┌──────────────────┬──────────────────┐
│     PRIMAL       │      DUAL        │
├──────────────────┼──────────────────┤
│ aᵢᵀx ≥ bᵢ       │ yᵢ ≥ 0           │
│ aᵢᵀx ≤ bᵢ       │ yᵢ ≤ 0           │
│ aᵢᵀx = bᵢ       │ yᵢ free          │
├──────────────────┼──────────────────┤
│ xⱼ ≥ 0          │ aⱼᵀy ≤ cⱼ       │
│ xⱼ ≤ 0          │ aⱼᵀy ≥ cⱼ       │
│ xⱼ free         │ aⱼᵀy = cⱼ       │
└──────────────────┴──────────────────┘
```

---

## 📐 Weak Duality Theorem

**Theorem:** For any feasible x (primal) and y (dual): bᵀy ≤ cᵀx

**Proof:**

```
Given:
• y ≥ 0, Ax ≥ b (primal feasibility)
• x ≥ 0, Aᵀy ≤ c (dual feasibility)

Step 1:
y ≥ 0 and Ax ≥ b implies:
yᵀ(Ax) ≥ yᵀb = bᵀy

Step 2:
x ≥ 0 and Aᵀy ≤ c implies:
xᵀ(Aᵀy) ≤ xᵀc = cᵀx

Step 3:
Note that yᵀAx = xᵀAᵀy (scalar transpose)

Combining:
bᵀy ≤ yᵀAx = xᵀAᵀy ≤ cᵀx

Therefore: bᵀy ≤ cᵀx  ∎

Implication: Any dual feasible solution gives a lower bound on primal optimal!
```

---

## 📐 Strong Duality Theorem

**Theorem:** If primal has optimal solution x*, then dual has optimal solution y* with cᵀx* = bᵀy*

```
The duality gap is zero at optimum!

Conditions for strong duality:
• Both primal and dual are feasible
• (Slater's condition for convex programs)

Proof outline (via complementary slackness):
At optimum, KKT conditions hold:
1. ∇L = 0 (stationarity)
2. Primal and dual feasibility
3. μᵢgᵢ(x) = 0 (complementarity)

These imply strong duality through Lagrangian saddle point. ∎
```

### Complementary Slackness

```
At optimum (x*, y*):

Primal slack × Dual variable = 0:
    y*ⱼ · (aⱼᵀx* - bⱼ) = 0  for all j

Dual slack × Primal variable = 0:
    x*ᵢ · (cᵢ - aᵢᵀy*) = 0  for all i

Interpretation:
• If constraint is slack, dual variable = 0
• If dual variable > 0, constraint is tight
```

---

## 💻 Duality Code Example

```python
import numpy as np
from scipy.optimize import linprog

def solve_primal_dual(c, A_ub, b_ub):
    """
    Solve primal and verify dual relationships
    
    Primal: min cᵀx s.t. Ax ≤ b, x ≥ 0
    Dual:   max bᵀy s.t. Aᵀy ≤ c, y ≥ 0
    """
    # Solve primal
    result_primal = linprog(c, A_ub=A_ub, b_ub=b_ub, method='highs')
    x_opt = result_primal.x
    primal_obj = result_primal.fun
    
    # Solve dual: max bᵀy → min -bᵀy
    m, n = A_ub.shape
    result_dual = linprog(
        -b_ub,  # Negate for maximization
        A_ub=A_ub.T,  # Transpose
        b_ub=c,
        method='highs'
    )
    y_opt = result_dual.x
    dual_obj = -result_dual.fun  # Negate back
    
    print(f"Primal optimal: {primal_obj:.6f}")
    print(f"Dual optimal:   {dual_obj:.6f}")
    print(f"Duality gap:    {abs(primal_obj - dual_obj):.6f}")
    
    # Verify complementary slackness
    primal_slack = b_ub - A_ub @ x_opt
    print(f"\nComplementary slackness check:")
    for j in range(m):
        cs = y_opt[j] * primal_slack[j]
        print(f"  y[{j}] * slack[{j}] = {y_opt[j]:.4f} * {primal_slack[j]:.4f} = {cs:.6f}")
    
    return x_opt, y_opt

# Example: Simple production problem
c = np.array([4, 3])  # Profits (minimize negative)
A = np.array([
    [2, 1],   # Machine 1 hours
    [1, 2],   # Machine 2 hours
])
b = np.array([8, 7])  # Available hours

x_opt, y_opt = solve_primal_dual(-c, A, b)  # Note: negate c for max
```

---

## 🌍 Economic Interpretation: Shadow Prices

```
Shadow Prices (Dual Variables):

y*ⱼ = ∂(optimal objective) / ∂bⱼ

• Marginal value of relaxing constraint j
• How much would you pay for one more unit of resource j?

Example:
If y*₁ = 1.5 for machine-1 hours constraint
→ One additional hour of machine 1 improves profit by $1.50
```

---

# Part 3: Interior Point Methods

## 🎯 Visual Overview

*Interior point methods follow the central path through the interior, converging to the optimal solution in polynomial time O(√n log(1/ε)).*

---

## 📐 Mathematical Foundations

### Barrier Method

```
Original LP:
    min  cᵀx
    s.t. Ax = b, x ≥ 0

Barrier formulation:
    min  cᵀx - μ Σᵢ log(xᵢ)
    s.t. Ax = b

• Barrier: -log(x) → ∞ as x → 0 (keeps iterates interior)
• μ > 0: barrier parameter
• As μ → 0, solution approaches LP optimum
```

### Central Path

```
For each μ > 0, barrier problem has unique solution x*(μ)

Central path: { x*(μ) : μ > 0 }

Properties:
• x*(μ) strictly in interior (all xᵢ > 0)
• As μ → 0⁺, x*(μ) → x* (LP solution)
• Path is smooth curve through interior
```

### KKT Conditions for Barrier Problem

```
Optimality conditions at x*(μ):

∇ₓL = c - μX⁻¹e + Aᵀλ = 0
Ax = b
x > 0

Where X = diag(x), e = (1,1,...,1)ᵀ

Rearranging: Xλ = μe  (complementarity modified)
```

### Primal-Dual Interior Point

```
Solve primal AND dual simultaneously!

Primal: min cᵀx,  Ax = b, x ≥ 0
Dual:   max bᵀy,  Aᵀy + s = c, s ≥ 0

Modified KKT system:
┌           ┐ ┌    ┐   ┌        ┐
│  A   0  0 │ │ Δx │   │ b - Ax │
│  0  Aᵀ  I │ │ Δy │ = │ c-Aᵀy-s│
│  S   0  X │ │ Δs │   │ μe - XSe│
└           ┘ └    ┘   └        ┘

Where X = diag(x), S = diag(s)
```

---

## 📐 Convergence Analysis

**Theorem:** Interior point methods converge in O(√n log(1/ε)) iterations.

**Proof sketch:**

```
Step 1: Define duality measure
  μ = xᵀs / n (average complementarity)

Step 2: Show μ decreases geometrically
  After each Newton step: μ_new ≤ (1 - θ/√n) μ
  where θ ∈ (0,1) is step parameter

Step 3: Count iterations
  Starting μ₀, target ε:
  Need k iterations where (1 - θ/√n)^k μ₀ ≤ ε
  
  Taking logs:
  k ≥ log(μ₀/ε) / log(1/(1-θ/√n))
    ≈ (√n/θ) log(μ₀/ε)
    = O(√n log(1/ε))  ∎
```

### Complexity Analysis

```
Number of iterations: O(√n log(1/ε))

Per iteration: O(n³) for solving linear system

Total: O(n^3.5 log(1/ε))

Compare to Simplex:
• Simplex: O(2ⁿ) worst case, O(m) average
• Interior: O(√n) iterations (polynomial guaranteed)
```

---

## 💻 Interior Point Code Example

```python
import numpy as np

def interior_point_lp(c, A, b, tol=1e-8, mu_factor=0.1):
    """
    Primal-Dual Interior Point for LP
    min cᵀx s.t. Ax = b, x ≥ 0
    """
    m, n = A.shape
    
    # Initialize (strictly feasible)
    x = np.ones(n)
    s = np.ones(n)  # Slack for dual
    y = np.zeros(m)  # Dual variables
    
    for iteration in range(100):
        # Duality gap and barrier parameter
        mu = (x @ s) / n
        if mu < tol:
            break
        
        # Target (reduce μ)
        sigma = mu_factor
        mu_target = sigma * mu
        
        # Residuals
        r_primal = b - A @ x
        r_dual = c - A.T @ y - s
        r_cent = mu_target * np.ones(n) - x * s
        
        # Form and solve Newton system
        # [  A    0    0  ] [dx]   [r_primal]
        # [  0   Aᵀ   I  ] [dy] = [r_dual  ]
        # [  S    0    X  ] [ds]   [r_cent  ]
        
        X_inv = np.diag(1/x)
        S = np.diag(s)
        
        # Schur complement: (A X⁻¹ S Aᵀ) dy = ...
        M = A @ X_inv @ S @ A.T
        rhs = r_primal + A @ X_inv @ (r_cent - x * r_dual)
        
        dy = np.linalg.solve(M, rhs)
        ds = r_dual - A.T @ dy
        dx = X_inv @ (r_cent - x * ds)
        
        # Step size (stay interior: x, s > 0)
        alpha_primal = min(1.0, 0.99 * min(-x[dx<0] / dx[dx<0], default=1.0))
        alpha_dual = min(1.0, 0.99 * min(-s[ds<0] / ds[ds<0], default=1.0))
        
        # Update
        x = x + alpha_primal * dx
        y = y + alpha_dual * dy
        s = s + alpha_dual * ds
        
        print(f"Iter {iteration}: μ = {mu:.2e}, obj = {c@x:.6f}")
    
    return x, c @ x

# Example
c = np.array([-1, -2])  # min -x₁ - 2x₂ (maximize x₁ + 2x₂)
A = np.array([[1, 1, 1, 0],
              [2, 1, 0, 1]])  # With slacks
b = np.array([4, 5])

# Note: Need to handle equality constraints properly
```

---

## 📊 Comparison: Simplex vs Interior Point

| Aspect | Simplex | Interior Point |
|--------|---------|----------------|
| **Path** | Along edges (vertices) | Through interior |
| **Complexity** | Exponential worst case | Polynomial O(√n) |
| **Practice** | Fast for sparse/small | Fast for dense/large |
| **Warm start** | Easy | Difficult |
| **Sensitivity** | Direct from basis | From KKT |

---

## 📚 Resources

| Type | Title | Link |
|------|-------|------|
| 📖 | Bertsimas & Tsitsiklis | [Book](https://www.amazon.com/Introduction-Linear-Optimization-Scientific-Computation/dp/1886529191) |
| 📖 | Vanderbei - LP | [Free PDF](https://vanderbei.princeton.edu/LPbook/) |
| 📖 | Boyd & Vandenberghe Ch. 11 | [Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/) |
| 🛠️ | Gurobi (Solver) | [Link](https://www.gurobi.com/) |
| 🛠️ | OR-Tools (Google) | [Link](https://developers.google.com/optimization) |
| 📄 | Karmarkar (1984) | Original polynomial-time algorithm |
| 📄 | Mehrotra (1992) | Predictor-corrector method |
| 🇨🇳 | 知乎 线性规划 | [知乎](https://zhuanlan.zhihu.com/p/26377904) |
| 🇨🇳 | 单纯形法详解 | [知乎](https://zhuanlan.zhihu.com/p/31644892) |
| 🇨🇳 | 内点法详解 | [知乎](https://zhuanlan.zhihu.com/p/48476987) |

---

## 🔗 Where This Topic Is Used

| Application | Usage |
|-------------|-------|
| **Machine Learning** | SVM, sparse optimization |
| **Operations Research** | Supply chain, scheduling |
| **Finance** | Portfolio optimization |
| **Network Design** | Flow optimization |

---

⬅️ [Back: Constrained Optimization](../05_constrained_optimization/) | ➡️ [Next: Integer Programming](../07_integer_programming/)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=FF6B6B&height=80&section=footer" width="100%"/>
</p>
