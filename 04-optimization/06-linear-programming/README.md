<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=FF6B6B&height=120&section=header&text=Linear%20Programming&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-04-FF6B6B?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 📂 Topics in This Folder

| Folder | Topic | Applications |
|--------|-------|--------------|
| [simplex/](./simplex/) | Simplex Method | Classic LP solver |
| [duality/](./duality/) | LP Duality Theory | Shadow prices, bounds |
| [interior-point/](./interior-point/) | Interior Point Methods | Large-scale LP |

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

## 📚 Resources

| Type | Title | Link |
|------|-------|------|
| 📖 | Bertsimas & Tsitsiklis | [Book](https://www.amazon.com/Introduction-Linear-Optimization-Scientific-Computation/dp/1886529191) |
| 📖 | Vanderbei - LP | [Free PDF](https://vanderbei.princeton.edu/LPbook/) |
| 🛠️ | Gurobi (Solver) | [Link](https://www.gurobi.com/) |
| 🛠️ | OR-Tools (Google) | [Link](https://developers.google.com/optimization) |
| 🇨🇳 | 知乎 线性规划 | [知乎](https://zhuanlan.zhihu.com/p/26377904) |

---

⬅️ [Back: Constrained Optimization](../05-constrained-optimization/) | ➡️ [Next: Integer Programming](../07-integer-programming/)

---

## 🔗 Where This Topic Is Used

| Application | Usage |
|-------------|-------|
| **Machine Learning** | Core concept for ML systems |
| **Deep Learning** | Foundation for neural networks |
| **Research** | Important for understanding papers |

---

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=FF6B6B&height=80&section=footer" width="100%"/>
</p>
