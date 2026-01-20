<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=FF6B6B&height=120&section=header&text=Integer%20Programming&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-04-FF6B6B?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

*Integer programming deals with optimization where variables must be integers. Solved via branch & bound, cutting planes, and branch & cut methods.*

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

---

## 📐 LP Relaxation

```
Relax integer constraint x ∈ ℤⁿ to x ∈ ℝⁿ:

LP relaxation provides:
• Lower bound on ILP optimal (for minimization)
• Upper bound for maximization
• Fractional solutions guide branching

Integrality gap = (ILP optimal) / (LP relaxation optimal)
```

### Integrality Gap Analysis

**Definition:** The integrality gap is the worst-case ratio between the integer optimum and LP relaxation.

```
For minimization:
  IG = sup { OPT_ILP(I) / OPT_LP(I) : I is an instance }

For maximization:
  IG = inf { OPT_ILP(I) / OPT_LP(I) : I is an instance }

Example - Vertex Cover:
• LP relaxation: x_v ≥ 1/2 for edge endpoints
• ILP: x_v ∈ {0,1}
• Integrality gap = 2

This means LP-based approximations can be at most
2x worse than optimal for vertex cover!
```

---

# Part 1: Branch and Bound

## 🎯 Branch and Bound Algorithm

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
```

### Visualization

```
                    Root LP: z=15.5, x₂=0.5
                         /          \
                        /            \
              x₂ ≤ 0                  x₂ ≥ 1
               /                        \
          z=16, x₃=0.3              z=17 (integer!)
           /       \                  ← Pruned by z*=17
      x₃ ≤ 0     x₃ ≥ 1
        /           \
    z=18         Infeasible
    (integer!)      ← Pruned
    
    Best found: z* = 17
```

---

## 📐 Branching Strategies

### Proof: Branch and Bound Correctness

**Theorem:** Branch and Bound returns an optimal integer solution if one exists.

**Proof:**

```
Step 1: Tree covers all integer solutions
  Every integer feasible x belongs to exactly one leaf node.
  At each branching on xᵢ, either xᵢ ≤ ⌊x̄ᵢ⌋ or xᵢ ≥ ⌈x̄ᵢ⌉.
  Since x is integer, xᵢ satisfies exactly one branch.

Step 2: Pruning is safe
  - Infeasible prune: No integer solutions in subtree
  - Bound prune: If LP_bound ≥ z*, all integer solutions 
    in subtree have objective ≥ LP_bound ≥ z*
    (LP relaxation is a lower bound)

Step 3: Termination
  - Tree is finite (bounded domain)
  - Each node processed once
  
Step 4: Optimality
  - All integer solutions either explored or safely pruned
  - z* is best among explored solutions
  - Pruned solutions are no better than z*
  
  Therefore z* is optimal. ∎
```

### Branching Strategies

| Strategy | Description | When to Use |
|----------|-------------|-------------|
| **Most fractional** | Branch on variable closest to 0.5 | Simple, fast |
| **Strong branching** | Evaluate LP bounds of potential branches | Better bounds, slower |
| **Pseudocost** | Use historical branching information | After warmup |
| **Reliability** | Hybrid of strong + pseudocost | State of the art |

```python
def select_branch_variable_strong(x_frac, model, fractional_vars):
    """
    Strong branching: evaluate actual LP bounds
    More expensive but better tree size
    """
    best_score = -np.inf
    best_var = None
    
    for j in fractional_vars:
        # Evaluate down branch (x_j ≤ floor)
        model.add_constraint(x[j] <= floor(x_frac[j]))
        z_down = model.solve_lp()
        model.remove_last_constraint()
        
        # Evaluate up branch (x_j ≥ ceil)
        model.add_constraint(x[j] >= ceil(x_frac[j]))
        z_up = model.solve_lp()
        model.remove_last_constraint()
        
        # Score: product of improvements (common choice)
        score = max(z_down - z_root, eps) * max(z_up - z_root, eps)
        
        if score > best_score:
            best_score = score
            best_var = j
    
    return best_var
```

---

## 📐 Cutting Planes

### Gomory Cuts

```
Strengthen LP relaxation by adding valid inequalities:

From optimal simplex tableau row:
Σⱼ fⱼ xⱼ ≥ f₀

where fⱼ = fractional part of coefficient

Properties:
• Cut off fractional solution
• Do not cut off any integer feasible point
• Finite convergence (Gomory's algorithm)
```

### Derivation of Gomory Cut

```
Given basic variable x_i with fractional value x̄_i in optimal tableau:

Row i: x_i + Σⱼ∈N āᵢⱼ xⱼ = b̄ᵢ

Since x_i is integer, and x̄_i = b̄ᵢ is fractional:

Define f₀ = b̄ᵢ - ⌊b̄ᵢ⌋ (fractional part of RHS)
       fⱼ = āᵢⱼ - ⌊āᵢⱼ⌋ (fractional part of coefficient)

Valid inequality:
Σⱼ∈N fⱼ xⱼ ≥ f₀

Proof that this is valid:
x_i = b̄ᵢ - Σⱼ āᵢⱼ xⱼ
    = ⌊b̄ᵢ⌋ + f₀ - Σⱼ (⌊āᵢⱼ⌋ + fⱼ) xⱼ
    = (⌊b̄ᵢ⌋ - Σⱼ ⌊āᵢⱼ⌋ xⱼ) + (f₀ - Σⱼ fⱼ xⱼ)

For integer solution:
• LHS x_i is integer
• First term is integer
• Therefore (f₀ - Σⱼ fⱼ xⱼ) must be integer

Since 0 < f₀ < 1 and fⱼ ∈ [0,1), xⱼ ≥ 0:
Either Σⱼ fⱼ xⱼ ≥ f₀ or Σⱼ fⱼ xⱼ ≤ f₀ - 1 < 0 (impossible) ∎
```

---

## 📐 Branch and Cut

```
Combine branch & bound with cutting planes:

1. At each node:
   - Solve LP relaxation
   - Generate violated cuts
   - Add cuts and re-solve
   
2. Branch when no more cuts found

Most successful approach for modern MIP solvers
```

### Branch and Cut Algorithm

```python
def branch_and_cut(c, A, b, integer_vars):
    """
    Branch and Cut for ILP
    """
    # Initialize
    queue = [Node(bounds={})]  # Root node
    best_solution = None
    best_obj = float('inf')
    
    while queue:
        node = queue.pop()
        
        # Solve LP relaxation with cuts
        while True:
            x_lp, z_lp = solve_lp(c, A, b, node.bounds)
            
            if x_lp is None:  # Infeasible
                break
                
            if z_lp >= best_obj:  # Bound
                break
                
            # Try to find violated cuts
            cuts = find_cuts(x_lp, A)
            
            if not cuts:  # No more cuts
                break
                
            # Add cuts and re-solve
            A, b = add_cuts(A, b, cuts)
        
        # Check if integer
        if is_integer(x_lp, integer_vars):
            if z_lp < best_obj:
                best_obj = z_lp
                best_solution = x_lp
        else:
            # Branch
            j = select_branch_var(x_lp, integer_vars)
            queue.append(Node(bounds={**node.bounds, j: ('<=', floor(x_lp[j]))}))
            queue.append(Node(bounds={**node.bounds, j: ('>=', ceil(x_lp[j]))}))
    
    return best_solution, best_obj
```

---

# Part 2: Mixed Integer Linear Programming (MILP)

## 📐 MILP Formulation

```
minimize    cᵀx + dᵀy
subject to  Ax + By ≤ b
            x ∈ ℤᵖ     (p integer variables)
            y ∈ ℝᵍ     (q continuous variables)
            x, y ≥ 0

Combines discrete decisions with continuous optimization
```

### Big-M Formulation

A common technique to model logical constraints:

```
If-then constraint: x = 1 ⟹ y ≤ 5

Using Big-M:
y ≤ 5 + M(1 - x)

Where M is a large constant.

When x = 1: y ≤ 5 + 0 = 5 (constraint active)
When x = 0: y ≤ 5 + M (constraint inactive, M large enough)
```

### Indicator Constraints

```
Model: z = 1 ⟺ (y > 0)

Using Big-M:
y ≤ M·z           (if z=0, then y≤0)
y ≥ ε·z           (if z=1, then y≥ε)

Or equivalently:
y - ε ≤ (M-ε)·z
y ≥ ε·z
```

---

## 💻 Code Examples

### Basic MILP with SciPy

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

### Facility Location with PuLP

```python
from pulp import *

def facility_location(n_facilities, n_customers, costs, demands, capacities):
    """
    Capacitated Facility Location Problem
    
    min  Σᵢ fᵢyᵢ + Σᵢⱼ cᵢⱼxᵢⱼ  (fixed + transport costs)
    s.t. Σᵢ xᵢⱼ = 1  ∀j        (each customer served)
         Σⱼ dⱼxᵢⱼ ≤ Kᵢyᵢ  ∀i  (capacity if open)
         xᵢⱼ ∈ [0,1], yᵢ ∈ {0,1}
    """
    prob = LpProblem("FacilityLocation", LpMinimize)
    
    # Variables
    y = LpVariable.dicts("open", range(n_facilities), cat='Binary')
    x = LpVariable.dicts("assign", 
                         ((i,j) for i in range(n_facilities) 
                                for j in range(n_customers)),
                         lowBound=0, upBound=1)
    
    # Objective: minimize total cost
    prob += (lpSum(fixed_costs[i] * y[i] for i in range(n_facilities)) +
             lpSum(costs[i][j] * x[i,j] for i in range(n_facilities)
                                        for j in range(n_customers)))
    
    # Each customer assigned to exactly one facility
    for j in range(n_customers):
        prob += lpSum(x[i,j] for i in range(n_facilities)) == 1
    
    # Capacity constraints (only if facility open)
    for i in range(n_facilities):
        prob += lpSum(demands[j] * x[i,j] for j in range(n_customers)) <= capacities[i] * y[i]
    
    prob.solve()
    return prob
```

### Traveling Salesman with OR-Tools

```python
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

def solve_tsp(distance_matrix):
    """
    Traveling Salesman Problem using OR-Tools
    """
    n = len(distance_matrix)
    
    # Create routing index manager
    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    
    # Create routing model
    routing = pywrapcp.RoutingModel(manager)
    
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node]
    
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    # Set search parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    
    # Solve
    solution = routing.SolveWithParameters(search_parameters)
    
    if solution:
        route = []
        index = routing.Start(0)
        while not routing.IsEnd(index):
            route.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        route.append(manager.IndexToNode(index))
        return route, solution.ObjectiveValue()
    
    return None, None
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

## 📐 Approximation Algorithms

For NP-hard problems, sometimes approximations are sufficient:

### LP Rounding

```
Technique for deriving integer solutions from LP:

1. Solve LP relaxation → fractional x*
2. Round to integers (various strategies):
   - Randomized rounding: round x_i to 1 with probability x*_i
   - Deterministic rounding: threshold at 1/2
   - Dependent rounding: maintain marginals

Example - Set Cover:
Randomized rounding achieves O(log n) approximation
(optimal unless P=NP!)
```

### Approximation Guarantee Proof (Set Cover)

```
Set Cover LP relaxation:
min  Σᵢ cᵢ xᵢ
s.t. Σᵢ:e∈Sᵢ xᵢ ≥ 1  ∀ element e
     xᵢ ∈ [0,1]

Randomized rounding:
Include set Sᵢ with probability x*ᵢ

Expected cost = Σᵢ cᵢ x*ᵢ = OPT_LP ≤ OPT_ILP

Coverage probability for element e:
P(e covered) = 1 - Πᵢ:e∈Sᵢ (1 - x*ᵢ)
             ≥ 1 - exp(-Σᵢ x*ᵢ)  (1-x ≤ e⁻ˣ)
             ≥ 1 - 1/e           (since Σᵢ x*ᵢ ≥ 1)

Repeat O(log n) times → high probability all covered
```

---

## 📚 References

| Type | Resource | Link |
|------|----------|------|
| 📖 | Wolsey | Integer Programming |
| 📖 | Korte & Vygen | Combinatorial Optimization |
| 📄 | Gomory (1958) | Cutting plane method |
| 🛠️ | Gurobi | [gurobi.com](https://www.gurobi.com/) |
| 🛠️ | SCIP | [scipopt.org](https://www.scipopt.org/) |
| 🛠️ | OR-Tools | [Google](https://developers.google.com/optimization) |
| 🇨🇳 | 整数规划基础 | [知乎](https://zhuanlan.zhihu.com/p/25110450) |
| 🇨🇳 | 分支定界法 | [CSDN](https://blog.csdn.net/u011285477/article/details/80274246) |

---

## 🔗 Where This Topic Is Used

| Application | Usage |
|-------------|-------|
| **Logistics** | Vehicle routing, scheduling |
| **Finance** | Portfolio with transaction costs |
| **Manufacturing** | Production planning |
| **ML** | Feature selection, neural architecture search |

---

⬅️ [Back: Linear Programming](../06_linear_programming/) | ➡️ [Next: Machine Learning](../08_machine_learning/)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=FF6B6B&height=80&section=footer" width="100%"/>
</p>
