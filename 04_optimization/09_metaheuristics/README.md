<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=FF6B6B&height=120&section=header&text=Metaheuristics&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-04-FF6B6B?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 📂 Subtopics

| Folder | Topic | Best For |
|--------|-------|----------|
| [01_genetic_algorithm/](./01_genetic_algorithm/) | Genetic Algorithms | Discrete, combinatorial |

---

## 🎯 When to Use Metaheuristics?

```
+---------------------------------------------------------+
|                                                         |
|   Use Gradient-Based When:       Use Metaheuristics:    |
|   ------------------------       -------------------    |
|                                                         |
|   • Continuous variables         • Discrete/combinatorial|
|   • Differentiable function      • Black-box function   |
|   • Single objective             • Multiple objectives  |
|   • Need precision               • Need "good enough"   |
|                                                         |
|   Examples:                      Examples:              |
|   • Neural network training      • Traveling salesman   |
|   • Logistic regression          • Job scheduling       |
|   • Portfolio optimization       • Neural arch search   |
|                                                         |
+---------------------------------------------------------+
```

---

## 📐 Mathematical Foundations

### Genetic Algorithm

```
Population P_t = {x₁, ..., xₙ}

Selection: P(xᵢ selected) ∝ fitness(xᵢ)
Crossover: x_child = crossover(x_parent1, x_parent2)
Mutation: x' = x + ε, ε ~ N(0, σ²)
```

### Simulated Annealing

```
Accept worse solution with probability:
P(accept) = exp(-ΔE / T)

Temperature schedule:
T(t) = T₀ × cooling_rate^t

As T → 0, converges to greedy
```

### Particle Swarm

```
vᵢ = w×vᵢ + c₁r₁(pᵢ - xᵢ) + c₂r₂(g - xᵢ)
xᵢ = xᵢ + vᵢ

Where:
• pᵢ = particle's best position
• g = global best position
• w = inertia, c₁,c₂ = learning rates
```

---

## 📊 Comparison of Methods

| Method | Inspiration | Pros | Cons |
|--------|-------------|------|------|
| **Genetic Algorithm** | Evolution | Explores well | Slow convergence |
| **Simulated Annealing** | Metal cooling | Simple, escapes local | Single solution |
| **Particle Swarm** | Bird flocking | Easy to implement | May converge early |
| **Ant Colony** | Ant foraging | Good for paths | Problem-specific |
| **Bayesian Optimization** | Statistics | Sample efficient | Expensive per step |

---

## 🌍 Real-World Applications

| Application | Method | Why |
|-------------|--------|-----|
| **Neural Architecture Search** | GA, RL | Discrete architecture choices |
| **Hyperparameter Tuning** | Bayesian, GA | Black-box expensive function |
| **Chip Design** | Simulated Annealing | Google uses for TPU layout |
| **Game AI** | NEAT (Neuroevolution) | Evolve network topology |
| **Scheduling** | GA, Ant Colony | NP-hard combinatorial |
| **AutoML** | All of above | Full pipeline optimization |

---

## 🔗 No Gradients? No Problem!

```
Gradient-based:               Metaheuristic:

         ∇f                    Random search +
          |                    Selection pressure
          |                           |
          v                           v
         ●                           ● ●
        ╱|╲                         ╱ ● ╲
       ╱ | ╲                       ╱  |  ╲
      ╱  |  ╲                     ● --● --●
     ●   |   ●                   ╱    |    ╲
     |   |   |                  ●     ●     ●
     |      ▼      |
     |   ●   |                  Population evolves
     
   Follow gradient           Best solutions survive
```

---

# Part 1: Simulated Annealing

## 🎯 Visual Overview

*Simulated Annealing accepts worse solutions with decreasing probability, allowing escape from local minima while converging to global optimum as temperature decreases.*

---

## 📐 Core Algorithm

```
1. Initialize: x₀, T₀ (high temperature)
2. For t = 0, 1, 2, ...:
   a. Generate neighbor: x' ∈ N(xₜ)
   b. Compute: Δ = f(x') - f(xₜ)
   c. Accept with probability:
      
      P(accept) = { 1           if Δ ≤ 0 (improvement)
                  { exp(-Δ/T)   if Δ > 0 (worse solution)
   
   d. Update: xₜ₊₁ = x' if accepted, else xₜ
   e. Cool: T ← cooling(T)
```

---

## 📐 Metropolis-Hastings Criterion

```
Acceptance probability (for minimization):

P(x → x') = min(1, exp(-Δ/T))

where Δ = f(x') - f(x)

Interpretation:
• T high: exp(-Δ/T) ≈ 1, accept almost anything
• T low: exp(-Δ/T) ≈ 0, only accept improvements
• Δ small: more likely to accept worse solutions
• Δ large: less likely to accept
```

---

## 📐 Cooling Schedules

```
Geometric (most common):
Tₖ = α · Tₖ₋₁ = T₀ · αᵏ
where α ∈ [0.8, 0.99], typically 0.95

Logarithmic (theoretical):
Tₖ = T₀ / log(1 + k)
Guarantees convergence but too slow in practice

Linear:
Tₖ = T₀ - k · ΔT

Adaptive:
Adjust based on acceptance rate
If acceptance too high: cool faster
If acceptance too low: cool slower
```

---

## 📐 Convergence Theory

**Theorem (Geman & Geman, 1984):** If T(k) ≥ c / log(k+1) for large enough c, then P(x_k = x*) → 1 as k → ∞

**Proof Sketch:**

```
Step 1: Markov chain representation
SA defines a Markov chain with transition probabilities:
P(x → x') = q(x'|x) · min(1, exp(-Δ/T))

where q is the proposal distribution (neighbor generation)

Step 2: Stationary distribution
At fixed temperature T, the stationary distribution is:
π_T(x) = exp(-f(x)/T) / Z(T)

where Z(T) = Σ_y exp(-f(y)/T) is the partition function

Step 3: As T → 0
π_0(x) concentrates on global minima:
lim_{T→0} π_T(x) = { 1/|X*|  if x ∈ X* (global optima)
                   { 0       otherwise

Step 4: Cooling schedule requirement
For convergence in finite time, need T(k) → 0 slowly enough
to allow mixing of Markov chain at each temperature.

Logarithmic cooling satisfies this: T(k) = c/log(k+1)
where c ≥ max depth of local minima

This ensures enough transitions to escape all local minima
before temperature becomes too low. ∎
```

### In Practice

```
Practical considerations:
• Logarithmic too slow
• Geometric with reheating works well
• Final temperature should be small (10⁻⁶ to 10⁻³)
```

---

## 💻 Simulated Annealing Code

```python
import numpy as np

def simulated_annealing(f, x0, neighbor_fn, 
                        T0=100, T_min=1e-6, alpha=0.95,
                        max_iter=10000):
    """
    Simulated Annealing for minimization
    
    Args:
        f: objective function to minimize
        x0: initial solution
        neighbor_fn: generates random neighbor
        T0: initial temperature
        T_min: stopping temperature
        alpha: cooling rate
    """
    x = x0
    fx = f(x)
    x_best, f_best = x, fx
    T = T0
    
    history = {'T': [], 'f': [], 'accept_rate': []}
    accepted = 0
    
    for i in range(max_iter):
        if T < T_min:
            break
        
        # Generate neighbor
        x_new = neighbor_fn(x)
        fx_new = f(x_new)
        
        # Compute acceptance probability
        delta = fx_new - fx
        if delta <= 0:
            accept = True
        else:
            accept = np.random.random() < np.exp(-delta / T)
        
        # Update current solution
        if accept:
            x, fx = x_new, fx_new
            accepted += 1
            
            # Track best
            if fx < f_best:
                x_best, f_best = x, fx
        
        # Cool down
        T = alpha * T
        
        # Record history
        if i % 100 == 0:
            history['T'].append(T)
            history['f'].append(fx)
            history['accept_rate'].append(accepted / (i + 1))
    
    return x_best, f_best, history

def adaptive_sa(f, x0, neighbor_fn, T0=100, T_min=1e-6, 
                target_accept=0.4, max_iter=10000):
    """
    Adaptive SA with temperature adjustment based on acceptance rate
    """
    x = x0
    fx = f(x)
    x_best, f_best = x, fx
    T = T0
    
    window = 100
    accepts = []
    
    for i in range(max_iter):
        if T < T_min:
            break
        
        x_new = neighbor_fn(x)
        fx_new = f(x_new)
        
        delta = fx_new - fx
        accept = delta <= 0 or np.random.random() < np.exp(-delta / T)
        accepts.append(int(accept))
        
        if accept:
            x, fx = x_new, fx_new
            if fx < f_best:
                x_best, f_best = x, fx
        
        # Adaptive cooling
        if len(accepts) >= window:
            accept_rate = np.mean(accepts[-window:])
            if accept_rate > target_accept + 0.1:
                T *= 0.9  # Cool faster
            elif accept_rate < target_accept - 0.1:
                T *= 1.1  # Warm up
            else:
                T *= 0.95  # Normal cooling
    
    return x_best, f_best

# Example: Minimize Rastrigin function (many local minima!)
def rastrigin(x):
    A = 10
    return A * len(x) + sum(xi**2 - A * np.cos(2 * np.pi * xi) for xi in x)

def neighbor(x, step=0.5):
    return x + np.random.uniform(-step, step, len(x))

x0 = np.random.uniform(-5, 5, 10)
x_best, f_best, _ = simulated_annealing(rastrigin, x0, neighbor)
print(f"Best found: f(x) = {f_best:.6f}")  # Should be close to 0
```

---

## 📊 Parameter Guidelines

| Parameter | Typical Range | Effect |
|-----------|---------------|--------|
| **T₀** | f(x₀) to 10·f(x₀) | Start high enough to explore |
| **T_min** | 10⁻⁶ to 10⁻³ | End temperature |
| **α** | 0.85 to 0.99 | 0.95 common; lower = faster, less optimal |
| **Iterations per T** | 100 to 1000 | More = better at each temp |

---

# Part 2: Multi-Objective Optimization

## 📐 Problem Formulation

```
Multi-objective problem:

minimize  [f₁(x), f₂(x), ..., fₖ(x)]
subject to x ∈ X

No single optimal solution!
Instead: Pareto optimal set
```

---

## 📐 Pareto Dominance

```
Solution a dominates b (a ≻ b) if:
• fᵢ(a) ≤ fᵢ(b) for all i
• fⱼ(a) < fⱼ(b) for at least one j

Pareto optimal: Not dominated by any other solution

Pareto front: Set of all Pareto optimal objective vectors
```

### Visualization

```
     f₂
      │
    10├     ○  Non-dominated (Pareto front)
      │    ○
      │   ○    ×  Dominated points
    5 │  ○
      │ ○   ×
      │○     ×
    0 └────────────── f₁
      0     5     10

The Pareto front represents trade-offs between objectives
```

---

## 📐 NSGA-II Algorithm

The most widely used multi-objective evolutionary algorithm.

```
NSGA-II (Non-dominated Sorting Genetic Algorithm II):

1. Fast non-dominated sorting:
   - Rank 1: All non-dominated solutions
   - Rank 2: Non-dominated after removing rank 1
   - Continue until all ranked

2. Crowding distance:
   - Estimate density of solutions around each point
   - Prefer isolated solutions for diversity

3. Selection:
   - First: prefer lower rank (better Pareto front)
   - Tie-break: prefer higher crowding distance (diversity)
```

### Crowding Distance Formula

```
For solution i in front F:

d_i = Σⱼ (f_j^{i+1} - f_j^{i-1}) / (f_j^max - f_j^min)

where i+1, i-1 are neighbors when sorted by objective j

Intuition: Large crowding distance = isolated solution
Boundary solutions get infinite distance (always selected)
```

---

## 💻 NSGA-II Code

```python
import numpy as np
from typing import List, Tuple

def fast_non_dominated_sort(population: np.ndarray, 
                           objectives: np.ndarray) -> List[List[int]]:
    """
    Fast non-dominated sorting O(MN²)
    Returns list of fronts (each front is list of indices)
    """
    n = len(population)
    domination_count = np.zeros(n, dtype=int)
    dominated_by = [[] for _ in range(n)]
    fronts = [[]]
    
    # For each pair, check domination
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dominates(objectives[i], objectives[j]):
                dominated_by[i].append(j)
            elif dominates(objectives[j], objectives[i]):
                domination_count[i] += 1
        
        # If not dominated by anyone, add to first front
        if domination_count[i] == 0:
            fronts[0].append(i)
    
    # Build subsequent fronts
    k = 0
    while fronts[k]:
        next_front = []
        for i in fronts[k]:
            for j in dominated_by[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        k += 1
        if next_front:
            fronts.append(next_front)
    
    return fronts[:-1]  # Remove empty last front

def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """Check if a dominates b (all ≤ and at least one <)"""
    return np.all(a <= b) and np.any(a < b)

def crowding_distance(front_indices: List[int], 
                      objectives: np.ndarray) -> np.ndarray:
    """Compute crowding distance for solutions in a front"""
    n = len(front_indices)
    if n <= 2:
        return np.full(n, np.inf)
    
    distances = np.zeros(n)
    obj = objectives[front_indices]
    
    for m in range(objectives.shape[1]):
        # Sort by objective m
        sorted_idx = np.argsort(obj[:, m])
        
        # Boundary points get infinite distance
        distances[sorted_idx[0]] = np.inf
        distances[sorted_idx[-1]] = np.inf
        
        # Normalize by range
        obj_range = obj[sorted_idx[-1], m] - obj[sorted_idx[0], m]
        if obj_range == 0:
            continue
            
        # Interior points
        for i in range(1, n - 1):
            distances[sorted_idx[i]] += (
                obj[sorted_idx[i+1], m] - obj[sorted_idx[i-1], m]
            ) / obj_range
    
    return distances

def nsga2_select(population: np.ndarray, 
                 objectives: np.ndarray, 
                 n_select: int) -> np.ndarray:
    """Select n_select individuals using NSGA-II criteria"""
    fronts = fast_non_dominated_sort(population, objectives)
    selected = []
    
    for front in fronts:
        if len(selected) + len(front) <= n_select:
            selected.extend(front)
        else:
            # Need to select subset based on crowding distance
            n_needed = n_select - len(selected)
            cd = crowding_distance(front, objectives)
            best_idx = np.argsort(-cd)[:n_needed]  # Highest CD first
            selected.extend([front[i] for i in best_idx])
            break
    
    return population[selected]
```

---

## 📐 Scalarization Methods

Alternative to Pareto-based methods:

```
Weighted Sum:
minimize  Σᵢ wᵢfᵢ(x)

• Simple but can't find non-convex Pareto front regions
• Different weights → different solutions

ε-Constraint:
minimize  f₁(x)
subject to fᵢ(x) ≤ εᵢ  for i = 2,...,k

• Can find any Pareto solution
• Need to choose ε values
```

---

## 📊 When Metaheuristics Shine

```
Problem Characteristics → Best Method

Discrete variables          → Genetic Algorithm
Continuous + many local min → Simulated Annealing  
Expensive evaluations       → Bayesian Optimization
Multiple objectives         → NSGA-II, MOEA/D
Path finding                → Ant Colony
Swarm behavior              → Particle Swarm
```

---

## 💻 Quick Examples

### Genetic Algorithm (DEAP)
```python
from deap import base, creator, tools, algorithms
import random

# Minimize x² + y²
def fitness(individual):
    return sum(x**2 for x in individual),

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, -10, 10)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_float, n=2)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", fitness)

pop = toolbox.population(n=50)
result = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=100)
```

### Simulated Annealing (scipy)
```python
from scipy.optimize import dual_annealing

def f(x):
    return (x[0]-1)**2 + (x[1]-2.5)**2

result = dual_annealing(f, bounds=[(-10,10), (-10,10)])
print(f"Optimal: {result.x}")  # Close to [1, 2.5]
```

### Particle Swarm (PySwarms)
```python
import pyswarms as ps
import numpy as np

def sphere(x):
    return np.sum(x**2, axis=1)

options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2, 
                                     options=options)
best_cost, best_pos = optimizer.optimize(sphere, iters=100)
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | Essentials of Metaheuristics | [Free PDF](https://cs.gmu.edu/~sean/book/metaheuristics/) |
| 📄 | NSGA-II Paper | [IEEE](https://ieeexplore.ieee.org/document/996017) |
| 📄 | Kirkpatrick et al. (1983) | Original SA paper |
| 🛠️ | DEAP (Python GA) | [GitHub](https://github.com/DEAP/deap) |
| 🛠️ | Optuna (Hyperparameter) | [Link](https://optuna.org/) |
| 🛠️ | PySwarms | [Link](https://pyswarms.readthedocs.io/) |
| 📄 | NEAT Paper | [Paper](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf) |
| 🇨🇳 | 知乎 遗传算法 | [知乎](https://zhuanlan.zhihu.com/p/28328304) |
| 🇨🇳 | B站 进化算法 | [B站](https://www.bilibili.com/video/BV1Eb411t7Mc) |
| 🇨🇳 | 模拟退火算法详解 | [知乎](https://zhuanlan.zhihu.com/p/33184423) |

---

## 🔗 Where This Topic Is Used

| Application | Usage |
|-------------|-------|
| **AutoML** | Neural architecture search |
| **Robotics** | Motion planning |
| **Finance** | Portfolio optimization |
| **Engineering** | Design optimization |
| **Logistics** | Route planning |

---

⬅️ [Back: Machine Learning](../08_machine_learning/)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=FF6B6B&height=80&section=footer" width="100%"/>
</p>
