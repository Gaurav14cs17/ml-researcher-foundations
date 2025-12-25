<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=FF6B6B&height=120&section=header&text=Metaheuristics&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-04-FF6B6B?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

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

## 📂 Topics in This Folder

| Folder | Topic | Best For |
|--------|-------|----------|
| [genetic-algorithm/](./genetic-algorithm/) | Genetic Algorithms | Discrete, combinatorial |
| [simulated-annealing/](./simulated-annealing/) | Simulated Annealing | Avoiding local minima |
| [multi-objective/](./multi-objective/) | Multi-Objective Optimization | Pareto fronts |
| [particle-swarm/](./particle-swarm/) | Particle Swarm | Continuous, swarm |

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

## 📚 Resources

| Type | Title | Link |
|------|-------|------|
| 📖 | Essentials of Metaheuristics | [Free PDF](https://cs.gmu.edu/~sean/book/metaheuristics/) |
| 🛠️ | DEAP (Python GA) | [GitHub](https://github.com/DEAP/deap) |
| 🛠️ | Optuna (Hyperparameter) | [Link](https://optuna.org/) |
| 📄 | NEAT Paper | [Paper](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf) |
| 🇨🇳 | 知乎 遗传算法 | [知乎](https://zhuanlan.zhihu.com/p/28328304) |
| 🇨🇳 | B站 进化算法 | [B站](https://www.bilibili.com/video/BV1Eb411t7Mc) |

---

⬅️ [Back: Machine Learning](../08-machine-learning/)

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
