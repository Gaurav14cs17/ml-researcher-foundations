<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=FF6B6B&height=120&section=header&text=Genetic%20Algorithms&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-04-FF6B6B?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Core Concept

**Genetic Algorithms (GA)** are evolutionary algorithms that mimic biological evolution to solve optimization problems. They maintain a population of candidate solutions and iteratively evolve better solutions through selection, crossover, and mutation.

---

## 📐 Key Components

### 1. **Representation (Chromosome)**
```
Solution encoded as "genes"
Examples:
• Binary string: 101101001
• Real vector: [2.3, -1.5, 0.8]
• Permutation: [3, 1, 4, 2]
```

### 2. **Fitness Function**
```
Evaluates quality of a solution
f(x) → ℝ  (higher = better)

Examples:
• Minimize distance → fitness = -distance
• Maximize accuracy → fitness = accuracy
```

### 3. **Selection**
```
Choose parents for breeding
Methods:
• Roulette wheel: P(select) ∝ fitness
• Tournament: Best of k random
• Rank-based: Sort by fitness
```

### 4. **Crossover (Recombination)**
```
Combine two parents → offspring

Single-point:
Parent 1: 11011|001
Parent 2: 01101|110
         -----+---
Child 1:  11011|110
Child 2:  01101|001
```

### 5. **Mutation**
```
Random changes to maintain diversity

Binary: Flip bit with probability p
Real: Add Gaussian noise
Permutation: Swap two positions
```

---

## 📐 Algorithm

```python

# Initialize
population = random_population(size=N)

for generation in range(max_generations):

    # 1. Evaluate
    fitness = [evaluate(individual) for individual in population]
    
    # 2. Check termination
    if best(fitness) > threshold:
        return best_individual
    
    # 3. Selection
    parents = select(population, fitness, method='tournament')
    
    # 4. Crossover
    offspring = []
    for i in range(0, len(parents), 2):
        child1, child2 = crossover(parents[i], parents[i+1], rate=0.8)
        offspring.extend([child1, child2])
    
    # 5. Mutation
    offspring = [mutate(child, rate=0.01) for child in offspring]
    
    # 6. Replace
    population = select_survivors(population + offspring, size=N)

return best(population)
```

---

## 💻 Implementation

```python
import numpy as np

class GeneticAlgorithm:
    def __init__(self, pop_size=100, gene_length=10, 
                 crossover_rate=0.8, mutation_rate=0.01):
        self.pop_size = pop_size
        self.gene_length = gene_length
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
    
    def fitness(self, individual):
        """Maximize: number of 1s (example)"""
        return np.sum(individual)
    
    def select_parents(self, population, fitness):
        """Tournament selection"""
        k = 3  # Tournament size
        parents = []
        for _ in range(self.pop_size):
            tournament = np.random.choice(len(population), k)
            winner = tournament[np.argmax(fitness[tournament])]
            parents.append(population[winner])
        return parents
    
    def crossover(self, parent1, parent2):
        """Single-point crossover"""
        if np.random.rand() < self.crossover_rate:
            point = np.random.randint(1, self.gene_length)
            child1 = np.concatenate([parent1[:point], parent2[point:]])
            child2 = np.concatenate([parent2[:point], parent1[point:]])
            return child1, child2
        return parent1.copy(), parent2.copy()
    
    def mutate(self, individual):
        """Bit-flip mutation"""
        for i in range(len(individual)):
            if np.random.rand() < self.mutation_rate:
                individual[i] = 1 - individual[i]
        return individual
    
    def evolve(self, generations=100):

        # Initialize
        population = [np.random.randint(0, 2, self.gene_length) 
                     for _ in range(self.pop_size)]
        
        for gen in range(generations):

            # Evaluate
            fitness = [self.fitness(ind) for ind in population]
            
            # Track best
            best_idx = np.argmax(fitness)
            print(f"Gen {gen}: Best = {fitness[best_idx]}")
            
            # Selection
            parents = self.select_parents(population, fitness)
            
            # Crossover & Mutation
            offspring = []
            for i in range(0, len(parents), 2):
                child1, child2 = self.crossover(parents[i], parents[i+1])
                offspring.append(self.mutate(child1))
                offspring.append(self.mutate(child2))
            
            population = offspring[:self.pop_size]
        
        return max(population, key=self.fitness)

# Usage
ga = GeneticAlgorithm(pop_size=50, gene_length=20)
best = ga.evolve(generations=50)
print(f"Best solution: {best}")
```

---

## 📊 Example Problems

### 1. **Traveling Salesman Problem (TSP)**
```python

# Chromosome: Permutation of cities [3,1,4,2,5]
# Fitness: -total_distance
# Crossover: Order crossover (preserve relative order)
# Mutation: Swap two cities
```

### 2. **Function Optimization**
```python

# Chromosome: Real vector [x₁, x₂, ..., xₙ]
# Fitness: f(x)
# Crossover: Arithmetic (0.5*parent1 + 0.5*parent2)
# Mutation: Gaussian noise
```

### 3. **Neural Network Architecture Search**
```python

# Chromosome: [layers, neurons, activation, ...]
# Fitness: Validation accuracy
# Crossover: Mix architectures
# Mutation: Change layer size/type
```

---

## 🔑 Key Parameters

| Parameter | Typical Range | Effect |
|-----------|---------------|--------|
| **Population Size** | 50-500 | More = better exploration |
| **Crossover Rate** | 0.6-0.9 | Too high = lose good solutions |
| **Mutation Rate** | 0.001-0.1 | Too high = random search |
| **Generations** | 50-1000 | Problem dependent |

---

## 🌍 Advantages & Limitations

### ✅ Advantages
```
• No gradient needed
• Handles discrete/combinatorial problems
• Parallelizable
• Explores diverse solutions
• Can escape local optima
```

### ❌ Limitations
```
• Slow convergence
• Many hyperparameters
• No convergence guarantees
• Expensive function evaluations
• Not suitable for high-dimensional continuous problems
```

---

## 🔄 Variants

### **Evolution Strategies (ES)**
```
• Focus on mutation, not crossover
• Self-adaptive parameters
• Used in RL (OpenAI ES)
```

### **Genetic Programming (GP)**
```
• Evolve programs/trees
• Variable-length chromosomes
• Symbolic regression
```

### **Differential Evolution**
```
• For continuous optimization
• Mutation: xᵢ + F(xⱼ - xₖ)
• Often outperforms GA on continuous problems
```

---

## 🌍 Applications

| Domain | Application | Notes |
|--------|-------------|-------|
| **Scheduling** | Job shop, timetabling | Permutation encoding |
| **Design** | Antenna, circuits | Real-valued encoding |
| **ML** | Hyperparameter tuning | Mixed encoding |
| **Games** | Strategy evolution | Tree encoding |
| **Robotics** | Gait optimization | Real vectors |

---

## 📚 Resources

### Books
- **Introduction to Genetic Algorithms** - Sivanandam & Deepa
- **Genetic Algorithms in Search** - Goldberg (1989)
- **An Introduction to Evolutionary Algorithms** - Eiben & Smith

### Papers
- Holland (1975) - Adaptation in Natural and Artificial Systems
- Goldberg (1989) - Classic GA reference
- Salimans et al. (2017) - Evolution Strategies as RL

### Software
- **DEAP** (Python): Distributed EA framework
- **PyGAD**: Simple GA library
- **pymoo**: Multi-objective optimization

---

⬅️ [Back: Metaheuristics](../) | ⬆️ [Up: Optimization](../../)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=FF6B6B&height=80&section=footer" width="100%"/>
</p>
