<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=Simulated%20Annealing&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---

# Simulated Annealing

> **Probabilistic optimization inspired by metallurgical annealing**

---

## 🎯 Visual Overview

<img src="./images/metaheuristics.svg" width="100%">

*Caption: Simulated Annealing accepts worse solutions with decreasing probability, allowing escape from local minima while converging to global optimum as temperature decreases.*

---

## 📐 Mathematical Foundations

### Core Algorithm

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

### Metropolis-Hastings Criterion

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

### Cooling Schedules

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

### Convergence Theory

```
Theorem (Geman & Geman, 1984):
If T(k) ≥ c / log(k+1) for large enough c,
then P(x_k = x*) → 1 as k → ∞

In practice:
• Logarithmic too slow
• Geometric with reheating works well
• Final temperature should be small (10⁻⁶ to 10⁻³)
```

### Stationary Distribution

```
At fixed temperature T, SA is a Markov chain with 
stationary distribution:

π_T(x) = exp(-f(x)/T) / Z(T)

where Z(T) = Σₓ exp(-f(x)/T) (partition function)

As T → 0:
π₀(x) concentrates on global minima
```

---

## 💻 Code Implementation

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

# Example: Minimize Rastrigin function
def rastrigin(x):
    A = 10
    return A * len(x) + sum(xi**2 - A * np.cos(2 * np.pi * xi) for xi in x)

def neighbor(x, step=0.5):
    return x + np.random.uniform(-step, step, len(x))

x0 = np.random.uniform(-5, 5, 10)
x_best, f_best, _ = simulated_annealing(rastrigin, x0, neighbor)
print(f"Best found: f(x) = {f_best:.6f}")
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

## 📚 References

| Type | Resource | Link |
|------|----------|------|
| 📄 | Kirkpatrick et al. (1983) | Original SA paper |
| 📄 | Černý (1985) | Independent discovery |
| 📖 | Aarts & Korst | Simulated Annealing and Boltzmann Machines |
| 🎥 | MIT 6.046J | [SA Lecture](https://ocw.mit.edu/) |
| 🇨🇳 | 模拟退火算法详解 | [知乎](https://zhuanlan.zhihu.com/p/33184423) |
| 🇨🇳 | SA原理与实现 | [CSDN](https://blog.csdn.net/google19890102/article/details/45395257) |

---

---

⬅️ [Back: Multi Objective](./multi-objective.md)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>
