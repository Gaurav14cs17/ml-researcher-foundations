# Multi-Objective Optimization

## Overview

Multiple competing objectives. Find Pareto frontier of trade-offs.

## Key Formula

```
Problem:
min [f₁(x), f₂(x), ..., fₖ(x)]

Pareto Dominance:
x dominates y if:
  fᵢ(x) ≤ fᵢ(y) ∀i AND
  fⱼ(x) < fⱼ(y) for some j

Weighted Sum:
min Σwᵢfᵢ(x), wᵢ ≥ 0

ε-Constraint:
min f₁(x) s.t. fᵢ(x) ≤ εᵢ
```

## Key Concepts

- **Pareto Dominance** - Better in all, strictly better in one
- **Pareto Frontier** - Set of non-dominated solutions
- **Weighted Sum** - Scalarize objectives
- **NSGA-II** - Evolutionary multi-objective

---

---

➡️ [Next: Simulated Annealing](./simulated-annealing.md)
