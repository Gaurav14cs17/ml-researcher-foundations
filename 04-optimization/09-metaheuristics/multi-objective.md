<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=MultiObjective%20Optimization&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---

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

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>
