# 🌍 Model-Based RL

> **Learning and using environment models**

---

## 🎯 Visual Overview

<img src="./images/world-models.svg" width="100%">

*Caption: Model-based RL learns a world model (dynamics p(s'|s,a) and reward r(s,a)) from real experience, then plans or trains in imagination. This is how AlphaGo, MuZero, and Dreamer achieve sample efficiency.*

---

## 📐 Mathematical Foundations

### World Model
```
Learn: p̂(s'|s,a) and r̂(s,a)

From data: D = {(sₜ, aₜ, rₜ, sₜ₊₁)}

Minimize: L = E[||ŝ' - s'||² + (r̂ - r)²]
```

### Model Predictive Control (MPC)
```
At each step:
1. Plan: πₜ = argmax E[Σᵢ₌₀ᴴ γⁱ r̂(sₜ₊ᵢ, aₜ₊ᵢ)]
   using model p̂(s'|s,a)
2. Execute: Take action aₜ
3. Re-plan with new observation
```

### MCTS (UCT formula)
```
UCT score = Q(s,a)/N(s,a) + c√(ln N(s)/N(s,a))

Selection → Expansion → Simulation → Backpropagation
```

### Dreamer (RSSM)
```
Recurrent State-Space Model:
hₜ = f(hₜ₋₁, zₜ₋₁, aₜ₋₁)  (deterministic)
zₜ ~ q(zₜ|hₜ, oₜ)          (stochastic)

Learn in imagination:
Roll out latent trajectories, train actor-critic
```

---

## 📂 Topics

| Folder | Topic | Key Idea |
|--------|-------|----------|
| [world-models/](./world-models/) | Learn dynamics | p(s'\|s,a) |
| [planning/](./planning/) | Use model to plan | MPC |
| [mcts/](./mcts/) | Monte Carlo Tree Search | AlphaGo |
| [dreamer/](./dreamer/) | Imagination | Dream to learn |

---

## 🎯 Key Idea

```
Model-Free: Learn policy directly from experience
Model-Based: Learn model, then plan/imagine

Advantages:
• Sample efficient (fewer environment interactions)
• Can plan ahead
• Transfer to new tasks

Disadvantages:
• Model errors compound
• More computation
```

---

## 🔥 Notable Systems

| System | Method | Achievement |
|--------|--------|-------------|
| **AlphaGo** | MCTS + NN | Beat world champion |
| **MuZero** | Learned model + MCTS | Master multiple games |
| **Dreamer** | World model + imagination | Efficient learning |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | Dreamer Paper | [arXiv](https://arxiv.org/abs/1912.01603) |
| 📄 | MuZero Paper | [Nature](https://www.nature.com/articles/s41586-020-03051-4) |
| 📄 | World Models | [arXiv](https://arxiv.org/abs/1803.10122) |
| 🇨🇳 | 模型基强化学习 | [知乎](https://zhuanlan.zhihu.com/p/563656219) |
| 🇨🇳 | Dreamer详解 | [CSDN](https://blog.csdn.net/qq_37006625/article/details/123629543) |
| 🇨🇳 | AlphaGo原理 | [B站](https://www.bilibili.com/video/BV1C34y1H7Eq) |


## 🔗 Where This Topic Is Used

| Application | Model-Based RL |
|-------------|---------------|
| **Sample Efficiency** | Fewer real interactions |
| **Robotics** | Safe exploration |
| **Games** | AlphaZero, MuZero |
| **Planning** | MPC, trajectory opt |

---

⬅️ [Back: 04-Exploration](../04-exploration/) | ➡️ [Next: 06-Applications](../06-applications/)


