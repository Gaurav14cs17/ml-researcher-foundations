<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=02 Value Methods&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---


# 📊 Value-Based Methods

> **Learning value functions to find optimal policies**

<img src="./images/bellman.svg" width="100%">

---

## 📂 Topics

| Folder | Topic | Key Concepts |
|--------|-------|--------------|
| [bellman/](./bellman/) | Bellman equations | V(s), Q(s,a) |
| [dynamic-programming/](./dynamic-programming/) | Policy/Value iteration | Known dynamics |
| [td-learning/](./td-learning/) | Temporal difference | TD(0), TD(λ) |
| [q-learning/](./q-learning/) | Q-learning | Off-policy TD |
| [dqn/](./dqn/) | Deep Q-Networks | Neural Q-function |

---

## 📐 Mathematical Foundations

### Bellman Expectation Equations
```
V^π(s) = Σ_a π(a|s) [R(s,a) + γ Σ_{s'} P(s'|s,a) V^π(s')]

Q^π(s,a) = R(s,a) + γ Σ_{s'} P(s'|s,a) Σ_{a'} π(a'|s') Q^π(s',a')
```

### Bellman Optimality Equations
```
V*(s) = max_a [R(s,a) + γ Σ_{s'} P(s'|s,a) V*(s')]

Q*(s,a) = R(s,a) + γ Σ_{s'} P(s'|s,a) max_{a'} Q*(s',a')
```

### TD Error
```
δₜ = rₜ + γV(sₜ₊₁) - V(sₜ)

TD(0) update:
V(sₜ) ← V(sₜ) + α δₜ

TD(λ) combines n-step returns:
G^λ = (1-λ) Σₙ₌₁^∞ λⁿ⁻¹ Gₜ:ₜ₊ₙ
```

---

## 🔑 Key Equations

```
Value function:
V(s) = E[Σₜ γᵗrₜ | s₀ = s]

Action-value (Q-function):
Q(s,a) = E[Σₜ γᵗrₜ | s₀ = s, a₀ = a]

Q-learning update:
Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
```

---

## 🔗 Where This Topic Is Used

| Topic | How Value Methods Are Used |
|-------|---------------------------|
| **DQN** | Neural network approximates Q(s,a) |
| **Actor-Critic** | Critic = Value function V(s) or Q(s,a) |
| **PPO** | Uses value function as baseline |
| **A3C / A2C** | Advantage = Q - V |
| **AlphaGo** | Value network estimates winning probability |
| **Model Predictive Control** | Value for planning |
| **Inverse RL** | Recover reward from value function |
| **Reward Shaping** | Potential-based using V(s) |

### Prerequisite For

```
Value Methods --> DQN, Rainbow
             --> Actor-Critic methods
             --> PPO (uses value baseline)
             --> Model-Based RL planning
```

### Used By These Papers

| Paper | How It Uses Value Methods |
|-------|--------------------------|
| DQN (2013) | Deep Q-learning for Atari |
| A3C (2016) | Async actor-critic with V(s) |
| PPO (2017) | Generalized advantage estimation |
| AlphaGo (2016) | Value network + MCTS |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | Sutton & Barto Ch. 6-8 | [RL Book](http://incompleteideas.net/book/) |
| 📄 | DQN Paper | [Nature](https://www.nature.com/articles/nature14236) |
| 🎥 | David Silver Lectures | [YouTube](https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ) |
| 🇨🇳 | 价值函数方法详解 | [知乎](https://zhuanlan.zhihu.com/p/26052182) |
| 🇨🇳 | Q-Learning到DQN | [CSDN](https://blog.csdn.net/qq_30615903/article/details/80739243) |
| 🇨🇳 | 强化学习基础 | [B站](https://www.bilibili.com/video/BV1sd4y167NS) |

---

⬅️ [Back: 01-MDP](../01-mdp/) | ➡️ [Next: 03-Policy Methods](../03-policy-methods/)


---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>
