<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=F39C12&height=120&section=header&text=Exploration&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-07-F39C12?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/exploration-vs-exploitation.svg" width="100%">

*Caption: The exploration-exploitation trade-off is fundamental to RL. Explore too much and you waste time; exploit too much and you miss better options. Different methods (ε-greedy, UCB, curiosity) balance this differently.*

---

## 📐 Mathematical Foundations

### ε-Greedy
```
π(a|s) = {
  1 - ε + ε/|A|  if a = argmax Q(s,a)
  ε/|A|          otherwise
}

ε typically decays: εₜ = ε₀ × decay^t
```

### Upper Confidence Bound (UCB)
```
UCB1: a = argmax[Q(s,a) + c√(ln(t)/N(s,a))]

Where:
• Q(s,a) = estimated value (exploitation)
• √(ln(t)/N(s,a)) = uncertainty bonus (exploration)
• c = exploration coefficient
```

### Entropy Regularization
```
Objective: J(π) = E[Σₜ rₜ + α H(π(·|sₜ))]

Where H(π(·|s)) = -Σₐ π(a|s) log π(a|s)

Maximum entropy RL encourages exploration
```

### Intrinsic Curiosity
```
ICM bonus: rᵢ = ||ŝ' - s'||²  (prediction error)

RND bonus: rᵢ = ||f(s') - f̂(s')||²
Where f is random fixed, f̂ is learned
```

---

## 📂 Topics

| Folder | Topic | Key Idea |
|--------|-------|----------|
| [epsilon-greedy/](./epsilon-greedy/) | ε-greedy | Random with prob ε |
| [ucb/](./ucb/) | Upper Confidence Bound | Optimism |
| [intrinsic/](./intrinsic/) | Intrinsic motivation | Curiosity |
| [curiosity/](./curiosity/) | ICM, RND | Prediction error |

---

## 🎯 The Dilemma

```
Exploration: Try new things to learn
Exploitation: Use what you know to get reward

Too much exploration → Never get good rewards
Too little → Miss better options
```

---

## 📊 Methods Comparison

| Method | Pros | Cons |
|--------|------|------|
| **ε-greedy** | Simple | Not directed |
| **UCB** | Principled | Requires counts |
| **Curiosity** | Works on hard exploration | May get stuck |
| **Entropy bonus** | Smooth exploration | Hyperparameter |

---

## 🔗 Where This Topic Is Used

| Topic | How Exploration Is Used |
|-------|------------------------|
| **DQN** | ε-greedy action selection |
| **PPO** | Entropy bonus in objective |
| **A/B Testing** | UCB, Thompson sampling |
| **Recommender Systems** | Exploration-exploitation tradeoff |
| **AlphaGo** | MCTS exploration (UCB) |
| **Curiosity-driven RL** | ICM, RND for sparse rewards |
| **Safe RL** | Constrained exploration |
| **Multi-armed Bandits** | UCB, Thompson sampling |

### Exploration In Famous Systems

| System | Exploration Method |
|--------|-------------------|
| **DQN (Atari)** | ε-greedy |
| **AlphaGo** | UCB in MCTS |
| **OpenAI Five** | Entropy bonus |
| **Montezuma** | Curiosity (ICM/RND) |
| **Ad systems** | Thompson sampling |

### Prerequisite For

```
Exploration --> Any RL algorithm
           --> Bandit algorithms
           --> MCTS (Monte Carlo Tree Search)
           --> Safe reinforcement learning
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | Sutton & Barto Ch. 2 | [RL Book](http://incompleteideas.net/book/) |
| 📄 | ICM Paper | [arXiv](https://arxiv.org/abs/1705.05363) |
| 📄 | RND Paper | [arXiv](https://arxiv.org/abs/1810.12894) |
| 🇨🇳 | 探索与利用详解 | [知乎](https://zhuanlan.zhihu.com/p/32356077) |
| 🇨🇳 | 好奇心驱动探索 | [CSDN](https://blog.csdn.net/qq_30615903/article/details/80952771) |
| 🇨🇳 | 强化学习探索策略 | [B站](https://www.bilibili.com/video/BV1yp4y1s7Qw) |

---

⬅️ [Back: 03-Policy Methods](../03-policy-methods/) | ➡️ [Next: 05-Model-Based](../05-model-based/)


---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=F39C12&height=80&section=footer" width="100%"/>
</p>
