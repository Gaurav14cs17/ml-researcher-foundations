<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=F39C12&height=120&section=header&text=Upper%20Confidence%20Bound%20(UCB)&fontSize=28&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-07-F39C12?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## 🔗 Navigation

⬅️ [Back: Intrinsic](../03_intrinsic/) | ➡️ [Next: Model-Based](../../05_model_based/)

---

## 🎯 Visual Overview

<img src="./images/ucb.svg" width="100%">

*Caption: UCB selects actions by adding an exploration bonus to the estimated value. Actions tried less often have higher uncertainty and thus higher bonus. This guarantees exploration of all actions.*

---

## 📂 Overview

Upper Confidence Bound (UCB) is a principled exploration strategy from bandit theory. It implements the principle of "optimism in the face of uncertainty" - be optimistic about actions you haven't tried much.

---

## 📐 Mathematical Foundation

### The Multi-Armed Bandit Problem

```
Setup:
• K actions (arms) with unknown reward distributions
• At each round t, select action a_t ∈ {1, ..., K}
• Receive reward r_t ~ P_a with mean μ_a

Objective:
Minimize regret: R_T = Σ_{t=1}^T (μ* - μ_{a_t})
where μ* = max_a μ_a
```

### UCB1 Algorithm

```
UCB(a) = Q̂(a) + c · √(ln(t) / N(a))
         ↑          ↑
    exploitation   exploration bonus

Select: a_t = argmax_a UCB(a)

Where:
• Q̂(a) = (1/N(a)) Σ r_i for action a (sample mean)
• N(a) = number of times action a was selected
• t = total time steps
• c = exploration coefficient (typically √2)
```

---

## 📐 Theoretical Analysis

### Hoeffding's Inequality

The exploration bonus is derived from concentration inequalities:

```
Theorem (Hoeffding's Bound):
For i.i.d. random variables X_1, ..., X_n ∈ [0,1] with mean μ:

P(|X̄_n - μ| > ε) ≤ 2 exp(-2nε²)

Setting the RHS to 1/t² and solving for ε:
ε = √(ln(t) / (2n))
```

### Upper Confidence Bound Derivation

```
With probability ≥ 1 - 1/t², the true mean μ_a lies in the interval:

[Q̂(a) - √(ln(t)/(2N(a))), Q̂(a) + √(ln(t)/(2N(a)))]

UCB uses the upper bound: UCB(a) = Q̂(a) + √(2ln(t)/N(a))

The factor √2 ensures high probability bounds.
```

### Regret Bound

```
Theorem: UCB1 achieves logarithmic regret:

R_T ≤ [8 Σ_{a:μ_a < μ*} (ln T)/Δ_a] + (1 + π²/3) Σ_a Δ_a

Where Δ_a = μ* - μ_a is the suboptimality gap.

This is asymptotically optimal! No algorithm can do better than O(ln T).
```

### Proof Sketch

```
Key insight: An arm a with Δ_a > 0 can only be pulled if either:

1. The estimate Q̂(a) is too optimistic (concentration failure)
2. The estimate Q̂(*) for the best arm is too pessimistic
3. The exploration bonus is still large

The probability of (1) and (2) decreases with t.
(3) happens at most O(ln T / Δ_a²) times.
```

---

## 📐 UCB Variants

### UCB1-Tuned

Uses variance estimates for tighter bounds:

```
UCB1-Tuned(a) = Q̂(a) + √(ln(t)/N(a) · min(1/4, V(a)))

Where V(a) = σ̂²(a) + √(2ln(t)/N(a)) is the empirical variance + exploration
```

### UCB-V (Variance-Aware)

```
UCB-V(a) = Q̂(a) + √(2σ̂²(a)ln(t)/N(a)) + 3b·ln(t)/N(a)

Better in high-variance settings
```

### KL-UCB

Uses KL divergence for Bernoulli bandits:

```
KL-UCB(a) = max{q ∈ [0,1] : N(a)·KL(Q̂(a), q) ≤ ln(t)}

Where KL(p,q) = p·ln(p/q) + (1-p)·ln((1-p)/(1-q))
```

### LinUCB (Contextual)

For linear bandits with context x:

```
UCB(a) = θ̂_a^T x + α√(x^T A_a^{-1} x)

Where:
• θ̂_a = A_a^{-1} b_a (ridge regression estimate)
• A_a = Σ x_t x_t^T + λI
• b_a = Σ r_t x_t
```

---

## 📐 UCB for MDPs: UCB-VI

For MDPs with unknown dynamics, UCB-VI combines value iteration with exploration bonuses:

```
Algorithm: UCB-VI

Initialize: Q(s,a) = H, N(s,a) = 0 for all s,a

For episode k = 1, 2, ...:
    For step h = H, H-1, ..., 1:
        For all (s,a):
            Bonus: b(s,a) = c·√(H/N(s,a))
            Q(s,a) ← min(H, r(s,a) + P̂(s'|s,a)·V(s') + b(s,a))
            V(s) ← max_a Q(s,a)
    
    Execute policy π(s) = argmax_a Q(s,a)
    Update counts N(s,a) and transition estimates P̂
```

---

## 📐 UCB in MCTS (AlphaGo)

UCB for Trees (UCT) is used in Monte Carlo Tree Search:

```
UCT(s, a) = Q(s, a)/N(s, a) + c·√(ln N(s) / N(s, a))

In AlphaGo, this is modified with a policy prior p(a|s):

PUCT(s, a) = Q(s, a) + c·p(a|s)·√N(s)/(1 + N(s, a))

The policy network guides exploration toward promising actions.
```

---

## 💻 Complete Implementation

```python
import numpy as np
from typing import List, Tuple

class UCBBandit:
    """UCB1 algorithm for multi-armed bandits"""
    
    def __init__(self, n_arms: int, c: float = 2.0):
        self.n_arms = n_arms
        self.c = c
        
        self.counts = np.zeros(n_arms)  # N(a)
        self.values = np.zeros(n_arms)  # Q̂(a)
        self.t = 0
    
    def select_action(self) -> int:
        self.t += 1
        
        # Play each arm once first
        if self.t <= self.n_arms:
            return self.t - 1
        
        # UCB action selection
        ucb_values = self.values + self.c * np.sqrt(np.log(self.t) / self.counts)
        return np.argmax(ucb_values)
    
    def update(self, action: int, reward: float):
        """Update estimates after receiving reward"""
        self.counts[action] += 1
        n = self.counts[action]
        
        # Incremental mean update
        self.values[action] += (reward - self.values[action]) / n
    
    def get_ucb_values(self) -> np.ndarray:
        """Return current UCB values for all arms"""
        if self.t == 0:
            return np.inf * np.ones(self.n_arms)
        return self.values + self.c * np.sqrt(np.log(self.t) / (self.counts + 1e-8))

class UCBTuned(UCBBandit):
    """UCB1-Tuned with variance estimation"""
    
    def __init__(self, n_arms: int, c: float = 2.0):
        super().__init__(n_arms, c)
        self.sum_squares = np.zeros(n_arms)  # For variance
    
    def update(self, action: int, reward: float):
        self.counts[action] += 1
        n = self.counts[action]
        
        # Update mean
        delta = reward - self.values[action]
        self.values[action] += delta / n
        
        # Update sum of squares for variance
        self.sum_squares[action] += delta * (reward - self.values[action])
    
    def get_variance(self) -> np.ndarray:
        """Empirical variance estimate"""
        with np.errstate(divide='ignore', invalid='ignore'):
            var = self.sum_squares / (self.counts - 1)
            var = np.nan_to_num(var, nan=0.25, posinf=0.25)
        return var
    
    def select_action(self) -> int:
        self.t += 1
        
        if self.t <= self.n_arms:
            return self.t - 1
        
        # UCB-Tuned
        variance = self.get_variance()
        exploration = np.sqrt(np.log(self.t) / self.counts)
        V = np.minimum(0.25, variance + np.sqrt(2 * np.log(self.t) / self.counts))
        ucb_values = self.values + np.sqrt(np.log(self.t) / self.counts * V)
        
        return np.argmax(ucb_values)

class LinUCB:
    """Contextual bandit with linear UCB"""
    
    def __init__(self, n_arms: int, d: int, alpha: float = 1.0, lambda_: float = 1.0):
        self.n_arms = n_arms
        self.d = d
        self.alpha = alpha
        
        # Per-arm matrices
        self.A = [lambda_ * np.eye(d) for _ in range(n_arms)]
        self.b = [np.zeros(d) for _ in range(n_arms)]
    
    def select_action(self, context: np.ndarray) -> int:
        """Select action given context using UCB"""
        ucb_values = np.zeros(self.n_arms)
        
        for a in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            
            # UCB = θ^T x + α√(x^T A^{-1} x)
            exploitation = theta @ context
            exploration = self.alpha * np.sqrt(context @ A_inv @ context)
            ucb_values[a] = exploitation + exploration
        
        return np.argmax(ucb_values)
    
    def update(self, action: int, context: np.ndarray, reward: float):
        """Update model after observing reward"""
        self.A[action] += np.outer(context, context)
        self.b[action] += reward * context

def simulate_bandit(bandit, true_means: List[float], n_rounds: int) -> Tuple[List, List]:
    """Simulate bandit experiment"""
    rewards = []
    regrets = []
    best_mean = max(true_means)
    
    for t in range(n_rounds):
        action = bandit.select_action()
        reward = np.random.normal(true_means[action], 1.0)
        bandit.update(action, reward)
        
        rewards.append(reward)
        regrets.append(best_mean - true_means[action])
    
    return rewards, regrets

# Example usage
if __name__ == "__main__":
    # Multi-armed bandit example
    true_means = [0.2, 0.5, 0.3, 0.8, 0.4]  # Arm 3 is best
    n_rounds = 10000
    
    # UCB1
    ucb = UCBBandit(n_arms=5, c=2.0)
    rewards, regrets = simulate_bandit(ucb, true_means, n_rounds)
    
    print(f"Total regret: {sum(regrets):.2f}")
    print(f"Arm pulls: {ucb.counts}")
    print(f"Estimated values: {ucb.values}")
    print(f"Best arm: {np.argmax(ucb.values)} (true best: {np.argmax(true_means)})")
```

---

## 📊 UCB vs Other Strategies

| Strategy | Regret | Pros | Cons |
|----------|--------|------|------|
| **ε-greedy** | O(T) or O(log T) | Simple | Not adaptive |
| **UCB1** | O(log T) | Optimal, principled | Requires counts |
| **Thompson Sampling** | O(log T) | Bayesian, empirically strong | Prior needed |
| **EXP3** | O(√T) | Adversarial | Higher regret |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | UCB1 Paper | [Auer et al. 2002](https://link.springer.com/article/10.1023/A:1013689704352) |
| 📄 | LinUCB Paper | [Li et al. 2010](https://arxiv.org/abs/1003.0146) |
| 📖 | Bandit Algorithms | [Lattimore & Szepesvári](https://tor-lattimore.com/downloads/book/book.pdf) |
| 🎥 | UCB Lecture | [YouTube](https://www.youtube.com/watch?v=ItKmRqyqQ_Y) |
| 🇨🇳 | UCB算法详解 | [知乎](https://zhuanlan.zhihu.com/p/32356077) |
| 🇨🇳 | 多臂老虎机 | [B站](https://www.bilibili.com/video/BV1yp4y1s7Qw) |

## 🔗 Where This Topic Is Used

| Application | UCB |
|-------------|-----|
| **Multi-Armed Bandits** | Optimal exploration |
| **MCTS (AlphaGo)** | UCT for tree search |
| **Hyperparameter Tuning** | Bayesian optimization |
| **Clinical Trials** | Adaptive allocation |
| **Recommendation Systems** | Cold-start exploration |
| **A/B Testing** | Adaptive experiments |

---

⬅️ [Back: Intrinsic](../03_intrinsic/) | ➡️ [Next: Model-Based](../../05_model_based/)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=F39C12&height=80&section=footer" width="100%"/>
</p>
