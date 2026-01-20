<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=F39C12&height=120&section=header&text=ε-Greedy%20Exploration&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-07-F39C12?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## 🔗 Navigation

⬅️ [Back: Curiosity](../01_curiosity/) | ➡️ [Next: Intrinsic](../03_intrinsic/)

---

## 🎯 Visual Overview

<img src="./images/epsilon-greedy.svg" width="100%">

*Caption: ε-greedy explores by taking random actions with probability ε, and exploits by taking the best known action with probability 1-ε. Typically ε decays over time: explore early, exploit later.*

---

## 📂 Overview

ε-greedy is the most common exploration strategy in RL due to its simplicity. It provides a tunable trade-off between exploration and exploitation.

---

## 📐 Mathematical Definition

### Action Selection

```
       ⎧ random action from A    with probability ε
a_t = ⎨
       ⎩ argmax_a Q(s_t, a)      with probability 1-ε

Where:
    ε ∈ [0, 1]: Exploration rate
    A: Action space
    Q(s, a): Estimated action-value function
```

### Expected Behavior

```
For |A| actions:

P(select optimal action) = (1 - ε) + ε/|A|
P(select random action) = ε × (|A|-1)/|A|

Example: |A| = 4, ε = 0.1
    P(optimal) = 0.9 + 0.1/4 = 0.925
    P(each non-optimal) = 0.1 × 3/4 × 1/3 = 0.025
```

---

## 📐 Decay Schedules

### Linear Decay

```
ε_t = max(ε_min, ε_0 - t × decay_rate)

Example: ε_0 = 1.0, ε_min = 0.01, decay_rate = 0.001
    t=0:    ε = 1.0
    t=500:  ε = 0.5
    t=990:  ε = 0.01 (minimum)
```

### Exponential Decay

```
ε_t = max(ε_min, ε_0 × decay^t)

Example: ε_0 = 1.0, decay = 0.995
    t=0:    ε = 1.0
    t=100:  ε ≈ 0.606
    t=500:  ε ≈ 0.082
```

### Inverse Decay

```
ε_t = 1 / (1 + t × k)

More exploration-heavy at start, slower decay
```

---

## 📐 Theoretical Analysis

### GLIE Condition

```
Theorem (Greedy in the Limit with Infinite Exploration):

Q-learning with ε-greedy converges to Q* if:

1. All state-action pairs are visited infinitely often:
   lim_{t→∞} N_t(s,a) = ∞  ∀s,a

2. Policy becomes greedy in the limit:
   lim_{t→∞} π_t(a|s) = 1  for a = argmax Q(s,a)

Sufficient condition: Σ_t ε_t = ∞ and Σ_t ε_t² < ∞

Example satisfying GLIE: ε_t = 1/t
```

### Regret Analysis

```
Definition: Regret R_T = Σ_{t=1}^T (r* - r_t)
  Where r* = optimal expected reward

For ε-greedy with ε_t = c/t on K-armed bandit:

  E[R_T] = O(K log T / Δ)
  
  Where Δ = min gap between optimal and suboptimal

Comparison:
  - ε-greedy: O(K log T / Δ)
  - UCB: O(K log T / Δ)  (similar, but directed exploration)
  - Thompson: O(K log T / Δ)  (Bayesian optimal)
```

### Probability of Optimal Selection

```
After t steps with ε_t = 1/t:

P(select optimal action) = (1 - 1/t) + 1/(t·|A|)
                        ≈ 1 - (|A|-1)/(t·|A|)  for large t

Convergence: P → 1 as t → ∞

Time to 99% exploitation (ε ≤ 0.01):
  For ε_t = 1/t: t = 100 steps
  For ε_t = ε₀·0.99^t: t ≈ 459 steps (if ε₀=1)
```

---

## 🔑 Key Concepts

| Concept | Description |
|---------|-------------|
| **ε = 1.0** | Pure exploration (all random) |
| **ε = 0.0** | Pure exploitation (all greedy) |
| **ε-decay** | Reduce ε over time (explore→exploit) |
| **ε-first** | Explore ε% of time, then exploit forever |
| **Annealing** | Smooth transition from explore to exploit |

---

## 📊 Comparison with Other Strategies

| Strategy | Exploration | Optimal? | Complexity |
|----------|-------------|----------|------------|
| **ε-greedy** | Random | Suboptimal | O(1) |
| **Softmax** | Probability ∝ Q | Better | O(\|A\|) |
| **UCB** | Uncertainty-based | Near-optimal | O(\|A\|) |
| **Thompson** | Bayesian sampling | Optimal | O(\|A\|) |

```
ε-greedy Limitation:

    Q(a₁) = 10 ± 0.1   ← Very confident
    Q(a₂) = 9  ± 5.0   ← Very uncertain
    
ε-greedy: Explores both equally (random)
UCB/Thompson: Explores a₂ more (might be better!)
```

---

## 💻 Code Examples

### Basic Implementation

```python
import numpy as np

def epsilon_greedy(q_values, epsilon):
    """
    Select action using epsilon-greedy policy
    
    Args:
        q_values: Q(s, a) for all actions
        epsilon: Exploration probability
    
    Returns:
        Selected action index
    """
    if np.random.random() < epsilon:
        return np.random.randint(len(q_values))  # Explore
    else:
        return np.argmax(q_values)  # Exploit
```

### With Decay Schedules

```python
class EpsilonScheduler:
    """Different epsilon decay strategies"""
    
    def __init__(self, start=1.0, end=0.01, decay_type='exponential'):
        self.epsilon = start
        self.start = start
        self.end = end
        self.decay_type = decay_type
        self.step = 0
    
    def get_epsilon(self):
        return self.epsilon
    
    def update(self, decay_param=0.995):
        self.step += 1
        
        if self.decay_type == 'exponential':
            self.epsilon = max(self.end, self.epsilon * decay_param)
        
        elif self.decay_type == 'linear':
            self.epsilon = max(self.end, self.start - self.step * decay_param)
        
        elif self.decay_type == 'inverse':
            self.epsilon = max(self.end, 1.0 / (1 + self.step * decay_param))
        
        return self.epsilon

# Usage in training loop
scheduler = EpsilonScheduler(start=1.0, end=0.01, decay_type='exponential')

for episode in range(1000):
    state = env.reset()
    epsilon = scheduler.get_epsilon()
    
    while not done:
        q_values = model(state)
        action = epsilon_greedy(q_values, epsilon)
        # ... take action, train model ...
    
    scheduler.update(decay_param=0.995)
```

### With Q-Learning

```python
def q_learning_with_epsilon_greedy(env, num_episodes, 
                                    alpha=0.1, gamma=0.99,
                                    epsilon_start=1.0, epsilon_end=0.01):
    """
    Q-Learning with decaying epsilon-greedy exploration
    """
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    epsilon = epsilon_start
    decay = (epsilon_start - epsilon_end) / num_episodes
    
    rewards_history = []
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # ε-greedy action selection
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])
            
            next_state, reward, done, _ = env.step(action)
            
            # Q-learning update
            td_target = reward + gamma * np.max(Q[next_state]) * (1 - done)
            Q[state, action] += alpha * (td_target - Q[state, action])
            
            state = next_state
            total_reward += reward
        
        # Decay epsilon
        epsilon = max(epsilon_end, epsilon - decay)
        rewards_history.append(total_reward)
    
    return Q, rewards_history
```

---

## 📊 Hyperparameter Guidelines

| Scenario | ε_start | ε_end | Decay |
|----------|---------|-------|-------|
| **Small env** | 1.0 | 0.01 | Fast (0.99) |
| **Large env** | 1.0 | 0.05 | Slow (0.999) |
| **DQN (Atari)** | 1.0 | 0.1 | Linear over 1M |
| **Bandits** | 0.1 | 0.01 | Very slow |

---

## 🔗 Connection to Other Strategies

```
ε-Greedy (simplest)
    |
    +-- Softmax/Boltzmann (smooth probability)
    |       a ~ exp(Q(s,a)/τ) / Σ exp(Q(s,a')/τ)
    |
    +-- UCB (uncertainty bonus)
    |       a = argmax Q(s,a) + c√(log t / N(a))
    |
    +-- Thompson Sampling (Bayesian)
            Sample Q ~ posterior, act greedily on sample
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | UCB Exploration | [../04_ucb/](../04_ucb/) |
| 📖 | Curiosity-Driven | [../01_curiosity/](../01_curiosity/) |
| 📖 | Sutton & Barto Ch. 2 | [RL Book](http://incompleteideas.net/book/) |
| 🎥 | David Silver Lecture 9 | [YouTube](https://www.youtube.com/watch?v=sGuiWX07sKw) |
| 🇨🇳 | 探索与利用权衡 | [知乎](https://zhuanlan.zhihu.com/p/32356077) |
| 🇨🇳 | DQN中的ε-greedy | [CSDN](https://blog.csdn.net/qq_30615903/article/details/80952771) |
| 🇨🇳 | 强化学习探索策略 | [B站](https://www.bilibili.com/video/BV1yp4y1s7Qw) |
| 🇨🇳 | 多臂老虎机问题 | [机器之心](https://www.jiqizhixin.com/articles/2017-08-14-2)

## 🔗 Where This Topic Is Used

| Application | ε-greedy |
|-------------|---------|
| **DQN** | Standard exploration |
| **Bandits** | Simple exploration |
| **A/B Testing** | Exploration-exploitation |
| **Recommendation** | Random recommendations |

---

⬅️ [Back: Curiosity](../01_curiosity/) | ➡️ [Next: Intrinsic](../03_intrinsic/)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=F39C12&height=80&section=footer" width="100%"/>
</p>
