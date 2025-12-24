# 🎯 Markov Decision Processes (MDP)

> **The mathematical framework for sequential decision-making**

---

## 🎯 Visual Overview

<img src="./images/mdp-diagram.svg" width="100%">

*Caption: An MDP models sequential decision-making as a tuple (S, A, P, R, γ). At each step, the agent observes state s, takes action a, receives reward r, and transitions to next state s'. The Markov property means the future depends only on the current state.*

---

## 📂 Topics

| Folder | Topic | Key Concepts |
|--------|-------|--------------|
| [states-actions/](./states-actions/) | State & Action spaces | S, A definitions |
| [rewards/](./rewards/) | Reward function | r(s,a,s'), shaping |
| [dynamics/](./dynamics/) | Transition dynamics | P(s'\|s,a) |
| [discounting/](./discounting/) | Discount factor | γ, infinite horizon |

---

## 📐 MDP Definition

```
MDP = (S, A, P, R, γ)

S: State space     — What the agent observes
A: Action space    — What the agent can do
P: P(s'|s,a)       — Transition probability (dynamics)
R: r(s,a,s')       — Reward function
γ: Discount factor — γ ∈ [0,1], importance of future rewards
```

### The Markov Property

```
P(sₜ₊₁ | sₜ, aₜ, sₜ₋₁, aₜ₋₁, ..., s₀, a₀) = P(sₜ₊₁ | sₜ, aₜ)

"The future is independent of the past given the present"

This simplifies planning: we only need to know the current state!
```

---

## 🎯 Goal: Find Optimal Policy

```
Find policy π(a|s) that maximizes:

    V^π(s) = E_π[Σₜ₌₀^∞ γᵗ rₜ | s₀ = s]
    
    Expected discounted cumulative reward from state s

Optimal Value:
    V*(s) = max_π V^π(s)
    
Optimal Policy:
    π*(s) = argmax_a Q*(s, a)
```

---

## 📐 Key Equations

### Value Function (State Value)

```
V^π(s) = E_π[rₜ + γV^π(sₜ₊₁) | sₜ = s]

       = Σ_a π(a|s) Σ_s' P(s'|s,a) [R(s,a,s') + γV^π(s')]
```

### Q-Function (Action Value)

```
Q^π(s, a) = E_π[rₜ + γV^π(sₜ₊₁) | sₜ = s, aₜ = a]

          = Σ_s' P(s'|s,a) [R(s,a,s') + γV^π(s')]
```

### Bellman Optimality Equations

```
V*(s) = max_a Σ_s' P(s'|s,a) [R(s,a,s') + γV*(s')]

Q*(s,a) = Σ_s' P(s'|s,a) [R(s,a,s') + γ max_a' Q*(s',a')]
```

---

## 💻 Code: MDP Environment

```python
import numpy as np

class SimpleMDP:
    """A simple MDP implementation"""
    
    def __init__(self, n_states, n_actions):
        self.n_states = n_states
        self.n_actions = n_actions
        
        # Transition probabilities P(s'|s,a)
        self.P = np.random.rand(n_states, n_actions, n_states)
        self.P = self.P / self.P.sum(axis=2, keepdims=True)
        
        # Reward function R(s,a,s')
        self.R = np.random.randn(n_states, n_actions, n_states)
        
        self.gamma = 0.99
        self.state = 0
    
    def reset(self):
        self.state = np.random.randint(self.n_states)
        return self.state
    
    def step(self, action):
        # Sample next state from P(s'|s,a)
        next_state = np.random.choice(
            self.n_states, 
            p=self.P[self.state, action]
        )
        
        # Get reward
        reward = self.R[self.state, action, next_state]
        
        self.state = next_state
        done = False  # Infinite horizon
        
        return next_state, reward, done, {}


def value_iteration(mdp, theta=1e-6):
    """Solve MDP using Value Iteration"""
    V = np.zeros(mdp.n_states)
    
    while True:
        delta = 0
        for s in range(mdp.n_states):
            v = V[s]
            # Bellman optimality update
            V[s] = max(
                sum(mdp.P[s,a,s_next] * (mdp.R[s,a,s_next] + mdp.gamma * V[s_next])
                    for s_next in range(mdp.n_states))
                for a in range(mdp.n_actions)
            )
            delta = max(delta, abs(v - V[s]))
        
        if delta < theta:
            break
    
    # Extract optimal policy
    policy = np.zeros(mdp.n_states, dtype=int)
    for s in range(mdp.n_states):
        policy[s] = np.argmax([
            sum(mdp.P[s,a,s_next] * (mdp.R[s,a,s_next] + mdp.gamma * V[s_next])
                for s_next in range(mdp.n_states))
            for a in range(mdp.n_actions)
        ])
    
    return V, policy
```

---

## 🔗 Where This Topic Is Used

| Topic | How MDP Is Used |
|-------|-----------------|
| **Q-Learning** | Learns Q(s,a) values for MDP |
| **Policy Gradient** | Optimizes π(a\|s) for MDP |
| **PPO / TRPO** | Policy optimization with MDP structure |
| **RLHF** | LLM as policy, human feedback as reward |
| **DPO** | Implicit MDP formulation |
| **AlphaGo / AlphaZero** | Game as MDP, MCTS planning |
| **Robotics Control** | Physical system as MDP |
| **Recommender Systems** | User session as MDP |
| **Autonomous Driving** | Driving as continuous MDP |
| **World Models** | Learn MDP dynamics P(s'\|s,a) |

### Prerequisite For

```
MDP --> Value Methods (Q-learning, DQN)
    --> Policy Methods (PPO, TRPO)
    --> Model-Based RL (Dreamer)
    --> RLHF / DPO
    --> Multi-Agent RL
```

---

## 📊 MDP Variants

| Variant | Modification | Use Case |
|---------|--------------|----------|
| **POMDP** | Partial observability | Real-world perception |
| **Continuous MDP** | Continuous S, A | Robotics |
| **Semi-MDP** | Variable duration actions | Options framework |
| **Multi-Agent** | Multiple agents | Games, economics |
| **Constrained MDP** | Safety constraints | Autonomous systems |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | Sutton & Barto Ch. 3 | [RL Book](http://incompleteideas.net/book/) |
| 📄 | Bellman 1957 | [Dynamic Programming](https://press.princeton.edu/books/paperback/9780691146683/dynamic-programming) |
| 🎥 | David Silver Lecture 2 | [YouTube](https://www.youtube.com/watch?v=lfHX2hHRMVQ) |
| 🇨🇳 | 马尔可夫决策过程详解 | [知乎](https://zhuanlan.zhihu.com/p/35261164) |
| 🇨🇳 | MDP与Bellman方程 | [CSDN](https://blog.csdn.net/qq_30615903/article/details/80686611) |
| 🇨🇳 | 强化学习-MDP | [B站](https://www.bilibili.com/video/BV1sd4y167NS) |
| 🇨🇳 | 强化学习入门MDP | [机器之心](https://www.jiqizhixin.com/articles/2018-02-13-4)

---

⬅️ [Back: Reinforcement Learning](../) | ➡️ [Next: 02-Value Methods](../02-value-methods/)
