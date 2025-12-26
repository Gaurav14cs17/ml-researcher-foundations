<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=F39C12&height=120&section=header&text=Reinforcement%20Learning%20Foundations&fontSize=28&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-07-F39C12?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## 🔗 Navigation

⬅️ [Back: Reinforcement Learning](../) | ➡️ [Next: MDP](./01_mdp/)

---

## 📐 Core Concepts

### The RL Framework

Reinforcement Learning studies how agents learn to make sequential decisions through interaction with an environment to maximize cumulative reward.

```
Agent-Environment Interaction Loop:

    +--------+    action a_t    +-----------+
    |        | ---------------> |           |
    |  Agent |                  | Environment |
    |        | <--------------- |           |
    +--------+   state s_t      +-----------+
                 reward r_t
```

### Formal Definition

```
At each timestep t:
1. Agent observes state s_t ∈ S
2. Agent selects action a_t ∈ A according to policy π(a|s)
3. Environment transitions: s_{t+1} ~ P(·|s_t, a_t)
4. Agent receives reward: r_t = R(s_t, a_t, s_{t+1})
5. Repeat until termination

Goal: Find policy π* that maximizes expected cumulative reward
      π* = argmax_π E_π[Σ_{t=0}^∞ γ^t r_t]
```

---

## 📐 Mathematical Framework

### Key Elements

| Symbol | Name | Definition |
|--------|------|------------|
| S | State space | Set of all possible states |
| A | Action space | Set of all possible actions |
| P(s'\|s,a) | Transition dynamics | Probability of next state |
| R(s,a,s') | Reward function | Immediate feedback signal |
| γ ∈ [0,1] | Discount factor | Present value of future rewards |
| π(a\|s) | Policy | Probability of action given state |

### The Markov Property

```
The Markov property states that the future is independent of the past 
given the present:

P(s_{t+1} | s_t, a_t, s_{t-1}, a_{t-1}, ..., s_0, a_0) = P(s_{t+1} | s_t, a_t)

This is the key assumption that makes RL tractable!
```

### Value Functions

```
State-Value Function:
V^π(s) = E_π[Σ_{t=0}^∞ γ^t r_t | s_0 = s]
       = E_π[r_0 + γr_1 + γ²r_2 + ... | s_0 = s]

"Expected total discounted reward starting from state s, following policy π"

Action-Value Function:
Q^π(s,a) = E_π[Σ_{t=0}^∞ γ^t r_t | s_0 = s, a_0 = a]

"Expected total discounted reward starting from (s,a), then following π"

Relationship:
V^π(s) = Σ_a π(a|s) Q^π(s,a)
Q^π(s,a) = R(s,a) + γ Σ_{s'} P(s'|s,a) V^π(s')
```

### Bellman Equations

```
Bellman Expectation Equations:

V^π(s) = Σ_a π(a|s) [R(s,a) + γ Σ_{s'} P(s'|s,a) V^π(s')]

Q^π(s,a) = R(s,a) + γ Σ_{s'} P(s'|s,a) Σ_{a'} π(a'|s') Q^π(s',a')

Bellman Optimality Equations:

V*(s) = max_a [R(s,a) + γ Σ_{s'} P(s'|s,a) V*(s')]

Q*(s,a) = R(s,a) + γ Σ_{s'} P(s'|s,a) max_{a'} Q*(s',a')

Optimal Policy:
π*(s) = argmax_a Q*(s,a)
```

---

## 📐 Key Theorems

### Theorem 1: Policy Improvement

```
For any policy π, if we define π' as:

π'(s) = argmax_a Q^π(s,a)

Then: V^{π'}(s) ≥ V^π(s) for all s

Proof sketch:
V^π(s) ≤ max_a Q^π(s,a)           (by definition of π')
       = Q^π(s, π'(s))            (π' picks the argmax)
       = R(s,π'(s)) + γ Σ_{s'} P(s'|s,π'(s)) V^π(s')
       ≤ R(s,π'(s)) + γ Σ_{s'} P(s'|s,π'(s)) max_{a'} Q^π(s',a')
       = ... (repeat recursively)
       = V^{π'}(s)
```

### Theorem 2: Contraction Property

```
The Bellman operator T is a γ-contraction:

||T Q_1 - T Q_2||_∞ ≤ γ ||Q_1 - Q_2||_∞

This guarantees:
1. Value iteration converges
2. Q* is the unique fixed point
3. Convergence rate is O(γ^n)
```

### Theorem 3: Policy Gradient

```
∇_θ J(θ) = E_{τ~π_θ} [Σ_t ∇_θ log π_θ(a_t|s_t) · Q^{π_θ}(s_t, a_t)]

This allows gradient-based optimization of policies!
```

---

## 📊 RL Taxonomy

```
RL Methods
│
├── Value-Based
│   ├── Dynamic Programming (known model)
│   │   ├── Policy Iteration
│   │   └── Value Iteration
│   │
│   └── Model-Free
│       ├── Monte Carlo
│       ├── TD Learning (SARSA)
│       └── Q-Learning → DQN → Rainbow
│
├── Policy-Based
│   ├── REINFORCE
│   ├── Actor-Critic (A2C, A3C)
│   └── PPO, TRPO → RLHF for LLMs
│
└── Model-Based
    ├── Dyna
    ├── MCTS (AlphaGo)
    └── World Models (Dreamer)
```

---

## 💻 Code: Basic RL Framework

```python
import numpy as np
from abc import ABC, abstractmethod

class Environment(ABC):
    """Abstract RL Environment"""
    
    @abstractmethod
    def reset(self) -> np.ndarray:
        """Reset and return initial state"""
        pass
    
    @abstractmethod
    def step(self, action: int) -> tuple:
        """Execute action, return (next_state, reward, done, info)"""
        pass

class Agent(ABC):
    """Abstract RL Agent"""
    
    @abstractmethod
    def select_action(self, state: np.ndarray) -> int:
        """Select action given current state"""
        pass
    
    @abstractmethod
    def update(self, state, action, reward, next_state, done):
        """Update agent from experience"""
        pass

def train(env: Environment, agent: Agent, n_episodes: int) -> list:
    """Standard RL training loop"""
    rewards_history = []
    
    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # Agent selects action
            action = agent.select_action(state)
            
            # Environment responds
            next_state, reward, done, info = env.step(action)
            
            # Agent learns
            agent.update(state, action, reward, next_state, done)
            
            total_reward += reward
            state = next_state
        
        rewards_history.append(total_reward)
    
    return rewards_history

# Example: Value Function Computation
def compute_value_function(env, policy, gamma=0.99, theta=1e-6):
    """
    Policy Evaluation: Compute V^π using iterative Bellman updates
    
    V(s) ← Σ_a π(a|s) [R(s,a) + γ Σ_{s'} P(s'|s,a) V(s')]
    """
    V = np.zeros(env.n_states)
    
    while True:
        delta = 0
        for s in range(env.n_states):
            v = V[s]
            V[s] = sum(
                policy[s, a] * sum(
                    env.P[s, a, s_next] * (env.R[s, a, s_next] + gamma * V[s_next])
                    for s_next in range(env.n_states)
                )
                for a in range(env.n_actions)
            )
            delta = max(delta, abs(v - V[s]))
        
        if delta < theta:
            break
    
    return V
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | Sutton & Barto (2018) | [RL: An Introduction](http://incompleteideas.net/book/) |
| 🎥 | David Silver Lectures | [YouTube](https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ) |
| 🎓 | Berkeley CS285 | [Deep RL Course](http://rail.eecs.berkeley.edu/deeprlcourse/) |
| 🇨🇳 | 强化学习入门 | [知乎](https://zhuanlan.zhihu.com/p/25498081) |
| 🇨🇳 | RL基础教程 | [B站](https://www.bilibili.com/video/BV1sd4y167NS) |

---

## 🔗 Navigation

⬅️ [Back: Reinforcement Learning](../) | ➡️ [Next: MDP](../01_mdp/)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=F39C12&height=80&section=footer" width="100%"/>
</p>
