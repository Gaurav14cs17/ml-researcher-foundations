<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=F39C12&height=120&section=header&text=Markov%20Decision%20Process&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-07-F39C12?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## 🔗 Navigation

⬅️ [Back: Foundations](../) | ➡️ [Next: MDP](../../01_mdp/)

---

## 📐 MDP Formal Definition

A Markov Decision Process is a 5-tuple:

```
MDP = (S, A, P, R, γ)

Where:
  S: State space (finite or continuous)
  A: Action space (finite or continuous)  
  P: S × A × S → [0,1], transition probability P(s'|s,a)
  R: S × A × S → ℝ, reward function R(s,a,s')
  γ: [0,1], discount factor

```

---

## 📐 The Markov Property (Proof)

### Statement

The future is conditionally independent of the past given the present:

```
P(s_{t+1} | s_t, a_t, s_{t-1}, a_{t-1}, ..., s_0, a_0) = P(s_{t+1} | s_t, a_t)

```

### Intuition

The current state contains all relevant information for predicting the future. The history provides no additional predictive power.

### Mathematical Formulation

```
For any sequence of states and actions:

P(s_{t+1}, r_{t+1} | s_0, a_0, r_1, s_1, a_1, ..., s_t, a_t) 
    = P(s_{t+1}, r_{t+1} | s_t, a_t)

This is the foundation of all RL algorithms!

```

---

## 📐 Value Functions

### State-Value Function V^π(s)

```
Definition:
V^π(s) = E_π[G_t | S_t = s]
       = E_π[Σ_{k=0}^∞ γ^k R_{t+k+1} | S_t = s]

Where G_t = R_{t+1} + γR_{t+2} + γ²R_{t+3} + ... is the return.

```

### Action-Value Function Q^π(s,a)

```
Definition:
Q^π(s,a) = E_π[G_t | S_t = s, A_t = a]
         = E_π[Σ_{k=0}^∞ γ^k R_{t+k+1} | S_t = s, A_t = a]

```

### Relationship Between V and Q

```
V^π(s) = Σ_a π(a|s) Q^π(s,a)

Q^π(s,a) = R(s,a) + γ Σ_{s'} P(s'|s,a) V^π(s')

Combining:
V^π(s) = Σ_a π(a|s) [R(s,a) + γ Σ_{s'} P(s'|s,a) V^π(s')]

```

---

## 📐 Bellman Equations (Complete Derivation)

### Bellman Expectation Equation for V

```
V^π(s) = E_π[G_t | S_t = s]
       = E_π[R_{t+1} + γG_{t+1} | S_t = s]           (definition of G_t)
       = E_π[R_{t+1} | S_t = s] + γ E_π[G_{t+1} | S_t = s]
       = Σ_a π(a|s) E[R_{t+1} | S_t = s, A_t = a] 
         + γ Σ_a π(a|s) Σ_{s'} P(s'|s,a) E_π[G_{t+1} | S_{t+1} = s']
       = Σ_a π(a|s) [R(s,a) + γ Σ_{s'} P(s'|s,a) V^π(s')]

This is the Bellman Expectation Equation for V!

```

### Bellman Expectation Equation for Q

```
Q^π(s,a) = E_π[G_t | S_t = s, A_t = a]
         = E[R_{t+1} + γ E_π[G_{t+1}] | S_t = s, A_t = a]
         = R(s,a) + γ Σ_{s'} P(s'|s,a) Σ_{a'} π(a'|s') Q^π(s',a')
         = R(s,a) + γ Σ_{s'} P(s'|s,a) V^π(s')

```

### Bellman Optimality Equations

```
The optimal value functions satisfy:

V*(s) = max_a Q*(s,a)
      = max_a [R(s,a) + γ Σ_{s'} P(s'|s,a) V*(s')]

Q*(s,a) = R(s,a) + γ Σ_{s'} P(s'|s,a) max_{a'} Q*(s',a')
        = R(s,a) + γ Σ_{s'} P(s'|s,a) V*(s')

```

---

## 📐 Optimal Policy Theorem

### Statement

For any MDP, there exists an optimal policy π* that is:
1. **Deterministic**: π*(s) = argmax_a Q*(s,a)
2. **Stationary**: Does not depend on time
3. **Unique value**: V^{π*}(s) = V*(s) for all s

### Proof Sketch

```
1. Any optimal policy must satisfy:
   π*(a|s) > 0 only if a ∈ argmax_a Q*(s,a)

2. Among all policies achieving V*, at least one is deterministic:
   π*(s) = argmax_a Q*(s,a)

3. If multiple actions are optimal, any deterministic selection works.

```

---

## 📐 Contraction Mapping Theorem

### The Bellman Operator

```
Define the Bellman optimality operator T:

(TQ)(s,a) = R(s,a) + γ Σ_{s'} P(s'|s,a) max_{a'} Q(s',a')

```

### Contraction Property

```
Theorem: T is a γ-contraction under the infinity norm:

||TQ₁ - TQ₂||_∞ ≤ γ ||Q₁ - Q₂||_∞

Proof:
|(TQ₁)(s,a) - (TQ₂)(s,a)| 
    = |γ Σ_{s'} P(s'|s,a) [max_{a'} Q₁(s',a') - max_{a'} Q₂(s',a')]|
    ≤ γ Σ_{s'} P(s'|s,a) |max_{a'} Q₁(s',a') - max_{a'} Q₂(s',a')|
    ≤ γ Σ_{s'} P(s'|s,a) max_{s',a'} |Q₁(s',a') - Q₂(s',a')|
    = γ ||Q₁ - Q₂||_∞

Since γ < 1, T is a contraction.

```

### Consequences

```
1. Q* is the unique fixed point: TQ* = Q*
2. Value iteration converges: Q_{n+1} = TQ_n → Q*
3. Convergence rate: ||Q_n - Q*||_∞ ≤ γⁿ ||Q₀ - Q*||_∞

```

---

## 💻 Code: Complete MDP Implementation

```python
import numpy as np
from typing import Tuple, Optional

class MDP:
    """
    Complete Markov Decision Process implementation
    """
    
    def __init__(self, n_states: int, n_actions: int, 
                 gamma: float = 0.99):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        
        # Transition probabilities P[s, a, s'] = P(s'|s,a)
        self.P = np.zeros((n_states, n_actions, n_states))
        
        # Rewards R[s, a, s']
        self.R = np.zeros((n_states, n_actions, n_states))
        
        # Current state
        self.state = 0
    
    def set_transition(self, s: int, a: int, s_next: int, 
                       prob: float, reward: float):
        """Set transition probability and reward"""
        self.P[s, a, s_next] = prob
        self.R[s, a, s_next] = reward
    
    def reset(self) -> int:
        """Reset to initial state"""
        self.state = np.random.randint(self.n_states)
        return self.state
    
    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        """Execute action, return (next_state, reward, done, info)"""
        # Sample next state
        probs = self.P[self.state, action]
        next_state = np.random.choice(self.n_states, p=probs)
        
        # Get reward
        reward = self.R[self.state, action, next_state]
        
        self.state = next_state
        done = False  # Infinite horizon
        
        return next_state, reward, done, {}

def policy_evaluation(mdp: MDP, policy: np.ndarray, 
                      theta: float = 1e-8) -> np.ndarray:
    """
    Iterative Policy Evaluation
    
    Computes V^π for a given policy π
    
    V(s) ← Σ_a π(a|s) Σ_{s'} P(s'|s,a)[R(s,a,s') + γV(s')]
    """
    V = np.zeros(mdp.n_states)
    
    while True:
        delta = 0
        for s in range(mdp.n_states):
            v = V[s]
            
            # Bellman expectation update
            new_v = 0
            for a in range(mdp.n_actions):
                for s_next in range(mdp.n_states):
                    new_v += policy[s, a] * mdp.P[s, a, s_next] * (
                        mdp.R[s, a, s_next] + mdp.gamma * V[s_next]
                    )
            
            V[s] = new_v
            delta = max(delta, abs(v - V[s]))
        
        if delta < theta:
            break
    
    return V

def policy_improvement(mdp: MDP, V: np.ndarray) -> np.ndarray:
    """
    Compute greedy policy with respect to V
    
    π(s) = argmax_a Σ_{s'} P(s'|s,a)[R(s,a,s') + γV(s')]
    """
    policy = np.zeros((mdp.n_states, mdp.n_actions))
    
    for s in range(mdp.n_states):
        q_values = np.zeros(mdp.n_actions)
        
        for a in range(mdp.n_actions):
            for s_next in range(mdp.n_states):
                q_values[a] += mdp.P[s, a, s_next] * (
                    mdp.R[s, a, s_next] + mdp.gamma * V[s_next]
                )
        
        best_action = np.argmax(q_values)
        policy[s, best_action] = 1.0
    
    return policy

def policy_iteration(mdp: MDP) -> Tuple[np.ndarray, np.ndarray]:
    """
    Policy Iteration Algorithm
    
    1. Initialize arbitrary policy
    2. Policy Evaluation: Compute V^π
    3. Policy Improvement: Compute greedy policy
    4. If policy unchanged, return; else go to 2
    """
    # Initialize random policy
    policy = np.ones((mdp.n_states, mdp.n_actions)) / mdp.n_actions
    
    while True:
        # Policy Evaluation
        V = policy_evaluation(mdp, policy)
        
        # Policy Improvement
        new_policy = policy_improvement(mdp, V)
        
        # Check for convergence
        if np.allclose(policy, new_policy):
            break
        
        policy = new_policy
    
    return V, policy

def value_iteration(mdp: MDP, theta: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """
    Value Iteration Algorithm
    
    V(s) ← max_a Σ_{s'} P(s'|s,a)[R(s,a,s') + γV(s')]
    """
    V = np.zeros(mdp.n_states)
    
    while True:
        delta = 0
        for s in range(mdp.n_states):
            v = V[s]
            
            # Bellman optimality update
            q_values = np.zeros(mdp.n_actions)
            for a in range(mdp.n_actions):
                for s_next in range(mdp.n_states):
                    q_values[a] += mdp.P[s, a, s_next] * (
                        mdp.R[s, a, s_next] + mdp.gamma * V[s_next]
                    )
            
            V[s] = np.max(q_values)
            delta = max(delta, abs(v - V[s]))
        
        if delta < theta:
            break
    
    # Extract optimal policy
    policy = policy_improvement(mdp, V)
    
    return V, policy

```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | Sutton & Barto Ch. 3-4 | [RL Book](http://incompleteideas.net/book/) |
| 📄 | Bellman 1957 | [Dynamic Programming](https://press.princeton.edu/books/paperback/9780691146683/dynamic-programming) |
| 🎥 | David Silver Lecture 2 | [YouTube](https://www.youtube.com/watch?v=lfHX2hHRMVQ) |
| 🇨🇳 | 马尔可夫决策过程详解 | [知乎](https://zhuanlan.zhihu.com/p/35261164) |
| 🇨🇳 | MDP与Bellman方程 | [CSDN](https://blog.csdn.net/qq_30615903/article/details/80686611) |

---

⬅️ [Back: Foundations](../) | ➡️ [Next: MDP Details](../../01_mdp/)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=F39C12&height=80&section=footer" width="100%"/>
</p>
