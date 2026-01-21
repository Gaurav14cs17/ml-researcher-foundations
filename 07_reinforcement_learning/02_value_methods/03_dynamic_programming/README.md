<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=F39C12&height=120&section=header&text=Dynamic%20Programming%20for%20RL&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-07-F39C12?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🔗 Navigation

⬅️ [Back: DQN](../02_dqn/) | ➡️ [Next: Q-Learning](../04_q_learning/)

---

## 🎯 Visual Overview

<img src="./images/dynamic-programming.svg" width="100%">

*Caption: DP methods require known MDP dynamics. Policy iteration alternates between evaluation and improvement. Value iteration combines both in one sweep.*

---

## 📂 Overview

Dynamic Programming (DP) computes optimal policies when the MDP model (transition probabilities and rewards) is fully known. While impractical for large state spaces, DP algorithms form the theoretical foundation for all RL methods.

---

## 📐 Mathematical Foundation

### MDP Definition

An MDP is defined by \((\mathcal{S}, \mathcal{A}, P, R, \gamma)\):

- \(\mathcal{S}\): State space

- \(\mathcal{A}\): Action space

- \(P(s'|s,a)\): Transition probability

- \(R(s,a,s')\): Reward function

- \(\gamma \in [0,1)\): Discount factor

### Bellman Equations

**State Value Function:**

$$V^\pi(s) = \mathbb{E}_\pi\left[\sum_{t=0}^\infty \gamma^t R_t | S_0 = s\right]$$

**Bellman Expectation Equation:**

$$V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) \left[R(s,a,s') + \gamma V^\pi(s')\right]$$

**Action Value Function:**

$$Q^\pi(s,a) = \sum_{s'} P(s'|s,a) \left[R(s,a,s') + \gamma \sum_{a'} \pi(a'|s') Q^\pi(s',a')\right]$$

### Bellman Optimality Equations

**Optimal State Value:**

$$V^*(s) = \max_a \sum_{s'} P(s'|s,a) \left[R(s,a,s') + \gamma V^*(s')\right]$$

**Optimal Action Value:**

$$Q^*(s,a) = \sum_{s'} P(s'|s,a) \left[R(s,a,s') + \gamma \max_{a'} Q^*(s',a')\right]$$

---

## 🔬 Policy Evaluation

### Iterative Policy Evaluation

**Goal:** Compute \(V^\pi\) for a given policy \(\pi\).

**Algorithm:**

```
Initialize V(s) arbitrarily for all s ∈ S
Repeat until Δ < θ:
    Δ ← 0
    For each s ∈ S:
        v ← V(s)
        V(s) ← Σ_a π(a|s) Σ_s' P(s'|s,a) [R(s,a,s') + γV(s')]
        Δ ← max(Δ, |v - V(s)|)

```

### Convergence Proof

**Theorem:** Iterative policy evaluation converges to \(V^\pi\).

**Proof:**

```
Define Bellman operator T^π:
(T^π V)(s) = Σ_a π(a|s) Σ_s' P(s'|s,a) [R + γV(s')]

T^π is a contraction with factor γ:
||T^π V₁ - T^π V₂||_∞ ≤ γ ||V₁ - V₂||_∞

Proof of contraction:

|(T^π V₁)(s) - (T^π V₂)(s)| 
= |Σ_a π(a|s) Σ_s' P(s'|s,a) γ[V₁(s') - V₂(s')]|
≤ Σ_a π(a|s) Σ_s' P(s'|s,a) γ|V₁(s') - V₂(s')|
≤ γ ||V₁ - V₂||_∞ · Σ_a π(a|s) Σ_s' P(s'|s,a)
= γ ||V₁ - V₂||_∞

By Banach fixed-point theorem:
V_k → V^π as k → ∞  ✓

```

**Convergence rate:**

$$\|V_k - V^\pi\|_\infty \leq \gamma^k \|V_0 - V^\pi\|_\infty$$

---

## 📊 Policy Iteration

### Algorithm

```
1. Initialization:
   V(s) ∈ ℝ arbitrarily for all s ∈ S
   π(s) ∈ A arbitrarily for all s ∈ S

2. Policy Evaluation:
   Repeat until Δ < θ:
       Δ ← 0
       For each s ∈ S:
           v ← V(s)
           V(s) ← Σ_s' P(s'|s,π(s)) [R(s,π(s),s') + γV(s')]
           Δ ← max(Δ, |v - V(s)|)

3. Policy Improvement:
   policy_stable ← true
   For each s ∈ S:
       old_action ← π(s)
       π(s) ← argmax_a Σ_s' P(s'|s,a) [R(s,a,s') + γV(s')]
       If old_action ≠ π(s): policy_stable ← false
   
   If policy_stable: return V, π
   Else: go to step 2

```

### Policy Improvement Theorem

**Theorem:** If \(\pi'\) is greedy with respect to \(V^\pi\), then \(\pi' \geq \pi\) (i.e., \(V^{\pi'}(s) \geq V^\pi(s)\) for all \(s\)).

**Proof:**

```
Let π'(s) = argmax_a Q^π(s,a)

Q^π(s, π'(s)) = max_a Q^π(s,a) ≥ Q^π(s, π(s)) = V^π(s)

Now show V^{π'}(s) ≥ V^π(s):

V^π(s) ≤ Q^π(s, π'(s))
       = E[R + γV^π(S') | S=s, A=π'(s)]
       ≤ E[R + γQ^π(S', π'(S')) | S=s, A=π'(s)]
       = E[R + γE[R' + γV^π(S'') | S', π'(S')] | S=s, π'(s)]
       = E[R + γR' + γ²V^π(S'') | s, π', π']
       ...
       ≤ E[Σ_{k=0}^∞ γ^k R_k | s, π']
       = V^{π'}(s)  ✓

```

### Convergence

**Theorem:** Policy iteration converges to optimal policy in finite number of iterations.

**Proof sketch:**

```
1. Policy space is finite (|A|^|S| policies)

2. Each improvement step strictly improves V or policy is optimal

3. No cycles possible (values strictly increase)

4. Must terminate in ≤ |A|^|S| iterations

In practice: Often converges in O(|S|²|A|) time

```

---

## 📊 Value Iteration

### Algorithm

```
1. Initialize V(s) arbitrarily for all s ∈ S

2. Repeat until Δ < θ:
       Δ ← 0
       For each s ∈ S:
           v ← V(s)
           V(s) ← max_a Σ_s' P(s'|s,a) [R(s,a,s') + γV(s')]
           Δ ← max(Δ, |v - V(s)|)

3. Output deterministic policy:
       π(s) = argmax_a Σ_s' P(s'|s,a) [R(s,a,s') + γV(s')]

```

### Convergence Proof

**Theorem:** Value iteration converges to \(V^*\).

**Proof:**

```
Define Bellman optimality operator T*:
(T* V)(s) = max_a Σ_s' P(s'|s,a) [R + γV(s')]

T* is a contraction:
||T* V₁ - T* V₂||_∞ ≤ γ ||V₁ - V₂||_∞

Proof:

|(T* V₁)(s) - (T* V₂)(s)|
= |max_a Q₁(s,a) - max_a Q₂(s,a)|
≤ max_a |Q₁(s,a) - Q₂(s,a)|
= max_a |Σ_s' P(s'|s,a) γ[V₁(s') - V₂(s')]|
≤ γ ||V₁ - V₂||_∞

By Banach theorem: V_k → V* ✓

```

### Value Iteration as Truncated Policy Iteration

```
Value Iteration = Policy Iteration with 1 evaluation sweep per improvement

Comparison:
Policy Iteration: Many evaluation sweeps, then improve
Value Iteration: One "evaluation" sweep that also improves

Trade-off:

- PI: Fewer iterations, expensive per iteration

- VI: More iterations, cheap per iteration

- Often VI wins for large state spaces

```

---

## 📐 Asynchronous Dynamic Programming

### Motivation

Standard DP does full sweeps over all states. Asynchronous methods update states in any order:

### Variants

**1. In-place DP:**

```
Use single value array, update in place
New values immediately used for next updates
Often faster convergence

```

**2. Prioritized Sweeping:**

```
Maintain priority queue of states
Priority = |Bellman error|
Update highest-priority state first
Efficient for sparse transitions

```

**3. Real-time DP:**

```
Only update states relevant to current trajectory
Focus computation where agent actually visits

```

---

## 💻 Complete Implementation

```python
import numpy as np
from typing import Tuple, Dict

class TabularMDP:
    """
    Tabular MDP with known dynamics
    """
    def __init__(self, n_states: int, n_actions: int, gamma: float = 0.99):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        
        # Transition probabilities: P[s][a][s'] = probability
        self.P = [[{} for _ in range(n_actions)] for _ in range(n_states)]
        
        # Rewards: R[s][a] = expected reward
        self.R = np.zeros((n_states, n_actions))
    
    def set_transition(self, s: int, a: int, s_next: int, prob: float, reward: float):
        """Add a transition"""
        self.P[s][a][s_next] = prob
        self.R[s, a] = reward

class DynamicProgramming:
    """
    Dynamic Programming algorithms for MDPs
    """
    def __init__(self, mdp: TabularMDP):
        self.mdp = mdp
        self.V = np.zeros(mdp.n_states)
        self.Q = np.zeros((mdp.n_states, mdp.n_actions))
        self.policy = np.zeros(mdp.n_states, dtype=int)
    
    def policy_evaluation(self, policy: np.ndarray, theta: float = 1e-8) -> np.ndarray:
        """
        Compute V^π using iterative policy evaluation
        
        Args:
            policy: deterministic policy, policy[s] = action
            theta: convergence threshold
        
        Returns:
            V: state value function
        """
        V = np.zeros(self.mdp.n_states)
        
        iteration = 0
        while True:
            delta = 0
            for s in range(self.mdp.n_states):
                v = V[s]
                a = policy[s]
                
                # Bellman expectation update
                new_v = 0
                for s_next, prob in self.mdp.P[s][a].items():
                    new_v += prob * (self.mdp.R[s, a] + self.mdp.gamma * V[s_next])
                
                V[s] = new_v
                delta = max(delta, abs(v - V[s]))
            
            iteration += 1
            if delta < theta:
                break
        
        print(f"Policy evaluation converged in {iteration} iterations")
        return V
    
    def policy_improvement(self, V: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Compute greedy policy with respect to V
        
        Returns:
            policy: improved policy
            stable: True if policy unchanged
        """
        policy = np.zeros(self.mdp.n_states, dtype=int)
        
        for s in range(self.mdp.n_states):
            q_values = np.zeros(self.mdp.n_actions)
            
            for a in range(self.mdp.n_actions):
                for s_next, prob in self.mdp.P[s][a].items():
                    q_values[a] += prob * (self.mdp.R[s, a] + self.mdp.gamma * V[s_next])
            
            policy[s] = np.argmax(q_values)
        
        stable = np.array_equal(policy, self.policy)
        return policy, stable
    
    def policy_iteration(self, theta: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
        """
        Policy Iteration algorithm
        
        Returns:
            V: optimal value function
            policy: optimal policy
        """
        # Initialize random policy
        self.policy = np.random.randint(0, self.mdp.n_actions, self.mdp.n_states)
        
        iteration = 0
        while True:
            # Policy Evaluation
            V = self.policy_evaluation(self.policy, theta)
            
            # Policy Improvement
            new_policy, stable = self.policy_improvement(V)
            
            iteration += 1
            print(f"Policy iteration {iteration}: stable={stable}")
            
            if stable:
                break
            
            self.policy = new_policy
        
        self.V = V
        return V, self.policy
    
    def value_iteration(self, theta: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
        """
        Value Iteration algorithm
        
        Returns:
            V: optimal value function
            policy: optimal policy
        """
        V = np.zeros(self.mdp.n_states)
        
        iteration = 0
        while True:
            delta = 0
            for s in range(self.mdp.n_states):
                v = V[s]
                
                # Bellman optimality update
                q_values = np.zeros(self.mdp.n_actions)
                for a in range(self.mdp.n_actions):
                    for s_next, prob in self.mdp.P[s][a].items():
                        q_values[a] += prob * (self.mdp.R[s, a] + self.mdp.gamma * V[s_next])
                
                V[s] = np.max(q_values)
                delta = max(delta, abs(v - V[s]))
            
            iteration += 1
            if delta < theta:
                break
        
        print(f"Value iteration converged in {iteration} iterations")
        
        # Extract policy
        policy = np.zeros(self.mdp.n_states, dtype=int)
        for s in range(self.mdp.n_states):
            q_values = np.zeros(self.mdp.n_actions)
            for a in range(self.mdp.n_actions):
                for s_next, prob in self.mdp.P[s][a].items():
                    q_values[a] += prob * (self.mdp.R[s, a] + self.mdp.gamma * V[s_next])
            policy[s] = np.argmax(q_values)
        
        self.V = V
        self.policy = policy
        return V, policy
    
    def compute_q_from_v(self, V: np.ndarray) -> np.ndarray:
        """Compute Q(s,a) from V(s)"""
        Q = np.zeros((self.mdp.n_states, self.mdp.n_actions))
        
        for s in range(self.mdp.n_states):
            for a in range(self.mdp.n_actions):
                for s_next, prob in self.mdp.P[s][a].items():
                    Q[s, a] += prob * (self.mdp.R[s, a] + self.mdp.gamma * V[s_next])
        
        return Q

class PrioritizedSweeping:
    """
    Asynchronous DP with prioritized updates
    """
    def __init__(self, mdp: TabularMDP):
        self.mdp = mdp
        self.V = np.zeros(mdp.n_states)
        
    def solve(self, theta: float = 1e-8, max_updates: int = 100000) -> np.ndarray:
        import heapq
        
        # Initialize with Bellman errors
        priorities = []
        for s in range(self.mdp.n_states):
            error = self._bellman_error(s)
            heapq.heappush(priorities, (-error, s))  # Max-heap via negation
        
        updates = 0
        while priorities and updates < max_updates:
            neg_priority, s = heapq.heappop(priorities)
            
            if -neg_priority < theta:
                continue
            
            # Update state
            old_v = self.V[s]
            self._update_state(s)
            updates += 1
            
            # Add predecessors to queue
            for s_pred in self._get_predecessors(s):
                error = self._bellman_error(s_pred)
                if error > theta:
                    heapq.heappush(priorities, (-error, s_pred))
        
        print(f"Prioritized sweeping: {updates} updates")
        return self.V
    
    def _bellman_error(self, s: int) -> float:
        """Compute |V(s) - max_a Q(s,a)|"""
        max_q = float('-inf')
        for a in range(self.mdp.n_actions):
            q = 0
            for s_next, prob in self.mdp.P[s][a].items():
                q += prob * (self.mdp.R[s, a] + self.mdp.gamma * self.V[s_next])
            max_q = max(max_q, q)
        return abs(self.V[s] - max_q)
    
    def _update_state(self, s: int):
        """Bellman optimality backup"""
        max_q = float('-inf')
        for a in range(self.mdp.n_actions):
            q = 0
            for s_next, prob in self.mdp.P[s][a].items():
                q += prob * (self.mdp.R[s, a] + self.mdp.gamma * self.V[s_next])
            max_q = max(max_q, q)
        self.V[s] = max_q
    
    def _get_predecessors(self, s: int):
        """Find states that can transition to s"""
        predecessors = set()
        for s_pred in range(self.mdp.n_states):
            for a in range(self.mdp.n_actions):
                if s in self.mdp.P[s_pred][a]:
                    predecessors.add(s_pred)
        return predecessors

# Example: Gridworld
def create_gridworld(size: int = 4) -> TabularMDP:
    """Create simple gridworld MDP"""
    n_states = size * size
    n_actions = 4  # up, down, left, right
    
    mdp = TabularMDP(n_states, n_actions, gamma=0.99)
    
    # Set transitions
    for s in range(n_states):
        row, col = s // size, s % size
        
        for a, (dr, dc) in enumerate([(-1, 0), (1, 0), (0, -1), (0, 1)]):
            new_row = max(0, min(size-1, row + dr))
            new_col = max(0, min(size-1, col + dc))
            s_next = new_row * size + new_col
            
            # Reward: -1 per step, 0 at goal (bottom-right)
            reward = 0 if s_next == n_states - 1 else -1
            
            mdp.set_transition(s, a, s_next, 1.0, reward)
    
    return mdp

# Run example
mdp = create_gridworld(4)
dp = DynamicProgramming(mdp)

print("=== Value Iteration ===")
V_vi, policy_vi = dp.value_iteration()
print(f"Value function:\n{V_vi.reshape(4, 4)}")

print("\n=== Policy Iteration ===")
dp2 = DynamicProgramming(mdp)
V_pi, policy_pi = dp2.policy_iteration()
print(f"Value function:\n{V_pi.reshape(4, 4)}")

print("\n=== Prioritized Sweeping ===")
ps = PrioritizedSweeping(mdp)
V_ps = ps.solve()
print(f"Value function:\n{V_ps.reshape(4, 4)}")

```

---

## 📊 Complexity Analysis

| Algorithm | Time per Iteration | Convergence | Total |
|-----------|-------------------|-------------|-------|
| **Policy Evaluation** | \(O(S^2A)\) | \(O(\log(1/\epsilon)/\log(1/\gamma))\) | \(O(S^2A/\epsilon)\) |
| **Policy Iteration** | \(O(S^2A + S^3)\) | \(O(A^S)\) worst, \(O(S)\) typical | Varies |
| **Value Iteration** | \(O(S^2A)\) | \(O(\log(1/\epsilon)/\log(1/\gamma))\) | \(O(S^2A/\epsilon)\) |
| **Prioritized Sweeping** | \(O(\log S)\) per update | Problem-dependent | Often \(O(S \log S)\) |

---

## 📚 References

| Type | Resource | Link |
|------|----------|------|
| 📖 | Sutton & Barto Ch. 4 | [Free Online](http://incompleteideas.net/book/the-book.html) |
| 📖 | Bertsekas & Tsitsiklis | Dynamic Programming and Optimal Control |
| 📄 | Prioritized Sweeping | Moore & Atkeson, 1993 |

---

## 🗺️ Navigation

| ⬅️ Previous | 🏠 Home | ➡️ Next |
|:-----------:|:-------:|:-------:|
| [DQN](../02_dqn/README.md) | [Value Methods](../README.md) | [Q-Learning](../04_q_learning/README.md) |

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=F39C12&height=80&section=footer" width="100%"/>
</p>
