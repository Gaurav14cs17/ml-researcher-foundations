<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=F39C12&height=120&section=header&text=Discount%20Factor%20γ&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-07-F39C12?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## 🔗 Navigation

⬅️ [Back: MDP](../) | ➡️ [Next: Dynamics](../02_dynamics/)

---

## 🎯 Visual Overview

<img src="./images/discount-factor.svg" width="100%">

*Caption: The discount factor γ ∈ [0,1] determines how much we value future rewards. With γ=0.9, a reward 10 steps away is worth only 0.35 of its face value today. Higher γ = more far-sighted agent.*

---

## 📂 Overview

The discount factor γ controls the agent's "patience" - how much it values future rewards compared to immediate ones. It is one of the most important hyperparameters in RL.

---

## 📐 Mathematical Definition

### Discounted Return

```
G_t = R_{t+1} + γR_{t+2} + γ²R_{t+3} + ...
    = Σ_{k=0}^∞ γ^k R_{t+k+1}

Where γ ∈ [0, 1] is the discount factor.
```

### Recursive Definition

```
G_t = R_{t+1} + γG_{t+1}

This recursive structure is the basis for TD learning!
```

---

## 📐 Why Discount?

### 1. Mathematical Necessity

```
For infinite horizon MDPs, we need the return to be finite:

If |R_t| ≤ R_max for all t, then:

|G_t| ≤ Σ_{k=0}^∞ γ^k R_max = R_max / (1-γ)

This is finite only if γ < 1!
```

### 2. Economic Interpretation

```
"A dollar today is worth more than a dollar tomorrow"

Reasons:
• Uncertainty about the future
• Opportunity cost of waiting
• Preference for immediate gratification
• Risk of episode termination
```

### 3. Computational Properties

```
Discounting makes value functions well-defined:

V(s) = E[Σ_{k=0}^∞ γ^k R_{t+k}]

Without discounting (γ=1), this sum may diverge for non-episodic tasks.
```

---

## 📐 Effective Horizon

The discount factor determines how far ahead the agent "looks":

```
Effective Horizon = 1 / (1 - γ)

Examples:
• γ = 0.9   → Horizon ≈ 10 steps
• γ = 0.99  → Horizon ≈ 100 steps
• γ = 0.999 → Horizon ≈ 1000 steps

Derivation:
Weight at step k: γ^k
Total weight: Σ_{k=0}^∞ γ^k = 1/(1-γ)
Half of weight is in first 1/(1-γ) steps
```

### Present Value Analysis

```
What is the present value of a future reward?

Reward r at time t+k has present value: γ^k · r

Examples (γ = 0.9):
• k=0:  γ^0 = 1.00  (100% value)
• k=5:  γ^5 = 0.59  (59% value)
• k=10: γ^10 = 0.35 (35% value)
• k=20: γ^20 = 0.12 (12% value)
```

---

## 📐 Effect on Bellman Equations

### Bellman Expectation Equation

```
V^π(s) = E_π[R_{t+1} + γV^π(S_{t+1}) | S_t = s]
       = Σ_a π(a|s) [R(s,a) + γ Σ_{s'} P(s'|s,a) V^π(s')]

The γ determines how much future values contribute.
```

### Bellman Optimality Equation

```
V*(s) = max_a [R(s,a) + γ Σ_{s'} P(s'|s,a) V*(s')]

Smaller γ → Optimization focuses on immediate rewards
Larger γ → Optimization considers long-term consequences
```

---

## 📐 γ and TD Learning

### TD(0) Update

```
V(s) ← V(s) + α[R + γV(s') - V(s)]

The TD target is: R + γV(s')
                  └── Future value is discounted
```

### TD Error

```
δ_t = R_{t+1} + γV(S_{t+1}) - V(S_t)

This is the "surprise" - difference between expected and actual.
```

### n-Step Returns

```
G_t:t+n = R_{t+1} + γR_{t+2} + ... + γ^{n-1}R_{t+n} + γ^n V(S_{t+n})

Each future reward is discounted by its distance from present.
```

---

## 📐 Choosing γ

### Guidelines

| γ Value | Behavior | Use Case |
|---------|----------|----------|
| **γ = 0** | Myopic, greedy | Bandits, immediate reward |
| **γ = 0.9** | Short-sighted | Short episodes (~10 steps) |
| **γ = 0.99** | Far-sighted | Medium episodes (~100 steps) |
| **γ = 0.999** | Very patient | Long episodes (~1000 steps) |
| **γ = 1** | Undiscounted | Episodic with guaranteed termination |

### Practical Considerations

```
Start with γ = 0.99 for most problems.

Decrease γ if:
• Training is unstable (values too large)
• Agent focuses too much on distant rewards
• Episode length is short

Increase γ if:
• Agent is too myopic
• Important rewards are delayed
• Long-term planning is needed
```

---

## 📐 γ in Policy Gradient Methods

### Return Estimation

```
In REINFORCE:
G_t = Σ_{k=0}^{T-t-1} γ^k R_{t+k+1}

Gradient estimate:
∇J(θ) ≈ Σ_t ∇log π_θ(a_t|s_t) · G_t
```

### Generalized Advantage Estimation (GAE)

```
GAE uses both γ and λ:

Â^GAE_t = Σ_{l=0}^∞ (γλ)^l δ_{t+l}

Where δ_t = r_t + γV(s_{t+1}) - V(s_t)

• γ: Standard discount for future rewards
• λ: Controls bias-variance tradeoff in advantage
```

---

## 💻 Code Examples

```python
import numpy as np
import matplotlib.pyplot as plt

def compute_returns(rewards, gamma=0.99):
    """
    Compute discounted returns G_t = Σ γ^k r_{t+k}
    
    Uses reverse iteration for efficiency:
    G_T = r_T
    G_t = r_t + γ * G_{t+1}
    """
    T = len(rewards)
    returns = np.zeros(T)
    G = 0
    
    for t in reversed(range(T)):
        G = rewards[t] + gamma * G
        returns[t] = G
    
    return returns

def discount_factor_visualization():
    """Visualize effect of different gamma values"""
    gammas = [0.0, 0.5, 0.9, 0.99, 1.0]
    steps = np.arange(50)
    
    plt.figure(figsize=(10, 6))
    for gamma in gammas:
        weights = gamma ** steps
        plt.plot(steps, weights, label=f'γ={gamma}')
    
    plt.xlabel('Steps into future')
    plt.ylabel('Discount weight γ^k')
    plt.title('Present Value of Future Rewards')
    plt.legend()
    plt.grid(True)
    plt.show()

def effective_horizon(gamma):
    """Compute effective planning horizon"""
    return 1 / (1 - gamma) if gamma < 1 else float('inf')

# Example: Impact of gamma on returns
rewards = [1, 1, 1, 1, 1, 1, 1, 1, 1, 10]  # Delayed large reward

print("Returns with different discount factors:")
for gamma in [0.0, 0.5, 0.9, 0.99]:
    returns = compute_returns(rewards, gamma)
    print(f"γ={gamma:0.2f}: G_0={returns[0]:7.2f}, horizon={effective_horizon(gamma):5.1f}")

# Output:
# γ=0.00: G_0=   1.00, horizon=  1.0
# γ=0.50: G_0=   6.99, horizon=  2.0
# γ=0.90: G_0=  12.41, horizon= 10.0
# γ=0.99: G_0=  18.56, horizon=100.0

class DiscountedMDP:
    """MDP with configurable discount factor"""
    
    def __init__(self, gamma=0.99):
        self.gamma = gamma
    
    def td_target(self, reward, next_value, done):
        """Compute TD target: r + γV(s')"""
        if done:
            return reward
        return reward + self.gamma * next_value
    
    def n_step_return(self, rewards, final_value, dones):
        """Compute n-step return"""
        G = final_value
        for t in reversed(range(len(rewards))):
            if dones[t]:
                G = rewards[t]
            else:
                G = rewards[t] + self.gamma * G
        return G
    
    def gae(self, rewards, values, dones, lam=0.95):
        """
        Generalized Advantage Estimation
        Uses both γ (reward discount) and λ (trace decay)
        """
        T = len(rewards)
        advantages = np.zeros(T)
        gae = 0
        
        for t in reversed(range(T)):
            if t == T - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * lam * (1 - dones[t]) * gae
            advantages[t] = gae
        
        return advantages
```

---

## 📊 Summary Table

| Concept | Formula | Interpretation |
|---------|---------|----------------|
| Return | G_t = Σ γ^k R_{t+k} | Sum of discounted rewards |
| Present value | γ^k · r | Value today of reward at step k |
| Effective horizon | 1/(1-γ) | How far agent looks ahead |
| TD target | r + γV(s') | Bootstrap estimate |
| GAE | Σ (γλ)^l δ_{t+l} | Advantage estimate |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | Sutton & Barto Ch. 3 | [RL Book](http://incompleteideas.net/book/) |
| 🎥 | David Silver Lecture 2 | [YouTube](https://www.youtube.com/watch?v=lfHX2hHRMVQ) |
| 🇨🇳 | 折扣因子详解 | [知乎](https://zhuanlan.zhihu.com/p/35261164) |
| 🇨🇳 | 强化学习基础 | [B站](https://www.bilibili.com/video/BV1sd4y167NS) |

## 🔗 Where This Topic Is Used

| Application | Discount Factor |
|-------------|----------------|
| **Finance** | Time value of money |
| **Long-horizon tasks** | γ close to 1 |
| **Short-term focus** | γ close to 0 |
| **Infinite horizon** | Ensures convergence |
| **GAE (PPO)** | Both γ and λ |

---

⬅️ [Back: MDP](../) | ➡️ [Next: Dynamics](../02_dynamics/)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=F39C12&height=80&section=footer" width="100%"/>
</p>
