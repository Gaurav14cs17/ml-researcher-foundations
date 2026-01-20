<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=F39C12&height=120&section=header&text=Planning%20in%20RL&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-07-F39C12?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## 🔗 Navigation

⬅️ [Back: MCTS](../02_mcts/) | ➡️ [Next: World Models](../04_world_models/)

---

## 🎯 Visual Overview

<img src="./images/planning.svg" width="100%">

*Caption: Planning uses a world model to simulate future trajectories and select the best action. Methods include random shooting, CEM, and MCTS.*

---

## 📂 Overview

Planning algorithms use learned or known dynamics models to simulate possible futures and select optimal actions without real environment interaction.

---

## 🔑 Planning Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| **Random Shooting** | Sample random action sequences | Simple, fast |
| **CEM** | Iteratively refine action distribution | Continuous control |
| **MCTS** | Tree search with UCB | Games, discrete |
| **MPC** | Re-plan at each step | Robotics |

---

## 📐 Mathematical Framework

### Planning Problem Formulation

```
Given:
  - Current state s_t
  - Dynamics model: p̂(s'|s, a) or ŝ' = f(s, a)
  - Reward model: r̂ = R(s, a)
  - Planning horizon: H

Find:
  a*_{t:t+H} = argmax_{a_{t:t+H}} E[Σ_{k=0}^H γ^k R(s_{t+k}, a_{t+k})]
  
  Subject to: s_{t+k+1} = f(s_{t+k}, a_{t+k})
```

### Model Predictive Control (MPC)

```
Algorithm:
  At each timestep t:
  1. Solve: a*_{t:t+H} = argmax Σ_{k=0}^H γ^k R̂(s_{t+k}, a_{t+k})
  2. Execute only first action: apply a*_t
  3. Observe new state s_{t+1}
  4. Repeat with new state

Receding Horizon Property:
  - Re-plan at every step
  - Corrects for model errors
  - Adapts to new observations
```

---

## 📐 Cross-Entropy Method (CEM)

### Algorithm

```
Input: state s, horizon H, iterations N, samples K, elite fraction ρ

Initialize:
  μ = 0 ∈ ℝ^{H×|A|}  (mean)
  σ = 1 ∈ ℝ^{H×|A|}  (std)

For n = 1 to N:
  1. Sample: a^(i) ~ N(μ, σ²), i = 1,...,K
  
  2. Evaluate: J^(i) = Σ_{t=0}^H γ^t R̂(ŝ_t, a^(i)_t)
     where ŝ_{t+1} = f(ŝ_t, a^(i)_t)
  
  3. Select elite: E = top ρK samples by J
  
  4. Refit: μ = mean(E), σ = std(E)

Return: a*_0 = μ[0]  (first action of final mean)
```

### CEM Convergence

```
Theorem: CEM converges to local optimum as N → ∞, ρ → 0

Proof sketch:
  - Each iteration reduces entropy of distribution
  - Elite selection biases toward high-return regions
  - In limit, distribution concentrates on local max
```

---

## 📐 Random Shooting

### Simple Baseline

```
Algorithm:
  1. Sample K random action sequences: a^(i)_{0:H} ~ Uniform(A)^H
  2. Evaluate each: J^(i) = Σ_t γ^t R̂(ŝ_t, a^(i)_t)
  3. Return: argmax_i J^(i)

Properties:
  - Simple to implement
  - No gradient needed
  - Scales poorly with horizon (exponential)
```

---

## 📐 Trajectory Optimization

### Shooting Methods

```
Single Shooting:
  - Optimize over action sequence: min_u L(x₀, u₀, u₁, ..., u_H)
  - States computed from dynamics: x_{t+1} = f(x_t, u_t)

Multiple Shooting:
  - Optimize over both states and actions
  - Add constraints: x_{t+1} = f(x_t, u_t)
  - More stable numerically
```

### iLQR (Iterative Linear Quadratic Regulator)

```
For nonlinear dynamics f(x, u):

1. Forward pass: Roll out with current policy
   x̄_{t+1} = f(x̄_t, ū_t)

2. Backward pass: Linearize and solve LQR
   δx_{t+1} = A_t δx_t + B_t δu_t
   where A_t = ∂f/∂x|_{x̄_t, ū_t}, B_t = ∂f/∂u|_{x̄_t, ū_t}

3. Update: u_t ← ū_t + α δu_t

Repeat until convergence.

LQR solution gives optimal linear feedback:
  δu_t = -K_t δx_t
```

---

## 📐 Model Uncertainty

### Ensemble Planning

```
Train ensemble of models: {f_1, ..., f_M}

For planning:
  J(a) = E_{m~Uniform(1..M)} [Σ_t R(s^m_t, a_t)]
  
  where s^m_{t+1} = f_m(s^m_t, a_t)

Accounts for epistemic uncertainty in dynamics.
```

### Probabilistic Planning

```
For stochastic dynamics p(s'|s, a):

  J(a_{0:H}) = E_{s_1, ..., s_H} [Σ_t γ^t R(s_t, a_t)]
  
  where s_{t+1} ~ p(·|s_t, a_t)

Requires sampling or moment matching.
```

---

## 💻 Code

```python
def cem_planning(model, state, horizon, n_samples=500, n_elite=50, n_iters=5):
    """Cross-Entropy Method for action planning"""
    action_dim = model.action_dim
    
    # Initialize action distribution
    mean = np.zeros((horizon, action_dim))
    std = np.ones((horizon, action_dim))
    
    for _ in range(n_iters):

        # Sample action sequences
        actions = np.random.normal(mean, std, (n_samples, horizon, action_dim))
        
        # Evaluate with model
        returns = []
        for action_seq in actions:
            states, rewards = model.rollout(state, action_seq)
            returns.append(sum(rewards))
        
        # Select elite samples
        elite_idx = np.argsort(returns)[-n_elite:]
        elite_actions = actions[elite_idx]
        
        # Update distribution
        mean = elite_actions.mean(axis=0)
        std = elite_actions.std(axis=0)
    
    return mean[0]  # Return first action
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | CEM Paper | [arXiv](https://arxiv.org/abs/1712.00885) |
| 📄 | PETS Paper | [arXiv](https://arxiv.org/abs/1805.12114) |
| 📖 | MPC Tutorial | [Docs](https://pytorch.org/tutorials/) |
| 🇨🇳 | MPC详解 | [知乎](https://zhuanlan.zhihu.com/p/563656219) |
| 🇨🇳 | 规划算法 | [CSDN](https://blog.csdn.net/qq_37006625/article/details/123629543) |
| 🇨🇳 | 机器人控制 | [B站](https://www.bilibili.com/video/BV1yp4y1s7Qw) |

## 🔗 Where This Topic Is Used

| Application | Planning |
|-------------|---------|
| **AlphaGo** | MCTS with neural nets |
| **MPC** | Receding horizon control |
| **Robotics** | Trajectory optimization |
| **Games** | Lookahead search |

---

⬅️ [Back: MCTS](../02_mcts/) | ➡️ [Next: World Models](../04_world_models/)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=F39C12&height=80&section=footer" width="100%"/>
</p>
