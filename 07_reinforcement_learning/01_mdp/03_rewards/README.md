<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=F39C12&height=120&section=header&text=Reward%20Functions&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-07-F39C12?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## 🔗 Navigation

⬅️ [Back: Dynamics](../02_dynamics/) | ➡️ [Next: States & Actions](../04_states_actions/)

---

## 🎯 Visual Overview

<img src="./images/reward-function.svg" width="100%">

*Caption: Rewards R(s,a,s') provide scalar feedback after each transition. The agent's goal is to maximize cumulative discounted rewards. Sparse rewards are hard to learn from; dense rewards are easier but may cause reward hacking.*

---

## 📂 Overview

The reward function defines what the agent should optimize. Good reward design is crucial - poorly designed rewards lead to unintended behaviors.

---

## 🔑 Key Concepts

| Concept | Description |
|---------|-------------|
| **Reward r** | Scalar feedback signal |
| **Return G** | Cumulative discounted reward |
| **Sparse Reward** | Signal only at goal |
| **Dense Reward** | Signal every step |
| **Reward Shaping** | Hand-designed hints |

---

## 📐 Mathematical Formulation

### Reward Function Definition

```
R: S × A × S → ℝ

R(s, a, s') = immediate reward for transitioning from s to s' via action a

Alternative formulations:
  R(s, a)    - Reward depends only on state-action
  R(s)       - Reward depends only on state
  R(s, a, s') - Full specification (most general)
```

### Expected Reward

```
r(s, a) = E[R(s, a, S')] = Σ_{s'} P(s'|s, a) R(s, a, s')

This is the expected immediate reward for taking action a in state s.
```

---

## 📐 Return and Value Functions

### Discounted Return

```
G_t = R_{t+1} + γR_{t+2} + γ²R_{t+3} + ...
    = Σ_{k=0}^∞ γ^k R_{t+k+1}

Properties:
  1. Finite if γ < 1 and rewards bounded: |G_t| ≤ R_max/(1-γ)
  2. Recursive: G_t = R_{t+1} + γG_{t+1}
```

### Value Function Derivation

```
V^π(s) = E_π[G_t | S_t = s]
       = E_π[R_{t+1} + γG_{t+1} | S_t = s]
       = E_π[R_{t+1} | S_t = s] + γ E_π[G_{t+1} | S_t = s]

By tower property and Markov:
       = Σ_a π(a|s) r(s,a) + γ Σ_a π(a|s) Σ_{s'} P(s'|s,a) V^π(s')
       = Σ_a π(a|s) [r(s,a) + γ Σ_{s'} P(s'|s,a) V^π(s')]
```

---

## 📐 Reward Shaping Theory

### Potential-Based Shaping

```
Theorem (Ng et al., 1999): Potential-based reward shaping preserves 
optimal policies.

Shaped reward:
  R'(s, a, s') = R(s, a, s') + γΦ(s') - Φ(s)

Where Φ: S → ℝ is any potential function.

Proof sketch:
  Let G'_t be return under R'. Then:
  G'_t = Σ_{k=0}^∞ γ^k [R_{t+k+1} + γΦ(S_{t+k+2}) - Φ(S_{t+k+1})]
       = Σ_{k=0}^∞ γ^k R_{t+k+1} + Σ_{k=0}^∞ γ^{k+1}Φ(S_{t+k+2}) - Σ_{k=0}^∞ γ^k Φ(S_{t+k+1})
       = G_t + γΦ(S_∞) - Φ(S_t)
       = G_t - Φ(s)  (assuming terminal Φ = 0)
       
  So V'(s) = V(s) - Φ(s)
  Same ordering of policies: π* unchanged! ∎
```

### Non-Potential Shaping

```
Warning: Non-potential shaping can change optimal policy!

Example: 
  Original: R(s,a) = 1 at goal, 0 elsewhere
  Bad shaping: R'(s,a) = R(s,a) + 0.1 for action "left"
  
  Result: Agent prefers "left" even when suboptimal!
```

---

## 📐 Reward Sparsity Analysis

### Sparse Reward Problem

```
For goal-reaching task:
  R(s) = { 1  if s = s_goal
         { 0  otherwise

Expected reward per episode:
  E[Σ_t R_t] = P(reach goal)

If P(reach goal) ≈ 0 via random exploration:
  - Gradient ≈ 0 (no learning signal)
  - Credit assignment over long horizons
  - Exploration becomes critical
```

### Information-Theoretic View

```
Reward signal entropy:
  H(R) = -Σ_r P(R=r) log P(R=r)

Sparse reward: H(R) ≈ 0 (almost always 0)
Dense reward: H(R) > 0 (varied feedback)

More informative rewards → faster learning
But risk of reward hacking with dense rewards!
```

---

## 📐 Intrinsic Motivation

### Curiosity-Based Rewards

```
Intrinsic reward = prediction error

ICM (Intrinsic Curiosity Module):
  r_i(s, a, s') = ||ŝ' - s'||²
  
  Where ŝ' = f(s, a) is predicted next state.
  Novel states → high error → high reward.
```

### Count-Based Exploration

```
r_i(s) = β / √N(s)

Where N(s) = visit count for state s.
Less-visited states get higher bonus.

Theoretical basis: Upper Confidence Bound
  UCB(s,a) = Q(s,a) + c√(log t / N(s,a))
```

---

## ⚠️ Reward Hacking

```
Problem: Agent finds unintended ways to maximize reward

Example: Racing game with reward for speed
- Agent learns to go in circles (high speed, no progress)
- Agent finds walls that give infinite speed glitch

Solution: Careful reward design, human oversight, RLHF
```

---

## 💻 Code

```python
def sparse_reward(state, goal):
    """Reward only at goal - hard to learn"""
    return 1.0 if state == goal else 0.0

def dense_reward(state, goal):
    """Reward based on progress - easier"""
    distance = np.linalg.norm(state - goal)
    return -distance  # Closer = higher reward

def shaped_reward(state, prev_state, goal):
    """Potential-based shaping - preserves optimal policy"""
    phi = lambda s: -np.linalg.norm(s - goal)
    return phi(state) - phi(prev_state)
```

## 🔗 Where This Topic Is Used

| Application | Reward Design |
|-------------|--------------|
| **RLHF** | Human preference scores |
| **Game AI** | Win/lose signals |
| **Robotics** | Task completion bonus |
| **Recommendation** | Click/engagement |

## 📚 References

| Type | Resource | Link |
|------|----------|------|
| 📖 | Textbook | See parent folder |
| 🎥 | Video Lectures | YouTube/Coursera |

---

⬅️ [Back: Dynamics](../02_dynamics/) | ➡️ [Next: States & Actions](../04_states_actions/)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=F39C12&height=80&section=footer" width="100%"/>
</p>
