<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=F39C12&height=120&section=header&text=Transition%20Dynamics&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-07-F39C12?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## 🔗 Navigation

⬅️ [Back: Discounting](../01_discounting/) | ➡️ [Next: Rewards](../03_rewards/)

---

## 🎯 Visual Overview

<img src="./images/transition-dynamics.svg" width="100%">

*Caption: Transition dynamics P(s'|s,a) define the probability of reaching state s' when taking action a in state s. Dynamics can be deterministic (same outcome) or stochastic (random outcomes). The Markov property means transitions only depend on current state.*

---

## 📂 Overview

Transition dynamics define how the environment evolves. They are essential for model-based RL and planning.

---

## 🔑 Key Concepts

| Concept | Description |
|---------|-------------|
| **P(s'\|s,a)** | Probability of next state given current state and action |
| **Deterministic** | P(s'\|s,a) = 1 for exactly one s' |
| **Stochastic** | Distribution over multiple next states |
| **Markov Property** | P(s'\|s,a) = P(s'\|s₀,a₀,...,s,a) |

---

## 📐 Mathematical Definition

### Transition Probability Function

```
P: S × A × S → [0, 1]

P(s'|s, a) = Pr(S_{t+1} = s' | S_t = s, A_t = a)

"Probability of transitioning to state s' given current state s and action a"
```

### Formal Properties

```
1. Non-negativity: 
   P(s'|s,a) ≥ 0  ∀s ∈ S, a ∈ A, s' ∈ S

2. Normalization (probability distribution):
   Σ_{s' ∈ S} P(s'|s,a) = 1  ∀s ∈ S, a ∈ A

3. Markov Property:
   P(S_{t+1}|S_t, A_t, S_{t-1}, A_{t-1}, ..., S_0, A_0) = P(S_{t+1}|S_t, A_t)
   
   "The future depends only on the present, not the past"
```

---

## 📐 The Markov Property: Proof of Importance

### Theorem: Markov Property Enables Recursive Value Computation

```
Claim: If P satisfies the Markov property, then:
  V^π(s) = Σ_a π(a|s) [R(s,a) + γ Σ_{s'} P(s'|s,a) V^π(s')]

Proof:
  V^π(s) = E_π[G_t | S_t = s]
         = E_π[R_{t+1} + γG_{t+1} | S_t = s]
         = E_π[R_{t+1} | S_t = s] + γ E_π[G_{t+1} | S_t = s]
         
  By Markov property, G_{t+1} only depends on S_{t+1}:
         = E_π[R_{t+1} | S_t = s] + γ E_π[E_π[G_{t+1} | S_{t+1}] | S_t = s]
         = E_π[R_{t+1} | S_t = s] + γ E_π[V^π(S_{t+1}) | S_t = s]
         
  Expanding expectations:
         = Σ_a π(a|s) R(s,a) + γ Σ_a π(a|s) Σ_{s'} P(s'|s,a) V^π(s')
         = Σ_a π(a|s) [R(s,a) + γ Σ_{s'} P(s'|s,a) V^π(s')]  ∎
```

### Why Markov Property Matters

```
Without Markov property:
  V(s) would depend on entire history h = (s_0, a_0, s_1, a_1, ..., s_t)
  State space becomes exponentially large: |S|^t possibilities
  
With Markov property:
  V(s) only depends on current state s
  State space is fixed: |S| states
  Enables tractable algorithms (DP, TD, etc.)
```

---

## 📐 Transition Matrix Representation

### For Finite MDPs

```
For fixed policy π, define transition matrix P^π:

P^π[i,j] = Σ_a π(a|s_i) P(s_j|s_i, a)

This is a stochastic matrix (rows sum to 1).

Value function satisfies:
  V^π = R^π + γ P^π V^π
  
Solving:
  V^π = (I - γP^π)^{-1} R^π

Where R^π[i] = Σ_a π(a|s_i) R(s_i, a)
```

### Eigenvalue Analysis

```
Theorem: P^π has eigenvalue 1 with eigenvector 1 (all ones).

Proof: 
  (P^π)ᵀ 1 = 1  (columns of Pᵀ sum to 1)
  So 1 is eigenvalue of (P^π)ᵀ, hence of P^π.

Consequence: 
  Stationary distribution d^π exists where (P^π)ᵀ d^π = d^π
  This is the long-run state distribution under policy π.
```

---

## 📐 Deterministic vs Stochastic Dynamics

### Deterministic Dynamics

```
P(s'|s,a) ∈ {0, 1}  for all s, a, s'

Transition function f: S × A → S
  s' = f(s, a)

Examples:
  • Chess, Go (game rules)
  • Idealized physics simulations
  • Deterministic control systems
```

### Stochastic Dynamics

```
P(s'|s,a) ∈ [0, 1]  (non-trivial distribution)

Examples:
  • Real-world robotics (noise, uncertainty)
  • Games with chance elements
  • Market dynamics
  
Modeling: Often use Gaussian transitions
  s' ~ N(f(s,a), Σ(s,a))
```

---

## 📐 Learning Dynamics (Model-Based RL)

### Maximum Likelihood Estimation

```
Given dataset D = {(s_i, a_i, s'_i)}_{i=1}^N

MLE estimate:
  P̂(s'|s,a) = Count(s,a,s') / Count(s,a)
  
  Where:
    Count(s,a,s') = Σ_i 𝟙[s_i=s, a_i=a, s'_i=s']
    Count(s,a) = Σ_i 𝟙[s_i=s, a_i=a]
```

### Neural Network Dynamics Model

```
Learn f_θ: S × A → S (deterministic)
  or p_θ(s'|s,a) (probabilistic)

Loss function:
  L(θ) = E_{(s,a,s')~D}[||s' - f_θ(s,a)||²]  (deterministic)
  L(θ) = -E_{(s,a,s')~D}[log p_θ(s'|s,a)]   (probabilistic)
```

---

## 🌍 Known vs Unknown Dynamics

| Known (Model-Based) | Unknown (Model-Free) |
|---------------------|---------------------|
| Can plan ahead | Must learn from experience |
| Simulate trajectories | Trial and error |
| More sample efficient | More general |
| Games, physics sims | Real world |

---

## 💻 Code

```python
# Deterministic dynamics (simple example)
def transition_deterministic(state, action):
    if action == "right":
        return state + 1
    elif action == "left":
        return state - 1
    return state

# Stochastic dynamics
def transition_stochastic(state, action):
    """Action succeeds 80% of time, fails 20%"""
    if np.random.rand() < 0.8:
        return intended_next_state(state, action)
    else:
        return random_adjacent_state(state)
```

## 🔗 Where This Topic Is Used

| Application | Dynamics Model |
|-------------|---------------|
| **Model-Based RL** | Learned transition model |
| **Planning** | Simulator for lookahead |
| **Robotics** | Physics simulation |
| **Games** | Game rules as transitions |

## 📚 References

| Type | Resource | Link |
|------|----------|------|
| 📖 | Textbook | See parent folder |
| 🎥 | Video Lectures | YouTube/Coursera |

---

⬅️ [Back: Discounting](../01_discounting/) | ➡️ [Next: Rewards](../03_rewards/)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=F39C12&height=80&section=footer" width="100%"/>
</p>
