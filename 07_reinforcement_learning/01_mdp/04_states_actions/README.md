<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=F39C12&height=120&section=header&text=States%20and%20Actions&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-07-F39C12?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## 🔗 Navigation

⬅️ [Back: Rewards](../03_rewards/) | ➡️ [Next: Value Methods](../../02_value_methods/)

---

## 🎯 Visual Overview

<img src="./images/states-actions.svg" width="100%">

*Caption: State space S contains all possible situations the agent can observe. Action space A contains all possible decisions the agent can make. These can be discrete (finite set) or continuous (real-valued).*

---

## 📂 Overview

States represent what the agent observes about the environment. Actions represent what the agent can do to change the environment.

---

## 🔑 Key Concepts

| Concept | Description |
|---------|-------------|
| **State s** | Complete description of environment at time t |
| **Action a** | Decision/control taken by agent |
| **State Space S** | Set of all possible states |
| **Action Space A** | Set of all possible actions |

---

## 📐 Mathematical Formalization

### State Space

```
S = {s₁, s₂, ..., sₙ}  (discrete, finite)
  or
S ⊆ ℝⁿ               (continuous)

Properties:
  1. Complete: Contains all distinguishable situations
  2. Markov: s_t encodes all relevant history
  3. Observable: Agent can perceive s_t at each step
```

### Action Space

```
A = {a₁, a₂, ..., aₘ}  (discrete, finite)
  or
A ⊆ ℝᵐ               (continuous)

Can be state-dependent: A(s) ⊆ A
  Example: Legal moves depend on board position
```

### Policy as Mapping

```
Deterministic policy: π: S → A
  a = π(s)

Stochastic policy: π: S → Δ(A)
  π(a|s) = P(A_t = a | S_t = s)
  
  Properties:
    π(a|s) ≥ 0  ∀a, s
    Σ_a π(a|s) = 1  ∀s
```

---

## 📐 State Representation Theory

### Sufficient Statistics

```
Theorem: A state representation φ(h_t) is sufficient if:

  P(R_{t+1}, S_{t+1} | φ(h_t), A_t) = P(R_{t+1}, S_{t+1} | h_t, A_t)

Where h_t = (S_0, A_0, R_1, S_1, ..., S_t) is history.

Sufficient statistics preserve Markov property.
```

### State Aggregation

```
Partition S into groups {G_1, G_2, ..., G_k}

Aggregated MDP is valid if for all s, s' ∈ G_i:
  P(s'' ∈ G_j | s, a) = P(s'' ∈ G_j | s', a)  ∀a, j
  R(s, a) = R(s', a)  ∀a

This preserves optimal value function on aggregate states.
```

---

## 📐 Continuous Spaces

### Discretization

```
For continuous S ⊆ ℝⁿ, discretize into grid:

S_discrete = {s_i}  where s_i = centers of grid cells

Trade-off:
  Fine grid: |S_discrete| large, curse of dimensionality
  Coarse grid: Loss of precision, suboptimal policies
```

### Function Approximation

```
Instead of discretizing, approximate value function:

V_θ(s) ≈ V^π(s)  for all s ∈ S

Common choices:
  - Linear: V_θ(s) = θᵀφ(s)
  - Neural network: V_θ(s) = NN_θ(s)
  
Advantage: Generalizes to unseen states
```

### Continuous Actions

```
For A ⊆ ℝᵐ, policy outputs parameters of distribution:

Gaussian policy:
  π_θ(a|s) = N(a; μ_θ(s), σ_θ(s)²)
  
  μ_θ(s) = neural network output
  σ_θ(s) = learned or fixed variance

Sampling: a ~ π_θ(·|s)
```

---

## 📐 Complexity Analysis

### Tabular Methods

```
Space complexity: O(|S| × |A|) for Q-table

Time per update: O(1)

Total for value iteration:
  O(|S|² × |A| × 1/(1-γ) × log(1/ε))
  
Curse of dimensionality: 
  If S ⊆ ℝⁿ discretized with k bins per dimension:
  |S| = kⁿ (exponential in dimension!)
```

### Deep RL

```
Parameter count: O(hidden_dims²)  (for MLP)
                 O(filters × kernel²)  (for CNN)

Time per update: O(batch_size × parameter_count)

Generalization to unseen states enables tractable learning
in high-dimensional spaces.
```

---

## 📐 Types of Spaces

| Type | State Examples | Action Examples |
|------|----------------|-----------------|
| **Discrete** | Grid positions, game boards | Left/Right/Up/Down |
| **Continuous** | Robot joint angles, velocity | Force, torque values |
| **High-Dim** | Images (84×84×4) | Multi-joint control |
| **Hybrid** | Mixed discrete + continuous | Discrete choice + continuous param |

---

## 💻 Code

```python
import gymnasium as gym

# Discrete: CartPole
env = gym.make("CartPole-v1")
print(f"State: {env.observation_space}")  # Box(4,) - continuous
print(f"Actions: {env.action_space}")     # Discrete(2) - left/right

# Continuous: MuJoCo
env = gym.make("HalfCheetah-v4")
print(f"State: {env.observation_space}")  # Box(17,) - joint positions/velocities
print(f"Actions: {env.action_space}")     # Box(6,) - continuous torques
```

## 🔗 Where This Topic Is Used

| Application | How States/Actions Are Used |
|-------------|---------------------------|
| **Game Playing** | Board state → legal moves |
| **Robotics** | Joint positions → torques |
| **Trading** | Market state → buy/sell |
| **Dialogue** | Conversation history → responses |

## 📚 References

| Type | Resource | Link |
|------|----------|------|
| 📖 | Textbook | See parent folder |
| 🎥 | Video Lectures | YouTube/Coursera |

---

⬅️ [Back: Rewards](../03_rewards/) | ➡️ [Next: Value Methods](../../02_value_methods/)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=F39C12&height=80&section=footer" width="100%"/>
</p>
