<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=F39C12&height=120&section=header&text=Transition%20Dynamics&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-07-F39C12?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

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

## 📐 Properties

```
Transition Function P: S × A → Δ(S)

1. P(s'|s,a) ≥ 0 for all s,a,s'    (non-negative)
2. Σₛ' P(s'|s,a) = 1 for all s,a    (sums to 1)
3. Markov: Future independent of past given present
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
| 🇨🇳 | 中文资源 | 知乎/B站 |

---

⬅️ [Back: MDP](../)

---

⬅️ [Back: Discounting](../discounting/) | ➡️ [Next: Rewards](../rewards/)

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=F39C12&height=80&section=footer" width="100%"/>
</p>
