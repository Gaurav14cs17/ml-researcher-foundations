# Reward Functions

> **The signal that tells the agent what to optimize**

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

## 📐 Return Formula

```
G_t = R_{t+1} + γR_{t+2} + γ²R_{t+3} + ...
    = Σₖ₌₀^∞ γᵏ R_{t+k+1}

Goal: Find policy π that maximizes E[G_t]
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
| 🇨🇳 | 中文资源 | 知乎/B站 |

---

⬅️ [Back: MDP](../)

---

⬅️ [Back: Dynamics](../dynamics/) | ➡️ [Next: States Actions](../states-actions/)
