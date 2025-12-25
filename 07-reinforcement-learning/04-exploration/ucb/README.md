<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=F39C12&height=120&section=header&text=Upper%20Confidence%20Bound%20UCB&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-07-F39C12?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/ucb.svg" width="100%">

*Caption: UCB selects actions by adding an exploration bonus to the estimated value. Actions tried less often have higher uncertainty and thus higher bonus. This guarantees exploration of all actions.*

---

## 📂 Overview

UCB is a principled exploration strategy from bandit theory. It balances exploitation (choosing high-value actions) with exploration (trying uncertain actions).

---

## 📐 Formula

```
UCB(a) = Q(a) + c * √(ln(t) / N(a))

Where:
- Q(a): Estimated value of action a
- N(a): Number of times action a was selected
- t: Total number of steps
- c: Exploration weight (typically √2)

Select: a* = argmax UCB(a)
```

---

## 🔑 Key Insight

```
Low N(a) → High uncertainty bonus → Gets selected
High N(a) → Low uncertainty bonus → Needs high Q(a)

Result: All actions eventually tried, but good ones
        tried more often. No random exploration!
```

---

## 💻 Code

```python
import numpy as np

def ucb_select(q_values, counts, t, c=2.0):
    """Select action using UCB"""
    # Avoid division by zero for unvisited actions
    ucb_values = np.where(
        counts > 0,
        q_values + c * np.sqrt(np.log(t) / counts),
        float('inf')  # Always try unvisited actions
    )
    return np.argmax(ucb_values)

# Usage
t = 0
for episode in range(1000):
    t += 1
    action = ucb_select(q_values, counts, t)
    # ... observe reward ...
    counts[action] += 1
    q_values[action] += (reward - q_values[action]) / counts[action]
```


## 🔗 Where This Topic Is Used

| Application | UCB |
|-------------|-----|
| **Multi-Armed Bandits** | Optimal exploration |
| **MCTS** | UCT for tree search |
| **Hyperparameter Tuning** | Bayesian optimization |
| **Clinical Trials** | Adaptive allocation |


## 📚 References

| Type | Resource | Link |
|------|----------|------|
| 📖 | Textbook | See parent folder |
| 🎥 | Video Lectures | YouTube/Coursera |
| 🇨🇳 | 中文资源 | 知乎/B站 |

---

⬅️ [Back: Exploration](../)

---

⬅️ [Back: Intrinsic](../intrinsic/)

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=F39C12&height=80&section=footer" width="100%"/>
</p>
