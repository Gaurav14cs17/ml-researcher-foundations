<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=F39C12&height=120&section=header&text=Discount%20Factor%20γ&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-07-F39C12?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/discount-factor.svg" width="100%">

*Caption: The discount factor γ ∈ [0,1] determines how much we value future rewards. With γ=0.9, a reward 10 steps away is worth only 0.35 of its face value today. Higher γ = more far-sighted agent.*

---

## 📂 Overview

The discount factor controls the agent's "patience" - how much it values future rewards compared to immediate ones.

---

## 🔑 Key Concepts

| γ Value | Behavior | Use Case |
|---------|----------|----------|
| **γ = 0** | Myopic, only immediate reward | Bandits |
| **γ = 0.9** | Balanced, typical | Most RL |
| **γ = 0.99** | Far-sighted | Long episodes |
| **γ = 1** | Undiscounted | Episodic, finite |

---

## 📐 Mathematical Role

```
Discounted Return:
G_t = R_{t+1} + γR_{t+2} + γ²R_{t+3} + ...

Why Discount?
1. Mathematical: Ensures G_t is finite for γ < 1
2. Economic: Future is uncertain, prefer now
3. Practical: Faster learning convergence

Effective Horizon: 1/(1-γ)
γ = 0.99 → looks ~100 steps ahead
γ = 0.9  → looks ~10 steps ahead
```

---

## 💻 Code

```python
def compute_returns(rewards, gamma=0.99):
    """Compute discounted returns for a trajectory"""
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns

# Example
rewards = [1, 1, 1, 10]  # Get 10 at the end
returns_099 = compute_returns(rewards, gamma=0.99)  # [12.9, 11.9, 10.9, 10]
returns_090 = compute_returns(rewards, gamma=0.9)   # [10.5, 10.6, 10.0, 10]
returns_000 = compute_returns(rewards, gamma=0.0)   # [1, 1, 1, 10]
```


## 🔗 Where This Topic Is Used

| Application | Discount Factor |
|-------------|----------------|
| **Finance** | Time value of money |
| **Long-horizon tasks** | γ close to 1 |
| **Short-term focus** | γ close to 0 |
| **Infinite horizon** | Ensures convergence |


## 📚 References

| Type | Resource | Link |
|------|----------|------|
| 📖 | Textbook | See parent folder |
| 🎥 | Video Lectures | YouTube/Coursera |
| 🇨🇳 | 中文资源 | 知乎/B站 |

---

⬅️ [Back: MDP](../)

---

➡️ [Next: Dynamics](../dynamics/)

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=F39C12&height=80&section=footer" width="100%"/>
</p>
