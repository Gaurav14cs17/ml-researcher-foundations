<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=F39C12&height=120&section=header&text=Dynamic%20Programming%20for%20RL&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-07-F39C12?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/dynamic-programming.svg" width="100%">

*Caption: DP methods require known MDP dynamics. Policy iteration alternates between evaluation and improvement. Value iteration combines both in one sweep.*

---

## 📂 Overview

Dynamic programming computes optimal policies when the MDP model (transitions and rewards) is fully known. It forms the theoretical foundation for all RL algorithms.

---

## 📐 Two Main Algorithms

### Policy Iteration

```
1. Initialize π₀ arbitrarily
2. Policy Evaluation: Compute V^π
   V(s) = Σₐ π(a|s) Σₛ' P(s'|s,a)[R + γV(s')]
   (iterate until convergence)
3. Policy Improvement: Update π
   π'(s) = argmax_a Σₛ' P(s'|s,a)[R + γV(s')]
4. If π changed, go to step 2
```

### Value Iteration

```
1. Initialize V₀ arbitrarily
2. V(s) ← max_a Σₛ' P(s'|s,a)[R(s,a,s') + γV(s')]
3. Repeat until convergence
4. Extract π(s) = argmax_a Q(s,a)
```

---

## 🔑 Key Properties

| Property | Policy Iteration | Value Iteration |
|----------|-----------------|-----------------|
| Convergence | Finite steps | Asymptotic |
| Per iteration | Expensive (eval) | Cheap |
| Total work | Often less | Often more |
| Intuition | Alternate E+I | Single sweep |

---

## 💻 Code

```python
def value_iteration(env, gamma=0.99, theta=1e-8):
    """Value iteration for tabular MDP"""
    n_states = env.n_states
    n_actions = env.n_actions
    V = np.zeros(n_states)
    
    while True:
        delta = 0
        for s in range(n_states):
            v = V[s]
            # Bellman optimality update
            V[s] = max(
                sum(env.P[s][a][s_next] * (r + gamma * V[s_next])
                    for s_next, r, _ in env.transitions(s, a))
                for a in range(n_actions)
            )
            delta = max(delta, abs(v - V[s]))
        
        if delta < theta:
            break
    
    # Extract optimal policy
    policy = np.zeros(n_states, dtype=int)
    for s in range(n_states):
        policy[s] = np.argmax([
            sum(env.P[s][a][s_next] * (r + gamma * V[s_next])
                for s_next, r, _ in env.transitions(s, a))
            for a in range(n_actions)
        ])
    
    return V, policy
```


## 🔗 Where This Topic Is Used

| Application | DP Method |
|-------------|----------|
| **Policy Evaluation** | Iterative update of V(s) |
| **Value Iteration** | Optimal value function |
| **Policy Iteration** | Alternating eval/improve |
| **Shortest Path** | Bellman-Ford algorithm |


## 📚 References

| Type | Resource | Link |
|------|----------|------|
| 📖 | Textbook | See parent folder |
| 🎥 | Video Lectures | YouTube/Coursera |
| 🇨🇳 | 中文资源 | 知乎/B站 |

---

⬅️ [Back: Value Methods](../)

---

⬅️ [Back: Dqn](../dqn/) | ➡️ [Next: Q Learning](../q-learning/)

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=F39C12&height=80&section=footer" width="100%"/>
</p>
