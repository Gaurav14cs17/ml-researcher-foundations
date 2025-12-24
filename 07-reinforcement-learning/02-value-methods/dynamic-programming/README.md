# Dynamic Programming for RL

> **Planning with known models**

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
