<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=F39C12&height=120&section=header&text=Planning%20in%20RL&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-07-F39C12?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

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

## 📐 Model Predictive Control (MPC)

```
At each timestep:
1. Use world model to simulate H steps ahead
2. Try many action sequences
3. Pick sequence with best predicted return
4. Execute only first action
5. Repeat

Why only first action?
- Model errors compound over time
- Re-planning corrects drift
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

⬅️ [Back: Model-Based](../)

---

⬅️ [Back: Mcts](../mcts/) | ➡️ [Next: World Models](../world-models/)

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=F39C12&height=80&section=footer" width="100%"/>
</p>
