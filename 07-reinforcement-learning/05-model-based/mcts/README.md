<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=Mcts&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---


# Monte Carlo Tree Search (MCTS)

> **Intelligent tree search for decision making**

---

## 🎯 Visual Overview

<img src="./images/mcts.svg" width="100%">

*Caption: MCTS builds a search tree through repeated selection, expansion, simulation, and backpropagation. UCT balances exploration and exploitation.*

---

## 📂 Overview

MCTS is the algorithm behind AlphaGo's superhuman performance. It efficiently searches large action spaces by focusing on promising branches.

---

## 📐 Four Phases

| Phase | Description |
|-------|-------------|
| **1. Selection** | Traverse tree using UCT until leaf |
| **2. Expansion** | Add child nodes at leaf |
| **3. Simulation** | Random rollout to estimate value |
| **4. Backpropagation** | Update statistics up the tree |

---

## 🔑 UCT Formula

```
UCT(s, a) = Q(s, a) + c × √(ln N(s) / N(s, a))

Q(s, a): Average value from action a at state s
N(s): Visit count for state s
N(s, a): Visit count for action a at state s
c: Exploration constant (often √2)
```

---

## 🌍 MCTS + Neural Networks

| System | Network Role |
|--------|--------------|
| **AlphaGo** | Policy prior + value estimate |
| **AlphaZero** | Replace rollouts with value net |
| **MuZero** | Learned dynamics model |

---

## 💻 Code

```python
class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = 0
    
    def uct(self, c=1.41):
        if self.visits == 0:
            return float('inf')
        return (self.value / self.visits + 
                c * np.sqrt(np.log(self.parent.visits) / self.visits))
    
    def select_child(self):
        return max(self.children.values(), key=lambda n: n.uct())
    
    def backpropagate(self, value):
        self.visits += 1
        self.value += value
        if self.parent:
            self.parent.backpropagate(value)

def mcts(root, model, n_simulations=800):
    for _ in range(n_simulations):
        node = root
        # Selection
        while node.children and not is_terminal(node.state):
            node = node.select_child()
        # Expansion & Simulation
        if not is_terminal(node.state):
            expand(node, model)
            value = simulate(node.state, model)
        else:
            value = get_outcome(node.state)
        # Backpropagation
        node.backpropagate(value)
    return best_child(root)
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | AlphaGo Paper | [Nature](https://www.nature.com/articles/nature16961) |
| 📄 | AlphaZero Paper | [Science](https://www.science.org/doi/10.1126/science.aar6404) |
| 📄 | MuZero Paper | [Nature](https://www.nature.com/articles/s41586-020-03051-4) |
| 🇨🇳 | MCTS详解 | [知乎](https://zhuanlan.zhihu.com/p/25345778) |
| 🇨🇳 | AlphaGo原理 | [CSDN](https://blog.csdn.net/qq_30615903/article/details/80952771) |
| 🇨🇳 | 蒙特卡洛树搜索 | [B站](https://www.bilibili.com/video/BV1C34y1H7Eq) |


## 🔗 Where This Topic Is Used

| Application | MCTS |
|-------------|-----|
| **AlphaGo/AlphaZero** | Game tree search |
| **MuZero** | Planning with learned model |
| **Game AI** | General game playing |
| **Decision Making** | Complex planning |

---

⬅️ [Back: Model-Based](../)

---

⬅️ [Back: Dreamer](../dreamer/) | ➡️ [Next: Planning](../planning/)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>
