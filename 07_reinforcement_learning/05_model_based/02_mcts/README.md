<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=F39C12&height=120&section=header&text=Monte%20Carlo%20Tree%20Search%20(MCTS)&fontSize=28&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-07-F39C12?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## 🔗 Navigation

⬅️ [Back: Dreamer](../01_dreamer/) | ➡️ [Next: Planning](../03_planning/)

---

## 🎯 Visual Overview

<img src="./images/mcts.svg" width="100%">

*Caption: MCTS builds a search tree through repeated selection, expansion, simulation, and backpropagation. UCT balances exploration and exploitation. This is the algorithm that powered AlphaGo's victory.*

---

## 📂 Overview

Monte Carlo Tree Search (MCTS) is a planning algorithm that builds a search tree using random simulations. Combined with neural networks in AlphaGo/AlphaZero, it achieved superhuman performance in Go, Chess, and Shogi.

---

## 📐 The Four Phases

MCTS repeats four phases for each simulation:

```
1. SELECTION:    Traverse tree using UCT until reaching a leaf

2. EXPANSION:    Add one or more child nodes to the leaf

3. SIMULATION:   Random rollout from new node to terminal state

4. BACKPROPAGATION: Update statistics along the path

```

---

## 📐 UCT (Upper Confidence Bound for Trees)

### Formula

```
UCT(s, a) = Q(s, a) + c · √(ln N(s) / N(s, a))

Where:
• Q(s, a) = W(s, a) / N(s, a)  (average value of action a at state s)
• N(s) = number of visits to state s
• N(s, a) = number of times action a was taken at s
• c = exploration constant (typically √2 or 1.41)

```

### Intuition

```
UCT = Exploitation + Exploration
      Q(s,a)         c√(ln N(s) / N(s,a))

• Q(s,a): Choose actions that have worked well
• Exploration term: Choose actions tried fewer times
• As N(s,a) → ∞, exploration term → 0 (converge to best action)

```

### Theoretical Guarantee

```
Theorem: UCT converges to minimax-optimal play as N → ∞

The exploration term ensures:

1. Every action is tried infinitely often

2. Best actions are tried exponentially more often

3. Regret is O(√N log N)

```

---

## 📐 MCTS Algorithm

### Complete Algorithm

```
function MCTS(root_state, n_simulations):
    root = create_node(root_state)
    
    for i in 1 to n_simulations:
        node = root
        state = root_state.copy()
        
        # 1. SELECTION - traverse tree using UCT
        while node is fully expanded and not terminal:
            action = argmax_a UCT(node, a)
            state = state.apply(action)
            node = node.children[action]
        
        # 2. EXPANSION - add a new child node
        if not terminal(state):
            action = random unexpanded action
            state = state.apply(action)
            child = create_node(state)
            node.children[action] = child
            node = child
        
        # 3. SIMULATION - random rollout
        while not terminal(state):
            action = random_action(state)
            state = state.apply(action)
        value = get_outcome(state)
        
        # 4. BACKPROPAGATION - update statistics
        while node is not None:
            node.N += 1
            node.W += value
            node = node.parent
            value = -value  # For two-player games
    
    return argmax_a N(root, a)  # Return most visited action

```

---

## 📐 MCTS + Neural Networks (AlphaGo)

### Neural Network Guided MCTS

AlphaGo replaces random rollouts with neural network evaluations:

```
Policy Network:  p(a|s) = probability of action a in state s
Value Network:   v(s) = probability of winning from state s

Modified UCT (PUCT - Polynomial UCT):

PUCT(s, a) = Q(s, a) + c_puct · p(a|s) · √N(s) / (1 + N(s, a))

Key differences from vanilla UCT:

1. Policy prior p(a|s) guides exploration toward good moves

2. Value network v(s) replaces random rollouts

3. No simulation phase needed!

```

### AlphaGo MCTS Algorithm

```
function AlphaGoMCTS(root_state, n_simulations):
    root = create_node(root_state)
    p, v = neural_network(root_state)  # Prior and value
    root.P = p  # Store action priors
    
    for i in 1 to n_simulations:
        node = root
        state = root_state.copy()
        path = [root]
        
        # 1. SELECTION with PUCT
        while node is expanded:
            action = argmax_a PUCT(node, a)
            state = state.apply(action)
            node = node.children[action]
            path.append(node)
        
        # 2. EXPANSION & EVALUATION (no simulation!)
        p, v = neural_network(state)
        node.P = p
        
        # 3. BACKPROPAGATION
        for node in reversed(path):
            node.N += 1
            node.W += v
            node.Q = node.W / node.N
            v = -v  # Flip for opponent
    
    # Return action proportional to visit counts
    return sample_action(root, temperature)

```

---

## 📐 AlphaZero Improvements

### Key Differences from AlphaGo

```
1. No human expert data: Learns entirely from self-play

2. Single neural network: Combined policy + value head

3. No handcrafted features: Raw board as input

4. Simpler search: No random rollouts, pure PUCT

Training loop:

1. Self-play: Generate games using MCTS

2. Train network: Minimize loss on (state, policy, value) tuples

3. Repeat

```

### Training Objective

```
Loss = (z - v(s))² - π^T log p(s) + c||θ||²

Where:
• z = game outcome (+1 win, -1 loss)
• v(s) = value network prediction
• π = MCTS search probabilities
• p(s) = policy network output
• c||θ||² = L2 regularization

```

---

## 📐 Theoretical Properties

### Convergence

```
Theorem: As N → ∞, MCTS with UCT:

1. Visits each action infinitely often

2. Converges to optimal minimax value

3. Selects optimal action with probability → 1

Proof sketch:

- UCT ensures infinite exploration (log term)

- Monte Carlo estimates converge (LLN)

- Tree policy becomes optimal

```

### Regret Bound

```
For UCT with c = √2:

Regret ≤ O(K^D log N)

Where:
• K = branching factor
• D = search depth
• N = number of simulations

This is much better than exhaustive search O(K^D).

```

---

## 💻 Complete Implementation

```python
import numpy as np
from collections import defaultdict
import math

class MCTSNode:
    """Node in the MCTS tree"""
    
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action  # Action that led to this node
        
        self.children = {}  # action -> MCTSNode
        self.N = 0  # Visit count
        self.W = 0  # Total value
        self.Q = 0  # Mean value (W/N)
        
        # For AlphaZero-style
        self.P = None  # Prior probabilities
    
    def is_expanded(self):
        """Check if all children are expanded"""
        return len(self.children) > 0
    
    def is_terminal(self):
        """Check if game is over"""
        return self.state.is_terminal()
    
    def uct_value(self, c=1.41):
        """UCT value for action selection"""
        if self.N == 0:
            return float('inf')
        return self.Q + c * math.sqrt(math.log(self.parent.N) / self.N)
    
    def puct_value(self, c_puct=1.0):
        """PUCT value (AlphaGo style)"""
        if self.N == 0:
            return float('inf')
        
        # Get prior from parent's P
        prior = self.parent.P[self.action] if self.parent.P is not None else 1.0
        exploration = c_puct * prior * math.sqrt(self.parent.N) / (1 + self.N)
        return self.Q + exploration
    
    def select_child(self, use_puct=False):
        """Select child with highest UCT/PUCT value"""
        if use_puct:
            return max(self.children.values(), key=lambda n: n.puct_value())
        return max(self.children.values(), key=lambda n: n.uct_value())

class MCTS:
    """Monte Carlo Tree Search"""
    
    def __init__(self, n_simulations=800, c=1.41):
        self.n_simulations = n_simulations
        self.c = c
    
    def search(self, state):
        """Run MCTS and return best action"""
        root = MCTSNode(state)
        
        for _ in range(self.n_simulations):
            node = root
            sim_state = state.copy()
            
            # 1. SELECTION
            while node.is_expanded() and not node.is_terminal():
                node = node.select_child()
                sim_state = sim_state.apply(node.action)
            
            # 2. EXPANSION
            if not node.is_terminal():
                actions = sim_state.get_legal_actions()
                for action in actions:
                    next_state = sim_state.apply(action)
                    child = MCTSNode(next_state, parent=node, action=action)
                    node.children[action] = child
                
                # Select random child for simulation
                action = np.random.choice(actions)
                node = node.children[action]
                sim_state = sim_state.apply(action)
            
            # 3. SIMULATION (random rollout)
            while not sim_state.is_terminal():
                action = np.random.choice(sim_state.get_legal_actions())
                sim_state = sim_state.apply(action)
            
            value = sim_state.get_result()  # +1 win, -1 loss, 0 draw
            
            # 4. BACKPROPAGATION
            while node is not None:
                node.N += 1
                node.W += value
                node.Q = node.W / node.N
                node = node.parent
                value = -value  # Flip for opponent
        
        # Return most visited action
        return max(root.children.items(), key=lambda x: x[1].N)[0]

class AlphaZeroMCTS:
    """AlphaZero-style MCTS with neural network"""
    
    def __init__(self, network, n_simulations=800, c_puct=1.0):
        self.network = network
        self.n_simulations = n_simulations
        self.c_puct = c_puct
    
    def search(self, state, temperature=1.0):
        """Run MCTS with neural network guidance"""
        root = MCTSNode(state)
        
        # Initial network evaluation
        policy, value = self.network(state)
        root.P = policy
        
        for _ in range(self.n_simulations):
            node = root
            sim_state = state.copy()
            path = [node]
            
            # 1. SELECTION with PUCT
            while node.is_expanded() and not node.is_terminal():
                node = node.select_child(use_puct=True)
                sim_state = sim_state.apply(node.action)
                path.append(node)
            
            # 2. EXPANSION & EVALUATION
            if not node.is_terminal():
                policy, value = self.network(sim_state)
                node.P = policy
                
                actions = sim_state.get_legal_actions()
                for action in actions:
                    next_state = sim_state.apply(action)
                    child = MCTSNode(next_state, parent=node, action=action)
                    node.children[action] = child
            else:
                value = sim_state.get_result()
            
            # 3. BACKPROPAGATION (no simulation phase!)
            for node in reversed(path):
                node.N += 1
                node.W += value
                node.Q = node.W / node.N
                value = -value
        
        # Compute search probabilities
        visit_counts = np.array([
            root.children[a].N if a in root.children else 0
            for a in range(state.action_space)
        ])
        
        if temperature == 0:
            # Deterministic: pick most visited
            action = np.argmax(visit_counts)
        else:
            # Stochastic selection
            probs = visit_counts ** (1 / temperature)
            probs = probs / probs.sum()
            action = np.random.choice(len(probs), p=probs)
        
        # Return action and search probabilities (for training)
        search_probs = visit_counts / visit_counts.sum()
        return action, search_probs
    
    def self_play_game(self, state, temp_threshold=30):
        """Generate self-play game for training"""
        game_data = []
        move_count = 0
        
        while not state.is_terminal():
            # Use temperature 1 early, 0 late
            temperature = 1.0 if move_count < temp_threshold else 0.0
            action, search_probs = self.search(state, temperature)
            
            game_data.append({
                'state': state.copy(),
                'probs': search_probs,
                'player': state.current_player
            })
            
            state = state.apply(action)
            move_count += 1
        
        # Assign game result to all positions
        result = state.get_result()
        for data in game_data:
            # Result from perspective of player who made the move
            data['value'] = result if data['player'] == 0 else -result
        
        return game_data

```

---

## 📊 MCTS Evolution

| System | Year | Key Innovation | Achievement |
|--------|------|----------------|-------------|
| **Basic MCTS** | 2006 | UCT for tree search | Computer Go breakthrough |
| **AlphaGo** | 2016 | Policy/value networks + MCTS | Beat world champion |
| **AlphaGo Zero** | 2017 | No human data, self-play only | Stronger than AlphaGo |
| **AlphaZero** | 2017 | Generalized to Chess, Shogi | Master multiple games |
| **MuZero** | 2019 | Learned dynamics model | No game rules needed |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | AlphaGo Paper | [Nature](https://www.nature.com/articles/nature16961) |
| 📄 | AlphaZero Paper | [Science](https://www.science.org/doi/10.1126/science.aar6404) |
| 📄 | MuZero Paper | [Nature](https://www.nature.com/articles/s41586-020-03051-4) |
| 📄 | UCT Paper | [Kocsis & Szepesvári 2006](https://link.springer.com/chapter/10.1007/11871842_29) |
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
| **Robotics** | Motion planning |

---

⬅️ [Back: Dreamer](../01_dreamer/) | ➡️ [Next: Planning](../03_planning/)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=F39C12&height=80&section=footer" width="100%"/>
</p>
