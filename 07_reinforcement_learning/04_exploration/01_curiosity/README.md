<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=F39C12&height=120&section=header&text=Curiosity-Driven%20Exploration&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-07-F39C12?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## 🔗 Navigation

⬅️ [Back: Exploration](../) | ➡️ [Next: Epsilon-Greedy](../02_epsilon_greedy/)

---

## 🎯 Visual Overview

<img src="./images/curiosity-driven.svg" width="100%">

*Caption: ICM (Intrinsic Curiosity Module) generates curiosity rewards from prediction error. The forward model predicts next state features; high prediction error means surprise, which becomes intrinsic reward.*

---

## 📂 Overview

Curiosity-driven exploration rewards the agent for encountering surprising outcomes that it cannot predict. This solves hard exploration problems without any external reward.

---

## 📐 Mathematical Foundation

### Intrinsic Curiosity Module (ICM)

```
Components:
1. Feature Encoder: φ: S → ℝᵈ
   Maps raw states to learned feature space
   
2. Forward Model: f: ℝᵈ × A → ℝᵈ
   φ̂(s_{t+1}) = f(φ(s_t), a_t)
   Predicts next state features
   
3. Inverse Model: g: ℝᵈ × ℝᵈ → A
   â_t = g(φ(s_t), φ(s_{t+1}))
   Predicts action from state transitions
```

### Curiosity Reward Definition

```
r_i(s_t, a_t, s_{t+1}) = η/2 · ||φ̂(s_{t+1}) - φ(s_{t+1})||²₂

Where:
  η = scaling factor
  φ̂(s_{t+1}) = f(φ(s_t), a_t)  (predicted features)
  φ(s_{t+1}) = encoder output   (actual features)
  
Total reward: r_total = r_extrinsic + r_intrinsic
```

### Training Objective

```
L_ICM = (1-β)L_forward + βL_inverse

Where:
  L_forward = ||φ̂(s') - φ(s')||²
  L_inverse = CrossEntropy(â, a)
  β ∈ [0,1] = weighting factor (typically 0.2)

Why inverse model?
  Forces φ to encode action-relevant features
  Ignores noise that doesn't affect dynamics
```

### Theoretical Justification

```
Theorem: Feature space trained with inverse model ignores 
noise that is not controllable by actions.

Proof sketch:
  If feature z is independent of action a given s:
    P(z|s,a) = P(z|s)
  Then inverse model cannot use z to predict a
  So gradient ∂L_inverse/∂z = 0
  Feature z is not learned → ignored  ∎

This solves the "noisy TV problem":
  Random noise on screen → high prediction error
  But noise is action-independent → filtered out
```

---

## 📐 Random Network Distillation (RND)

### Alternative Curiosity Formulation

```
RND uses two networks:
  1. Target f: S → ℝᵈ  (random, fixed)
  2. Predictor f̂: S → ℝᵈ  (learned)

Intrinsic reward:
  r_i(s) = ||f(s) - f̂(s)||²

Intuition:
  Novel states → predictor hasn't seen them → high error
  Familiar states → predictor learned them → low error
```

### RND Loss Function

```
L_RND = E_s~D [||f(s) - f̂(s)||²]

Properties:
  1. Self-supervised (no labels needed)
  2. Density estimation: r_i ∝ 1/ρ(s)
  3. Non-episodic: works across episodes
```

### Comparison: ICM vs RND

```
| Aspect          | ICM                  | RND                |
|-----------------|----------------------|--------------------|
| Novelty signal  | Prediction error     | Prediction error   |
| State features  | Learned (inverse)    | Random (fixed)     |
| Noise handling  | Inverse model        | Inherent           |
| Complexity      | Higher               | Lower              |
| Performance     | Good on visual       | Better on Atari    |
```

---

## 🔑 Key Features

| Feature | Description |
|---------|-------------|
| **Feature Space** | Ignores noise (TV static problem) |
| **Self-supervised** | No labels needed |
| **Scalable** | Works with high-dim states |
| **Sparse Reward** | Solves Montezuma's Revenge |

---

## 🌍 Results

| Environment | Without Curiosity | With Curiosity |
|-------------|-------------------|----------------|
| Montezuma's Revenge | 0 | 11,500 |
| VizDoom | Random | Explores map |
| Mario | Stuck at start | Completes levels |

---

## 💻 Code

```python
class ICM(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        # Feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # Forward model: predict next features
        self.forward_model = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def curiosity_reward(self, state, action, next_state):
        phi_s = self.encoder(state)
        phi_s_next = self.encoder(next_state)
        
        # Predict next state features
        action_onehot = F.one_hot(action, num_classes=self.action_dim)
        phi_s_next_pred = self.forward_model(torch.cat([phi_s, action_onehot], dim=-1))
        
        # Curiosity = prediction error
        return ((phi_s_next_pred - phi_s_next.detach()) ** 2).mean(dim=-1)
```

## 🔗 Where This Topic Is Used

| Application | Curiosity |
|-------------|----------|
| **ICM** | Prediction error as reward |
| **RND** | Random network distillation |
| **Hard Games** | Sparse reward navigation |
| **Lifelong Learning** | Continuous exploration |

## 📚 References

| Type | Resource | Link |
|------|----------|------|
| 📖 | Textbook | See parent folder |
| 🎥 | Video Lectures | YouTube/Coursera |

---

⬅️ [Back: Exploration](../) | ➡️ [Next: Epsilon-Greedy](../02_epsilon_greedy/)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=F39C12&height=80&section=footer" width="100%"/>
</p>
