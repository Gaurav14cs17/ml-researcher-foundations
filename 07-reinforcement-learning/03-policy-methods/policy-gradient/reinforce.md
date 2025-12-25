<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=REINFORCE&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---

# REINFORCE

> **The simplest policy gradient algorithm**

---

## 📐 Algorithm

```
For each episode:
    Generate trajectory τ = (s₀, a₀, r₀, s₁, a₁, r₁, ...)
    
    For each step t:
        R_t = Σₖ γᵏ r_{t+k}  # Return from t
        
    Update: θ ← θ + α Σₜ ∇log π_θ(aₜ|sₜ) · R_t
```

---

## 🔑 Policy Gradient Theorem

```
∇J(θ) = E_π[∇log π_θ(a|s) · Q^π(s,a)]

REINFORCE uses Monte Carlo estimate:
Q^π(s,a) ≈ R_t (sample return)
```

---

## ⚠️ High Variance

```
Problem: R_t has high variance
Solution: Subtract baseline b(s)

∇J(θ) = E[∇log π_θ(a|s) · (R_t - b(s))]

Common baseline: Value function V(s)
This gives Advantage: A(s,a) = R_t - V(s)
```

---

## 💻 Code

```python
def reinforce_update(policy, optimizer, states, actions, returns):
    """
    states: [T, state_dim]
    actions: [T]
    returns: [T] - discounted returns from each step
    """
    optimizer.zero_grad()
    
    # Log probabilities of taken actions
    log_probs = policy.log_prob(states, actions)
    
    # Policy gradient loss
    # Negative because we want to maximize
    loss = -(log_probs * returns).mean()
    
    loss.backward()
    optimizer.step()
```

---

## 📊 Comparison

| Aspect | REINFORCE | Actor-Critic |
|--------|-----------|--------------|
| Variance | High | Lower |
| Bias | None | Some |
| Sample efficiency | Low | Higher |

---

<- [Back](./README.md)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>
