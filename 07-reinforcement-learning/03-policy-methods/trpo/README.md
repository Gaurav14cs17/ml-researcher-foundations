# Trust Region Policy Optimization (TRPO)

> **Safe policy updates with KL constraints**

---

## 🎯 Visual Overview

<img src="./images/trpo.svg" width="100%">

*Caption: TRPO prevents policy collapse by constraining updates to a trust region defined by KL divergence. This ensures monotonic improvement.*

---

## 📂 Overview

TRPO is a policy gradient method that guarantees monotonic improvement by restricting how much the policy can change at each update.

---

## 📐 Key Concepts

```
Problem: Large policy updates can be catastrophic
Solution: Constrain KL divergence between old and new policy

TRPO Objective:
maximize L(θ) = E[π(a|s)/π_old(a|s) × A(s,a)]
subject to: KL(π_old || π) ≤ δ

δ = trust region size (typically 0.01)
```

---

## 🔑 Key Features

| Feature | Description |
|---------|-------------|
| **Trust Region** | KL constraint prevents large steps |
| **Natural Gradient** | Uses Fisher information matrix |
| **Conjugate Gradient** | Efficient constraint solving |
| **Monotonic Improvement** | Theoretical guarantees |

---

## 🆚 TRPO vs PPO

| Aspect | TRPO | PPO |
|--------|------|-----|
| Constraint | Hard KL constraint | Soft clipping |
| Computation | Conjugate gradient | Standard SGD |
| Complexity | High | Low |
| Performance | Similar | Similar |

---

## 💻 Code (Simplified)

```python
def trpo_update(policy, states, actions, advantages, old_log_probs, delta=0.01):
    """TRPO update with conjugate gradient"""
    
    def surrogate_loss(theta):
        log_probs = policy.log_prob(states, actions, theta)
        ratio = torch.exp(log_probs - old_log_probs)
        return (ratio * advantages).mean()
    
    def kl_divergence(theta):
        return kl_div(policy.old_params, theta, states).mean()
    
    # Compute gradients
    g = compute_gradient(surrogate_loss, policy.params)
    
    # Conjugate gradient to find step direction
    step_dir = conjugate_gradient(fisher_vector_product, g)
    
    # Line search to satisfy KL constraint
    step_size = np.sqrt(2 * delta / (step_dir @ fisher_vector_product(step_dir)))
    
    new_params = policy.params + step_size * step_dir
    
    # Backtrack if KL constraint violated
    while kl_divergence(new_params) > delta:
        step_size *= 0.5
        new_params = policy.params + step_size * step_dir
    
    policy.params = new_params
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | TRPO Paper | [arXiv](https://arxiv.org/abs/1502.05477) |
| 📖 | OpenAI Spinning Up TRPO | [Docs](https://spinningup.openai.com/en/latest/algorithms/trpo.html) |
| 📄 | Natural Policy Gradient | [Paper](https://papers.nips.cc/paper/2002/hash/5c04925674920eb58467fb52ce4ef728-Abstract.html) |
| 🇨🇳 | TRPO详解 | [知乎](https://zhuanlan.zhihu.com/p/26308073) |
| 🇨🇳 | 信赖域方法原理 | [CSDN](https://blog.csdn.net/qq_30615903/article/details/81275638) |
| 🇨🇳 | TRPO与PPO对比 | [B站](https://www.bilibili.com/video/BV1cP4y1Y7DN) |


## 🔗 Where This Topic Is Used

| Application | TRPO |
|-------------|-----|
| **Robotics** | Safe policy updates |
| **PPO Predecessor** | Led to simpler PPO |
| **Continuous Control** | MuJoCo benchmarks |
| **Safe RL** | Trust region constraints |

---

⬅️ [Back: Policy Methods](../)

---

⬅️ [Back: Ppo](../ppo/)
