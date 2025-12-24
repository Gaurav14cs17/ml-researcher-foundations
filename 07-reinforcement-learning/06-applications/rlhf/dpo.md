# Direct Preference Optimization (DPO)

> **RLHF without the RL**

---

## 📐 Key Insight

```
Optimal policy under KL-constrained reward maximization:

π*(y|x) = π_ref(y|x) exp(r(x,y)/β) / Z(x)

Rearranging:
r(x,y) = β log(π*(y|x)/π_ref(y|x)) + β log Z(x)
```

---

## 📐 DPO Loss

```
L_DPO(π_θ) = -E[log σ(β log(π_θ(y_w|x)/π_ref(y_w|x)) 
                     - β log(π_θ(y_l|x)/π_ref(y_l|x)))]

Where:
• y_w = preferred response
• y_l = rejected response
• π_ref = reference (SFT) policy
• β = temperature
```

---

## 🔑 Advantages

| vs RLHF | Benefit |
|---------|---------|
| No reward model | Simpler pipeline |
| No sampling | More efficient |
| No RL | More stable |
| Single stage | Faster |

---

## 💻 Code

```python
def dpo_loss(policy, ref_policy, chosen, rejected, beta=0.1):
    """
    Direct Preference Optimization loss
    """
    # Log probs for chosen
    logp_chosen = policy.log_prob(chosen)
    logp_chosen_ref = ref_policy.log_prob(chosen)
    
    # Log probs for rejected
    logp_rejected = policy.log_prob(rejected)
    logp_rejected_ref = ref_policy.log_prob(rejected)
    
    # Log ratios
    chosen_ratio = logp_chosen - logp_chosen_ref
    rejected_ratio = logp_rejected - logp_rejected_ref
    
    # DPO loss
    loss = -F.logsigmoid(beta * (chosen_ratio - rejected_ratio)).mean()
    
    return loss
```

---

## 📊 Comparison

| Method | Reward Model | RL | Stability |
|--------|--------------|------|-----------|
| RLHF | Yes | Yes (PPO) | Medium |
| DPO | No | No | High |
| IPO | No | No | Higher |
| KTO | No | No | High |

---

---

➡️ [Next: Reward Model](./reward-model.md)
