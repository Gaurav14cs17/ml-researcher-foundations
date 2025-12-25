<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=Reward%20Modeling%20for%20RLHF&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---

# Reward Modeling for RLHF

> **Learning rewards from human preferences**

---

## 📐 Preference Learning

```
Given pairs (y_w, y_l) where y_w is preferred:

P(y_w > y_l | x) = σ(r(x, y_w) - r(x, y_l))

Bradley-Terry model

Loss: -E[log σ(r(y_w) - r(y_l))]
```

---

## 🔑 Training Pipeline

```
1. Generate responses from policy
2. Collect human preferences (A vs B)
3. Train reward model on preferences
4. Use reward model for RL fine-tuning
```

---

## ⚠️ Challenges

| Challenge | Solution |
|-----------|----------|
| Expensive annotations | Active learning |
| Reward hacking | KL penalty |
| Distribution shift | Iterative training |
| Inconsistent humans | Model uncertainty |

---

## 💻 Code

```python
class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.backbone = base_model
        self.head = nn.Linear(hidden_size, 1)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids, attention_mask)
        # Use last token's hidden state
        reward = self.head(outputs[:, -1, :])
        return reward

def preference_loss(reward_model, chosen, rejected):
    """Bradley-Terry loss for preference learning"""
    r_chosen = reward_model(chosen['input_ids'], chosen['attention_mask'])
    r_rejected = reward_model(rejected['input_ids'], rejected['attention_mask'])
    
    # Log-sigmoid of reward difference
    loss = -F.logsigmoid(r_chosen - r_rejected).mean()
    return loss
```

---

## 📊 Alternatives to Full RL

| Method | How | Advantage |
|--------|-----|-----------|
| RLHF (PPO) | Full RL | Flexible |
| DPO | Direct optimization | No reward model |
| RRHF | Ranking loss | Simpler |
| IPO | Identity mapping | More stable |

---

---

⬅️ [Back: Dpo](./dpo.md)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>
