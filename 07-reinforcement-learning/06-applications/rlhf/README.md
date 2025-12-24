# 🔥 RLHF: Reinforcement Learning from Human Feedback

> **Aligning language models with human preferences**

---

## 🎯 Visual Overview

<img src="./images/rlhf-pipeline.svg" width="100%">

*Caption: The RLHF pipeline showing three stages: (1) Supervised Fine-Tuning (SFT) on demonstrations, (2) Reward Model training from human preferences, and (3) RL optimization using PPO to maximize reward while staying close to the SFT policy. This is how ChatGPT and Claude are aligned.*

---

## 📂 Topics in This Folder

| File | Topic | Application |
|------|-------|-------------|
| [reward-model.md](./reward-model.md) | Learning preferences | Human feedback |
| [dpo.md](./dpo.md) | Direct Preference Optimization | Simpler RLHF |

---

## 🎯 The RLHF Pipeline

```
+-------------------------------------------------------------+
|                      RLHF Pipeline                          |
+-------------------------------------------------------------+
|                                                             |
|  Step 1: Supervised Fine-Tuning (SFT)                      |
|  ----------------------------------                         |
|  Pretrained LLM → Fine-tune on demonstrations → SFT Model  |
|                                                             |
|  Step 2: Reward Model Training                              |
|  -----------------------------                              |
|  Human ranks outputs → Train reward model r(x, y)           |
|                                                             |
|  Comparison:  "Response A is better than Response B"        |
|  Loss: -log σ(r(x, y_w) - r(x, y_l))  (Bradley-Terry)      |
|                                                             |
|  Step 3: RL Fine-Tuning (PPO)                              |
|  -----------------------------                              |
|  Optimize: max E[r(x, y)] - β·KL(π || π_SFT)               |
|            ------------    ------------------               |
|            reward signal   stay close to SFT                |
|                                                             |
+-------------------------------------------------------------+
```

---

## 🌍 Where RLHF is Used

| Model | Year | Notes |
|-------|------|-------|
| **ChatGPT** | 2022 | Original RLHF for chat |
| **GPT-4** | 2023 | Advanced RLHF |
| **Claude** | 2023 | Constitutional AI variant |
| **LLaMA-2** | 2023 | Open RLHF |
| **Gemini** | 2023 | Google's approach |

---

## 📐 Key Formulas

### Reward Model Training

```
Given: Preference pairs (x, y_w, y_l) where y_w is preferred

Loss = -log σ(r(x, y_w) - r(x, y_l))

Train r to give higher scores to preferred completions
```

### PPO with KL Penalty

```
Objective:
max_π E_{x~D, y~π}[r(x, y) - β·KL(π(y|x) || π_ref(y|x))]

Where:
• r(x, y) = learned reward model
• π_ref = SFT policy (frozen)
• β = KL penalty coefficient (typically 0.01-0.1)

PPO surrogate:
L = min(r_t · A_t, clip(r_t, 1-ε, 1+ε) · A_t)
```

### DPO (Simpler Alternative!)

```
DPO eliminates the explicit reward model!

Directly optimize:
L = -log σ(β log(π(y_w|x)/π_ref(y_w|x)) - β log(π(y_l|x)/π_ref(y_l|x)))

Equivalent to reward:
r(x, y) = β log(π(y|x)/π_ref(y|x)) + β log Z(x)

No RL loop needed! Just supervised fine-tuning.
```

---

## 💻 Code Example

```python
import torch
import torch.nn.functional as F

def compute_reward_loss(reward_model, chosen, rejected):
    """Bradley-Terry loss for reward model"""
    r_chosen = reward_model(chosen)
    r_rejected = reward_model(rejected)
    
    # Preference loss: P(chosen > rejected) = σ(r_chosen - r_rejected)
    loss = -F.logsigmoid(r_chosen - r_rejected).mean()
    return loss

def dpo_loss(policy, ref_policy, chosen, rejected, beta=0.1):
    """Direct Preference Optimization"""
    # Log probabilities
    log_pi_chosen = policy.log_prob(chosen)
    log_pi_rejected = policy.log_prob(rejected)
    log_ref_chosen = ref_policy.log_prob(chosen).detach()
    log_ref_rejected = ref_policy.log_prob(rejected).detach()
    
    # Log ratios
    log_ratio_chosen = log_pi_chosen - log_ref_chosen
    log_ratio_rejected = log_pi_rejected - log_ref_rejected
    
    # DPO loss
    loss = -F.logsigmoid(beta * (log_ratio_chosen - log_ratio_rejected)).mean()
    return loss
```

---

## 📚 Resources

| Type | Title | Link |
|------|-------|------|
| 📄 | InstructGPT | [arXiv](https://arxiv.org/abs/2203.02155) |
| 📄 | DPO | [arXiv](https://arxiv.org/abs/2305.18290) |
| 🎥 | RLHF Explained | HuggingFace blog |
| 🇨🇳 | RLHF详解 | 知乎专栏 |

---

## 🔗 Where This Topic Is Used

| Topic | How RLHF/DPO Is Used |
|-------|---------------------|
| **ChatGPT** | RLHF with PPO for alignment |
| **Claude** | Constitutional AI (RLHF variant) |
| **GPT-4** | Advanced RLHF pipeline |
| **LLaMA-2 Chat** | RLHF + DPO |
| **Gemini** | RLHF for helpfulness |
| **Code Assistants** | RLHF for useful suggestions |
| **Instruction Following** | RLHF teaches following instructions |
| **Safety Alignment** | RLHF reduces harmful outputs |

### RLHF Components Used In

| Component | Used By |
|-----------|---------|
| **Reward Model** | ChatGPT, Claude, GPT-4 |
| **PPO** | Most RLHF implementations |
| **DPO** | Simpler alternative (LLaMA-2) |
| **KL Penalty** | Prevent policy collapse |
| **Human Feedback** | All aligned models |

### Prerequisite For

```
RLHF/DPO --> Building helpful AI assistants
        --> Understanding ChatGPT training
        --> Fine-tuning LLMs for chat
        --> AI safety research
        --> Constitutional AI
```

### Training Pipeline Position

```
Pretraining → SFT → RLHF/DPO
(language)   (task) (preference)
                        ↑
                   You are here!
```

---

⬅️ [Back: Applications](../) | ➡️ [Start Learning!](../../../)

---

⬅️ [Back: Games](../games/) | ➡️ [Next: Robotics](../robotics/)
