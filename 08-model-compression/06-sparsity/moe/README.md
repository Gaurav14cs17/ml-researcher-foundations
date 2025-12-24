# 🎯 Mixture of Experts (MoE)

> **Conditional computation for efficient scaling**

<img src="./images/moe-visual.svg" width="100%">

---

## 📂 Topics

| File | Topic | Key Concepts |
|------|-------|--------------|

---

## 🎯 The Architecture

```
+-------------------------------------------------------------+
|                   MoE Layer                                 |
+-------------------------------------------------------------+
|                                                             |
|  Input x                                                    |
|      |                                                      |
|      ▼      |
|  +------------+                                            |
|  |   Router   |  g(x) = softmax(W_g · x)                   |
|  |   g(x)     |  Select top-k experts                      |
|  +------------+                                            |
|      |                                                      |
|      +--> Expert 1: FFN₁(x)  [weight: g₁]                 |
|      +--> Expert 2: FFN₂(x)  [weight: g₂]                 |
|      +--> Expert 3: (skipped, g₃ ≈ 0)                     |
|      +--> Expert 4: (skipped, g₄ ≈ 0)                     |
|      +--> Expert 5: (skipped)                             |
|      +--> Expert 6: (skipped)                             |
|      +--> Expert 7: (skipped)                             |
|      +--> Expert 8: (skipped)                             |
|                                                             |
|  Output = g₁·FFN₁(x) + g₂·FFN₂(x)                         |
|                                                             |
+-------------------------------------------------------------+

Only 2/8 experts activated! (top-2 routing)
```

---

## 📊 MoE vs Dense Comparison

| Model | Total Params | Active Params | Quality |
|-------|-------------|---------------|---------|
| LLaMA-70B (dense) | 70B | 70B | Baseline |
| Mixtral 8x7B (MoE) | 46B | 12B | ≈ 70B quality |
| GPT-4 (rumored MoE) | ~1.8T | ~220B | SOTA |

---

## 🔥 Load Balancing

```
Problem: All tokens go to same expert!
+-------------------------------------+
| Expert 1: 90% tokens (overloaded!)  |
| Expert 2: 5%                        |
| Expert 3: 3%                        |
| Expert 4: 2%                        |
+-------------------------------------+

Solution: Auxiliary loss for balance
L_balance = α · Σᵢ fᵢ · Pᵢ

Where:
• fᵢ = fraction of tokens to expert i
• Pᵢ = average routing probability to expert i
• Minimize to encourage uniform distribution
```

---

## 💻 Code Sketch

```python
class MoELayer(nn.Module):
    def __init__(self, d_model, n_experts, top_k=2):
        super().__init__()
        self.router = nn.Linear(d_model, n_experts)
        self.experts = nn.ModuleList([
            FFN(d_model) for _ in range(n_experts)
        ])
        self.top_k = top_k
    
    def forward(self, x):
        # Routing
        router_logits = self.router(x)
        weights, indices = torch.topk(
            F.softmax(router_logits, dim=-1), 
            self.top_k
        )
        
        # Only compute top-k experts
        output = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            mask = (indices == i).any(dim=-1)
            if mask.any():
                output[mask] += weights[..., i:i+1] * expert(x[mask])
        
        return output
```

---

## 📐 Mathematical Foundations

<img src="./images/moe-math.svg" width="100%">

---

## 🔗 Where This Topic Is Used

| Topic | How MoE Is Used |
|-------|----------------|
| **Mixtral** | 8 experts, top-2 routing |
| **GPT-4** | Rumored MoE architecture |
| **Switch Transformer** | Simplified single-expert routing |
| **GShard** | Distributed MoE |
| **DeepSeek** | MoE for efficient LLMs |

### Prerequisite For

```
MoE --> Understanding Mixtral
    --> Efficient model scaling
    --> Conditional computation
    --> LLM architecture research
```

---

## 📚 References & Resources

### 📄 Key Papers

| Paper | Authors | Year | Key Contribution |
|-------|---------|------|------------------|
| [Outrageously Large NNs](https://arxiv.org/abs/1701.06538) | Shazeer et al. | 2017 | Original MoE for NLP |
| [Switch Transformer](https://arxiv.org/abs/2101.03961) | Fedus et al. | 2022 | Simplified top-1 routing |
| [GShard](https://arxiv.org/abs/2006.16668) | Lepikhin et al. | 2020 | Distributed MoE |
| [Mixtral](https://arxiv.org/abs/2401.04088) | Mistral AI | 2024 | Open MoE LLM |
| [DeepSeek-MoE](https://arxiv.org/abs/2401.06066) | DeepSeek | 2024 | Fine-grained experts |
| [Expert Choice Routing](https://arxiv.org/abs/2202.09368) | Zhou et al. | 2022 | Experts choose tokens |
| [ST-MoE](https://arxiv.org/abs/2202.08906) | Zoph et al. | 2022 | Stable training for MoE |
| 🇨🇳 MoE架构详解 | [知乎](https://zhuanlan.zhihu.com/p/674278454) | - | 从Switch到Mixtral |
| 🇨🇳 Mixtral原理分析 | [CSDN](https://blog.csdn.net/v_JULY_v/article/details/135386164) | - | 8x7B模型解读 |
| 🇨🇳 MoE综述 | [机器之心](https://www.jiqizhixin.com/articles/2024-01-16-2) | - | 条件计算方法总结 |
| 🇨🇳 MoE架构讲解 | [B站](https://www.bilibili.com/video/BV1CK411y7i3) | - | 视频教程 |

### 🏆 Notable MoE Models

| Model | Experts | Active | Quality |
|-------|---------|--------|---------|
| Mixtral 8x7B | 8 | 2 | ≈LLaMA-70B |
| DeepSeek-MoE | 64 | 4 | Efficient |
| GPT-4 (rumored) | 8 | 2 | SOTA |
| Switch-XXL | 2048 | 1 | Research |

---

⬅️ [Back: Sparsity](../) | ➡️ [Next: Sparse Networks](../sparse-networks/)

