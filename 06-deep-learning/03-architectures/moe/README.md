<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=120&section=header&text=Mixture%20of%20Experts%20MoE&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-06-45B7D1?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/moe-architecture.svg" width="100%">

*Caption: MoE layers replace dense FFN layers with multiple "expert" networks. A gating network learns to route each token to the top-K experts. This allows 8x+ parameters with similar FLOPs. Used in GPT-4, Mixtral, and Switch Transformer.*

---

## 📂 Overview

Mixture of Experts is a technique to scale model parameters efficiently by activating only a subset of the network for each input.

---

## 🔑 Key Concepts

| Concept | Description |
|---------|-------------|
| **Expert** | Independent FFN network (same architecture) |
| **Gating Network** | Routes tokens to experts via softmax |
| **Top-K Routing** | Activate K experts per token (sparse) |
| **Load Balancing** | Ensure experts are utilized equally |

---

## 📐 How It Works

```
Input x → Gating Network G(x) → Softmax weights [w₁, w₂, ..., wₙ]
                              ↓
       Select top-K experts by weight
                              ↓
Output = Σᵢ wᵢ · Expertᵢ(x)   (weighted sum)
```

---

## 💻 Code

```python
class MoELayer(nn.Module):
    def __init__(self, d_model, num_experts, top_k=2):
        self.experts = nn.ModuleList([FFN(d_model) for _ in range(num_experts)])
        self.gate = nn.Linear(d_model, num_experts)
        self.top_k = top_k
    
    def forward(self, x):
        # Get routing weights
        gate_logits = self.gate(x)
        weights, indices = torch.topk(gate_logits, self.top_k)
        weights = F.softmax(weights, dim=-1)
        
        # Compute weighted expert outputs
        output = sum(w * self.experts[i](x) for w, i in zip(weights.T, indices.T))
        return output
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | Switch Transformer | [arXiv](https://arxiv.org/abs/2101.03961) |
| 📄 | Mixtral Paper | [arXiv](https://arxiv.org/abs/2401.04088) |
| 📄 | GShard Paper | [arXiv](https://arxiv.org/abs/2006.16668) |
| 🇨🇳 | MoE架构详解 | [知乎](https://zhuanlan.zhihu.com/p/674278454) |
| 🇨🇳 | Mixtral原理分析 | [CSDN](https://blog.csdn.net/v_JULY_v/article/details/135386164) |
| 🇨🇳 | MoE视频讲解 | [B站](https://www.bilibili.com/video/BV1CK411y7i3) |


## 🔗 Where This Topic Is Used

| Model | MoE Application |
|-------|----------------|
| **Mixtral** | Sparse MoE LLM |
| **Switch Transformer** | Efficient scaling |
| **GPT-4** | Rumored MoE |
| **Vision** | Sparse vision models |

---

⬅️ [Back: Architectures](../)

---

⬅️ [Back: Mlp](../mlp/) | ➡️ [Next: Rnn](../rnn/)

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=80&section=footer" width="100%"/>
</p>
