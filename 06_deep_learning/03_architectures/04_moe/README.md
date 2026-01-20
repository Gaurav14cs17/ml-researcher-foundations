<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=120&section=header&text=Mixture%20of%20Experts%20MoE&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-06-45B7D1?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/moe-architecture.svg" width="100%">

*Caption: MoE layers replace dense FFN layers with multiple "expert" networks. A gating network learns to route each token to the top-K experts. This allows 8x+ parameters with similar FLOPs. Used in GPT-4, Mixtral, and Switch Transformer.*

---

## 📂 Overview

Mixture of Experts is a technique to scale model parameters efficiently by activating only a subset of the network for each input. The key insight is that different inputs may require different "expertise."

---

## 📐 Mathematical Foundations

### 1. Basic MoE Formulation

Given input $x \in \mathbb{R}^d$ and $N$ experts $\{E\_1, E\_2, ..., E\_N\}$:

```math
y = \sum_{i=1}^{N} G(x)_i \cdot E_i(x)

```

Where:

- $G(x) \in \mathbb{R}^N$ is the gating function output (routing weights)

- $E\_i(x)$ is the output of expert $i$

- Each expert is typically an FFN: $E\_i(x) = W\_2^{(i)} \cdot \text{ReLU}(W\_1^{(i)} x)$

### 2. Gating Function (Router)

**Softmax Gating:**

```math
G(x) = \text{softmax}(W_g \cdot x)

```

Where $W\_g \in \mathbb{R}^{N \times d}$ is the learnable gating weight matrix.

**Top-K Sparse Gating:**

```math
G(x)_i = \begin{cases}
\frac{\exp(h_i)}{\sum_{j \in \text{TopK}} \exp(h_j)} & \text{if } i \in \text{TopK}(h) \\
0 & \text{otherwise}
\end{cases}

```

Where $h = W\_g \cdot x$ and TopK selects indices with highest values.

### 3. Computational Complexity Analysis

**Dense Model:**

```math
\text{FLOPs}_{\text{dense}} = O(d \cdot d_{ff})

```

**Sparse MoE (Top-K routing):**

```math
\text{FLOPs}_{\text{MoE}} = O(d \cdot N) + K \cdot O(d \cdot d_{ff}) = O(K \cdot d \cdot d_{ff})

```

**Parameters:**

```math
\text{Params}_{\text{MoE}} = N \cdot \text{Params}_{\text{expert}} + \text{Params}_{\text{router}}

```

**Insight:** With $K=2$ and $N=8$:

- 8x parameters

- Same FLOPs as dense (2 experts active)

- Effective capacity increase without compute increase

---

## 🔬 Routing Mechanisms

### Top-K Routing Algorithm

```
Input: Token embedding x ∈ ℝ^d
Router weights: W_g ∈ ℝ^(N×d)

Algorithm:
1. Compute logits: h = W_g · x           # ℝ^N
2. Select top-K: indices = TopK(h, K)    # K indices
3. Compute weights: w = softmax(h[indices])  # ℝ^K
4. Compute output: y = Σ_{i∈indices} w_i · E_i(x)

Output: y ∈ ℝ^d

```

### Routing Variants

| Type | Method | Formula | Used In |
|------|--------|---------|---------|
| **Top-1** | Single best expert | $y = E\_{\arg\max(h)}(x)$ | Switch Transformer |
| **Top-2** | Two best experts | $y = \sum\_{i \in \text{Top2}} G\_i \cdot E\_i(x)$ | GShard, Mixtral |
| **Expert Choice** | Experts choose tokens | Capacity-based selection | Expert Choice (2022) |
| **Soft MoE** | Weighted average of all | $y = \sum\_i G\_i \cdot E\_i(x)$ | Soft MoE (2023) |

### Noisy Top-K Gating

To encourage exploration and load balancing:

```math
h_i = (W_g \cdot x)_i + \epsilon_i \cdot \text{softplus}((W_{\text{noise}} \cdot x)_i)

```

Where $\epsilon\_i \sim \mathcal{N}(0, 1)$ is random noise during training.

---

## ⚠️ Load Balancing

### The Problem

Without balancing, all tokens route to same experts → some experts undertrained.

### Auxiliary Loss Function

**Importance Loss (ensuring all experts are used):**

```math
L_{\text{importance}} = \text{CV}\left(\sum_{x \in B} G(x)\right)^2

```

Where CV is coefficient of variation: $\text{CV} = \frac{\sigma}{\mu}$

**Load Loss (ensuring equal tokens per expert):**

```math
L_{\text{load}} = N \cdot \sum_{i=1}^{N} f_i \cdot P_i

```

Where:

- $f\_i = \frac{1}{|B|} \sum\_{x \in B} \mathbf{1}[i \in \text{TopK}(G(x))]$ (fraction of tokens to expert $i$)

- $P\_i = \frac{1}{|B|} \sum\_{x \in B} G(x)\_i$ (average routing probability)

**Combined Loss:**

```math
L_{\text{total}} = L_{\text{task}} + \alpha \cdot L_{\text{aux}}

```

Where $\alpha \approx 0.01$ typically.

### Expert Capacity

To prevent expert overload:

```math
\text{Capacity} = \left\lceil \frac{\text{tokens per batch} \times K}{N} \times c \right\rceil

```

Where $c \in [1.0, 2.0]$ is the capacity factor.

Tokens exceeding capacity are either:
1. Dropped (Switch Transformer)
2. Sent to overflow expert
3. Processed by next-best expert

---

## 📊 Theoretical Analysis

### Gradient Flow in MoE

For a token routed to expert $i$:

```math
\frac{\partial L}{\partial W_i} = G(x)_i \cdot \frac{\partial L}{\partial E_i(x)} \cdot \frac{\partial E_i(x)}{\partial W_i}

```

**Issue:** Experts with low $G(x)\_i$ receive small gradients.

**Solution:** Auxiliary losses ensure all experts receive sufficient gradients.

### Effective Model Capacity

For Top-K routing with K=2, N=8 experts:

```math
\text{Effective Combinations} = \binom{N}{K} = \binom{8}{2} = 28 \text{ distinct expert pairs}

```

This allows the model to specialize for 28 different "modes" of input.

---

## 💻 Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    """Single FFN expert"""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()
    
    def forward(self, x):
        return self.w2(self.activation(self.w1(x)))

class TopKRouter(nn.Module):
    """Top-K routing with load balancing"""
    def __init__(self, d_model, num_experts, top_k=2):
        super().__init__()
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.top_k = top_k
        self.num_experts = num_experts
    
    def forward(self, x):
        # x: (batch, seq, d_model)
        logits = self.gate(x)  # (batch, seq, num_experts)
        
        # Top-K selection
        topk_logits, topk_indices = torch.topk(logits, self.top_k, dim=-1)
        topk_weights = F.softmax(topk_logits, dim=-1)
        
        # Compute load balancing loss
        router_probs = F.softmax(logits, dim=-1)
        self.aux_loss = self.load_balance_loss(router_probs, topk_indices)
        
        return topk_indices, topk_weights
    
    def load_balance_loss(self, router_probs, selected_experts):
        """
        Auxiliary loss for load balancing
        
        f_i: fraction of tokens routed to expert i
        P_i: average routing probability to expert i
        Loss = N * Σ_i (f_i * P_i)
        """
        # f_i: fraction of tokens choosing expert i
        one_hot = F.one_hot(selected_experts, self.num_experts).float()
        tokens_per_expert = one_hot.sum(dim=[0, 1, 2])  # (num_experts,)
        f = tokens_per_expert / tokens_per_expert.sum()
        
        # P_i: mean probability assigned to expert i
        P = router_probs.mean(dim=[0, 1])
        
        # Load balance loss
        return self.num_experts * (f * P).sum()

class MoELayer(nn.Module):
    """Mixture of Experts Layer"""
    def __init__(self, d_model, d_ff, num_experts=8, top_k=2):
        super().__init__()
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff) for _ in range(num_experts)
        ])
        self.router = TopKRouter(d_model, num_experts, top_k)
        self.top_k = top_k
    
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        # Route tokens
        topk_indices, topk_weights = self.router(x)  # (B, S, K), (B, S, K)
        
        # Compute expert outputs
        output = torch.zeros_like(x)
        
        # Efficient implementation: group tokens by expert
        flat_x = x.view(-1, d_model)  # (B*S, d_model)
        flat_indices = topk_indices.view(-1, self.top_k)  # (B*S, K)
        flat_weights = topk_weights.view(-1, self.top_k)  # (B*S, K)
        
        for k in range(self.top_k):
            expert_indices = flat_indices[:, k]  # (B*S,)
            weights_k = flat_weights[:, k:k+1]   # (B*S, 1)
            
            for i, expert in enumerate(self.experts):
                mask = (expert_indices == i)
                if mask.any():
                    expert_input = flat_x[mask]
                    expert_output = expert(expert_input)
                    output.view(-1, d_model)[mask] += weights_k[mask] * expert_output
        
        return output
    
    def get_aux_loss(self):
        return self.router.aux_loss

# Usage in Transformer
class MoETransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, num_experts=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.moe = MoELayer(d_model, d_ff, num_experts)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # Self-attention
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        # MoE FFN
        x = x + self.moe(self.ln2(x))
        return x

```

---

## 🌍 Real-World Applications

| Model | MoE Configuration | Key Features |
|-------|------------------|--------------|
| **Mixtral 8x7B** | 8 experts, Top-2 | 46.7B total, 12.9B active |
| **Switch Transformer** | Up to 2048 experts, Top-1 | 1.6T params, same compute |
| **GPT-4** (rumored) | Sparse MoE | ~1.8T params estimated |
| **Grok** | 8 experts | 314B total params |
| **DeepSeek-MoE** | 64 experts, shared + routed | Fine-grained experts |

### Why MoE for LLMs?

1. **Scaling Law:** More parameters → better performance
2. **Compute Efficiency:** Only K experts active per token
3. **Specialization:** Different experts for different domains
4. **Inference Speed:** Same as smaller dense model

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | Switch Transformer | [arXiv](https://arxiv.org/abs/2101.03961) |
| 📄 | Mixtral Paper | [arXiv](https://arxiv.org/abs/2401.04088) |
| 📄 | GShard Paper | [arXiv](https://arxiv.org/abs/2006.16668) |
| 📄 | Expert Choice Routing | [arXiv](https://arxiv.org/abs/2202.09368) |
| 📄 | Soft MoE | [arXiv](https://arxiv.org/abs/2308.00951) |
| 🇨🇳 | MoE架构详解 | [知乎](https://zhuanlan.zhihu.com/p/674278454) |
| 🇨🇳 | Mixtral原理分析 | [CSDN](https://blog.csdn.net/v_JULY_v/article/details/135386164) |
| 🇨🇳 | MoE视频讲解 | [B站](https://www.bilibili.com/video/BV1CK411y7i3) |

---

⬅️ [Back: MLP](../03_mlp/README.md) | ➡️ [Next: RNN](../05_rnn/README.md)

---

⬅️ [Back: Architectures](../../README.md)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=80&section=footer" width="100%"/>
</p>
