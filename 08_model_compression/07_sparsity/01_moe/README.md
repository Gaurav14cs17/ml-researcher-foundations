<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=E74C3C&height=120&section=header&text=Mixture%20of%20Experts%20(MoE)&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-08-E74C3C?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 📐 Mathematical Foundations

### 1. MoE Layer Formulation

**General MoE Output:**
$$y = \sum_{i=1}^{N} g_i(x) \cdot E_i(x)$$

Where:
- $N$ = number of experts
- $g_i(x)$ = gating weight for expert $i$ (routing)
- $E_i(x)$ = output of expert $i$

**Gating Function:**
$$g(x) = \text{softmax}(W_g \cdot x)$$
$$g_i(x) = \frac{\exp(W_g^{(i)} \cdot x)}{\sum_j \exp(W_g^{(j)} \cdot x)}$$

### 2. Top-K Sparse Routing

**Sparse Gating (Top-K):**
$$g(x) = \text{TopK}(\text{softmax}(W_g \cdot x + \epsilon))$$

Where:
- $\epsilon \sim \text{Gumbel}(0, 1)$ (optional exploration noise)
- Only top-$K$ values are non-zero

**Normalization after Top-K:**
$$\hat{g}_i(x) = \frac{g_i(x)}{\sum_{j \in \text{TopK}} g_j(x)}$$

**Sparse Output:**
$$y = \sum_{i \in \text{TopK}(g)} \hat{g}_i(x) \cdot E_i(x)$$

### 3. Computational Efficiency

**Dense Model FLOPs:**
$$\text{FLOPs}_{dense} = \text{FLOPs}_{attention} + \text{FLOPs}_{FFN}$$

**MoE Model FLOPs:**
$$\text{FLOPs}_{MoE} = \text{FLOPs}_{attention} + K \cdot \text{FLOPs}_{expert}$$

**Efficiency Gain:**
$$\text{Speedup} = \frac{N \cdot \text{FLOPs}_{expert}}{K \cdot \text{FLOPs}_{expert}} = \frac{N}{K}$$

For Mixtral (N=8, K=2): $4\times$ fewer FLOPs in FFN layers!

### 4. Load Balancing

**Problem:** Without balancing, router may collapse to using few experts.

**Auxiliary Loss (Switch Transformer):**
$$\mathcal{L}_{balance} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i$$

Where:
- $f_i = \frac{\text{tokens routed to expert } i}{\text{total tokens}}$ (actual fraction)
- $P_i = \frac{1}{T}\sum_{t=1}^{T} g_i(x_t)$ (average routing probability)

**Intuition:** Minimizing $\sum f_i P_i$ encourages uniform distribution.

**Proof of Effectiveness:**
$$\sum_i f_i P_i \geq \sum_i f_i^2 \geq \frac{1}{N}$$

Equality holds when $f_i = 1/N$ (perfectly balanced).

### 5. Expert Capacity

**Capacity Factor $C$:**
$$\text{Capacity} = C \cdot \frac{\text{batch\_size} \times \text{seq\_len}}{N}$$

Tokens beyond capacity are dropped or sent to secondary expert.

**Token Dropping:**
If expert $i$ receives more than capacity tokens:
- Drop excess tokens (use identity residual)
- Or route to second-choice expert

### 6. Expert Choice Routing (Reverse Routing)

**Standard:** Tokens choose experts
**Expert Choice:** Experts choose tokens

$$\text{TopK}_{tokens}(\text{softmax}(E_i^T X))$$

Each expert selects its top-$K$ tokens to process.

**Benefits:**
- Perfect load balance (each expert processes exactly $K$ tokens)
- No dropping
- Better gradient flow

### 7. Theoretical Analysis

**Capacity vs Quality Trade-off:**
$$\mathcal{L}_{total} = \mathcal{L}_{task} + \alpha \mathcal{L}_{balance}$$

**Theorem (Scaling with Experts):**
For fixed compute budget $C$:
$$\text{Quality}(N \text{ experts}) > \text{Quality}(\text{dense})$$

When $N > N^*$ where $N^*$ depends on task and routing efficiency.

---

## 🎯 The Architecture

```
+-------------------------------------------------------------+
|                   MoE Layer                                 |
+-------------------------------------------------------------+
|                                                             |
|  Input x                                                    |
|      |                                                      |
|      v                                                      |
|  +------------+                                             |
|  |   Router   |  g(x) = softmax(W_g · x)                    |
|  |   g(x)     |  Select top-k experts                       |
|  +------------+                                             |
|      |                                                      |
|      +--> Expert 1: FFN₁(x)  [weight: g₁]                  |
|      +--> Expert 2: FFN₂(x)  [weight: g₂]                  |
|      +--> Expert 3: (skipped, g₃ ≈ 0)                      |
|      +--> Expert 4: (skipped, g₄ ≈ 0)                      |
|      +--> Expert 5: (skipped)                              |
|      +--> Expert 6: (skipped)                              |
|      +--> Expert 7: (skipped)                              |
|      +--> Expert 8: (skipped)                              |
|                                                             |
|  Output = g₁·FFN₁(x) + g₂·FFN₂(x)                          |
|                                                             |
+-------------------------------------------------------------+

Only 2/8 experts activated! (top-2 routing)
```

---

## 💻 Code Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    """Single expert (FFN)"""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.act = nn.GELU()
    
    def forward(self, x):
        return self.w2(self.act(self.w1(x)))

class MoELayer(nn.Module):
    """
    Mixture of Experts Layer with Top-K routing
    """
    def __init__(self, d_model, d_ff, n_experts=8, top_k=2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        
        # Router
        self.router = nn.Linear(d_model, n_experts, bias=False)
        
        # Experts
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff) for _ in range(n_experts)
        ])
    
    def forward(self, x):
        """
        Args:
            x: [batch, seq, d_model]
        Returns:
            output: [batch, seq, d_model]
            aux_loss: Load balancing loss
        """
        B, S, D = x.shape
        x_flat = x.view(-1, D)  # [B*S, D]
        
        # Compute routing logits
        router_logits = self.router(x_flat)  # [B*S, n_experts]
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Top-K selection
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        
        # Normalize top-k probabilities
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Compute output
        output = torch.zeros_like(x_flat)
        
        for k in range(self.top_k):
            expert_idx = top_k_indices[:, k]  # [B*S]
            expert_prob = top_k_probs[:, k:k+1]  # [B*S, 1]
            
            for i, expert in enumerate(self.experts):
                mask = (expert_idx == i)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = expert(expert_input)
                    output[mask] += expert_prob[mask] * expert_output
        
        # Reshape output
        output = output.view(B, S, D)
        
        # Compute load balancing loss
        aux_loss = self.compute_load_balance_loss(router_probs, top_k_indices)
        
        return output, aux_loss
    
    def compute_load_balance_loss(self, router_probs, indices):
        """
        Auxiliary loss for load balancing
        L = α * N * Σᵢ fᵢ * Pᵢ
        """
        # Average routing probability per expert
        P = router_probs.mean(dim=0)  # [n_experts]
        
        # Fraction of tokens per expert
        one_hot = F.one_hot(indices, self.n_experts).float()  # [B*S, K, n_experts]
        f = one_hot.sum(dim=(0, 1)) / one_hot.sum()  # [n_experts]
        
        # Load balance loss
        aux_loss = self.n_experts * (f * P).sum()
        
        return aux_loss

# ========== Usage ==========
moe = MoELayer(d_model=512, d_ff=2048, n_experts=8, top_k=2)
x = torch.randn(2, 128, 512)
output, aux_loss = moe(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Aux loss: {aux_loss.item():.4f}")
```

---

## 📊 MoE vs Dense Comparison

| Model | Total Params | Active Params | Quality |
|-------|-------------|---------------|---------|
| LLaMA-70B (dense) | 70B | 70B | Baseline |
| Mixtral 8x7B (MoE) | 46B | 12B | ≈ 70B quality |
| GPT-4 (rumored MoE) | ~1.8T | ~220B | SOTA |
| DeepSeek-MoE | 145B | 22B | Strong |

---

## 📐 Mathematical Visualization

<img src="./images/moe-math.svg" width="100%">

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
| [Expert Choice Routing](https://arxiv.org/abs/2202.09368) | Zhou et al. | 2022 | Reverse routing |

---

⬅️ [Back: Sparsity](../README.md) | ➡️ [Next: Sparse Networks](../02_sparse_networks/README.md)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=E74C3C&height=80&section=footer" width="100%"/>
</p>
