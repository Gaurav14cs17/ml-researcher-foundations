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

## 📂 Overview

Mixture of Experts (MoE) is a neural network architecture that uses conditional computation - only a subset of parameters are active for each input. This enables scaling to trillion-parameter models while keeping inference cost manageable.

---

## 📐 Mathematical Formulation

### Basic MoE Layer

```math
y = \sum_{i=1}^{N} G(x)_i \cdot E_i(x)

```

where:

- \(N\): number of experts

- \(E_i(x)\): output of expert \(i\)

- \(G(x)_i\): gating weight for expert \(i\)

### Gating Function

**Softmax routing:**

```math
G(x) = \text{softmax}(W_g x)

```

**Top-k sparse routing:**

```math
G(x) = \text{softmax}(\text{TopK}(W_g x, k))

```

Only the top-k experts are activated, rest have zero weight.

---

## 🔬 Sparse MoE

### Top-k Routing

For each token, select top-k experts:

```math
G(x) = \text{Normalize}(\text{TopK}(\text{softmax}(W_g x), k))

```

**Computational savings:**

```
Dense FFN: O(d × d_ff)
MoE (k experts of N total): O(k × d × d_ff / N)

Example: 8 experts, top-2 routing
Compute: 2/8 = 25% of dense FFN
Parameters: 8× of dense FFN

```

### Load Balancing

**Problem:** Some experts may be underutilized (routing collapse).

**Auxiliary loss for load balancing:**

```math
L_{aux} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i

```

where:

- \(f_i\): fraction of tokens routed to expert \(i\)

- \(P_i\): average routing probability for expert \(i\)

- \(\alpha\): balancing coefficient (typically 0.01)

**Derivation:**

```
We want uniform load: f_i = 1/N for all i

The loss penalizes when high-probability experts (large P_i)
also receive many tokens (large f_i).

At optimum: P_i = f_i = 1/N → L_aux = α·N·(1/N)·(1/N)·N = α

```

---

## 📊 Expert Capacity

### Capacity Factor

Each expert has limited capacity:

```math
\text{capacity} = \frac{k \cdot \text{tokens}}{N} \cdot \text{capacity\_factor}

```

Tokens exceeding capacity are dropped or routed to alternate experts.

**Capacity factor analysis:**

```
capacity_factor = 1.0: No slack, some drops
capacity_factor = 1.25: 25% extra capacity, fewer drops
capacity_factor = 2.0: Double capacity, almost no drops

Trade-off: Higher capacity → more compute but less dropping

```

### Token Dropping

```python
# Compute assignments
gates = softmax(router(x))
top_k_indices = gates.topk(k).indices

# Count tokens per expert
expert_counts = bincount(top_k_indices)

# Drop tokens exceeding capacity
for expert_id in range(num_experts):
    if expert_counts[expert_id] > capacity:
        # Keep only first 'capacity' tokens
        drop_mask[expert_id] = True

```

---

## 📐 Advanced Routing Strategies

### 1. Switch Transformer (Top-1)

Use only 1 expert per token:

```math
G(x) = \text{onehot}(\arg\max_i (W_g x)_i)

```

**Benefits:**
- Simpler implementation

- Better hardware efficiency

- Works well in practice

### 2. Expert Choice Routing

Experts choose tokens instead of tokens choosing experts:

```math
P = \text{softmax}((W_g X)^T)

```

Each expert selects top-k tokens from the batch.

**Benefits:**
- Guarantees perfect load balance

- Avoids token dropping

- Better gradient flow

### 3. Soft MoE

Differentiable routing without hard assignment:

```math
y = \sum_i \text{softmax}(W_g x)_i \cdot E_i(x)

```

All experts contribute, weighted by routing scores.

---

## 💻 Complete Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class Expert(nn.Module):
    """
    Single expert network (typically an FFN)
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = F.silu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

class TopKRouter(nn.Module):
    """
    Top-K routing with load balancing
    """
    def __init__(self, d_model, num_experts, top_k=2, capacity_factor=1.25):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        
        self.gate = nn.Linear(d_model, num_experts, bias=False)
    
    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, d_model)
        
        Returns:
            gates: (batch, seq_len, top_k) routing weights
            indices: (batch, seq_len, top_k) expert indices
            aux_loss: load balancing loss
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute routing logits
        logits = self.gate(x)  # (batch, seq, num_experts)
        
        # Top-k selection
        gates, indices = logits.topk(self.top_k, dim=-1)
        gates = F.softmax(gates, dim=-1)
        
        # Compute auxiliary loss for load balancing
        aux_loss = self._compute_aux_loss(logits)
        
        return gates, indices, aux_loss
    
    def _compute_aux_loss(self, logits):
        """
        Load balancing auxiliary loss
        L_aux = α · N · Σᵢ fᵢ · Pᵢ
        """
        # Router probability (soft assignment)
        router_probs = F.softmax(logits, dim=-1)  # (batch, seq, experts)
        
        # Average probability per expert
        expert_probs = router_probs.mean(dim=[0, 1])  # (experts,)
        
        # Fraction of tokens routed (hard assignment for counting)
        expert_indices = logits.argmax(dim=-1)  # (batch, seq)
        expert_counts = torch.zeros(self.num_experts, device=logits.device)
        for i in range(self.num_experts):
            expert_counts[i] = (expert_indices == i).float().sum()
        expert_fracs = expert_counts / expert_counts.sum()
        
        # Auxiliary loss
        aux_loss = self.num_experts * (expert_fracs * expert_probs).sum()
        
        return aux_loss

class MoELayer(nn.Module):
    """
    Mixture of Experts Layer
    """
    def __init__(self, d_model, d_ff, num_experts, top_k=2, 
                 capacity_factor=1.25, dropout=0.1):
        super().__init__()
        
        self.num_experts = num_experts
        self.top_k = top_k
        self.d_model = d_model
        
        # Router
        self.router = TopKRouter(d_model, num_experts, top_k, capacity_factor)
        
        # Experts
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff, dropout) for _ in range(num_experts)
        ])
    
    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, d_model)
        
        Returns:
            output: (batch, seq_len, d_model)
            aux_loss: load balancing loss
        """
        batch_size, seq_len, d_model = x.shape
        
        # Get routing weights and indices
        gates, indices, aux_loss = self.router(x)
        # gates: (batch, seq, top_k)
        # indices: (batch, seq, top_k)
        
        # Initialize output
        output = torch.zeros_like(x)
        
        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert
            # (for each top-k slot)
            for k in range(self.top_k):
                mask = indices[:, :, k] == expert_idx  # (batch, seq)
                
                if mask.any():
                    # Get tokens for this expert
                    expert_input = x[mask]  # (num_tokens, d_model)
                    
                    # Apply expert
                    expert_output = self.experts[expert_idx](expert_input)
                    
                    # Scale by gate and add to output
                    gate_values = gates[:, :, k][mask]  # (num_tokens,)
                    output[mask] += gate_values.unsqueeze(-1) * expert_output
        
        return output, aux_loss

class MoELayerEfficient(nn.Module):
    """
    Efficient MoE implementation using batched operations
    """
    def __init__(self, d_model, d_ff, num_experts, top_k=2, dropout=0.1):
        super().__init__()
        
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Router
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        
        # All experts as single tensors for efficiency
        self.w1 = nn.Parameter(torch.randn(num_experts, d_model, d_ff) * 0.02)
        self.w2 = nn.Parameter(torch.randn(num_experts, d_ff, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, d_model = x.shape
        
        # Flatten batch and sequence
        x_flat = x.view(-1, d_model)  # (batch * seq, d_model)
        
        # Routing
        logits = self.gate(x_flat)  # (batch * seq, num_experts)
        gates, indices = logits.topk(self.top_k, dim=-1)
        gates = F.softmax(gates, dim=-1)  # (batch * seq, top_k)
        
        # Aux loss
        router_probs = F.softmax(logits, dim=-1).mean(dim=0)
        aux_loss = self.num_experts * (router_probs ** 2).sum()
        
        # Expert computation using einsum for efficiency
        # This is a simplified version - real implementations use more tricks
        output = torch.zeros_like(x_flat)
        
        for k in range(self.top_k):
            expert_idx = indices[:, k]  # (batch * seq,)
            gate_val = gates[:, k]  # (batch * seq,)
            
            for e in range(self.num_experts):
                mask = expert_idx == e
                if mask.any():
                    x_e = x_flat[mask]  # (num_tokens, d_model)
                    
                    # FFN forward
                    h = F.silu(x_e @ self.w1[e])  # (num_tokens, d_ff)
                    h = self.dropout(h)
                    y = h @ self.w2[e]  # (num_tokens, d_model)
                    
                    output[mask] += gate_val[mask].unsqueeze(-1) * y
        
        return output.view(batch_size, seq_len, d_model), aux_loss

class MoETransformerBlock(nn.Module):
    """
    Transformer block with MoE FFN
    """
    def __init__(self, d_model, n_heads, d_ff, num_experts, top_k=2, dropout=0.1):
        super().__init__()
        
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.moe = MoELayer(d_model, d_ff, num_experts, top_k, dropout=dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=mask)
        x = x + self.dropout(attn_out)
        
        # MoE FFN
        h = self.ln2(x)
        moe_out, aux_loss = self.moe(h)
        x = x + self.dropout(moe_out)
        
        return x, aux_loss

class SparseMoE(nn.Module):
    """
    Complete Sparse MoE Language Model
    """
    def __init__(self, vocab_size, d_model=512, n_heads=8, n_layers=6,
                 d_ff=2048, num_experts=8, top_k=2, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(2048, d_model)
        
        # Interleaved dense and MoE layers
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            if i % 2 == 0:  # Every other layer is MoE
                self.layers.append(
                    MoETransformerBlock(d_model, n_heads, d_ff, num_experts, top_k, dropout)
                )
            else:
                # Dense FFN layer
                self.layers.append(
                    nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, 
                                               activation='gelu', batch_first=True)
                )
        
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.token_emb.weight
    
    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        x = self.token_emb(input_ids)
        positions = torch.arange(seq_len, device=input_ids.device)
        x = x + self.pos_emb(positions)
        
        # Transformer layers
        total_aux_loss = 0
        for layer in self.layers:
            if isinstance(layer, MoETransformerBlock):
                x, aux_loss = layer(x)
                total_aux_loss += aux_loss
            else:
                x = layer(x)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits, total_aux_loss

# Example usage
vocab_size = 32000
model = SparseMoE(
    vocab_size=vocab_size,
    d_model=256,
    n_heads=8,
    n_layers=4,
    d_ff=1024,
    num_experts=8,
    top_k=2
)

# Forward pass
input_ids = torch.randint(0, vocab_size, (2, 128))
logits, aux_loss = model(input_ids)

print(f"Input: {input_ids.shape}")
print(f"Logits: {logits.shape}")
print(f"Aux loss: {aux_loss.item():.4f}")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params / 1e6:.2f}M")

```

---

## 📊 Notable MoE Models

| Model | Experts | Total Params | Active Params | Top-K |
|-------|---------|--------------|---------------|-------|
| **Switch Transformer** | 128-2048 | 1.6T | 12B | 1 |
| **Mixtral 8x7B** | 8 | 47B | 12B | 2 |
| **GPT-4 (rumored)** | ~16 | ~1.8T | ~220B | 2 |
| **Gemini 1.5** | ? | ~1T+ | ? | ? |

---

## 📚 Key Insights

| Insight | Details |
|---------|---------|
| **Capacity matters** | Too low → dropped tokens, too high → wasted compute |
| **Load balancing** | Critical for training stability |
| **Expert choice** | Can be better than token choice |
| **Interleaving** | Not every layer needs to be MoE |

---

## 📚 References

| Type | Resource | Link |
|------|----------|------|
| 📄 | Switch Transformer | [Fedus et al., 2021](https://arxiv.org/abs/2101.03961) |
| 📄 | Mixtral | [Jiang et al., 2024](https://arxiv.org/abs/2401.04088) |
| 📄 | Expert Choice | [Zhou et al., 2022](https://arxiv.org/abs/2202.09368) |
| 📄 | ST-MoE | [Zoph et al., 2022](https://arxiv.org/abs/2202.08906) |

---

## 🗺️ Navigation

| ⬅️ Previous | 🏠 Home | ➡️ Next |
|:-----------:|:-------:|:-------:|
| [Language Models](../06_language_models/README.md) | [Architectures](../README.md) | [ResNet](../08_resnet/README.md) |

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=80&section=footer" width="100%"/>
</p>
