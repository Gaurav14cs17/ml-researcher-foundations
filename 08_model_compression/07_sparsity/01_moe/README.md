<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=27AE60&height=100&section=header&text=Mixture%20of%20Experts%20(MoE)&fontSize=28&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-08.07.01-27AE60?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 📐 Mathematical Foundations

### 1. MoE Layer Formulation

**General MoE Output:**

```math
y = \sum_{i=1}^{N} g_i(x) \cdot E_i(x)

```

Where:

- $N$ = number of experts

- $g_i(x)$ = gating weight for expert $i$ (routing)

- $E_i(x)$ = output of expert $i$

**Gating Function:**

```math
g(x) = \text{softmax}(W_g \cdot x)
g_i(x) = \frac{\exp(W_g^{(i)} \cdot x)}{\sum_j \exp(W_g^{(j)} \cdot x)}

```

### 2. Top-K Sparse Routing

**Sparse Gating (Top-K):**

```math
g(x) = \text{TopK}(\text{softmax}(W_g \cdot x + \epsilon))

```

Where:

- $\epsilon \sim \text{Gumbel}(0, 1)$ (optional exploration noise)

- Only top-$K$ values are non-zero

**Normalization after Top-K:**

```math
\hat{g}_i(x) = \frac{g_i(x)}{\sum_{j \in \text{TopK}} g_j(x)}

```

**Sparse Output:**

```math
y = \sum_{i \in \text{TopK}(g)} \hat{g}_i(x) \cdot E_i(x)

```

### 3. Computational Efficiency

**Dense Model FLOPs:**

```math
\text{FLOPs}_{dense} = \text{FLOPs}_{attention} + \text{FLOPs}_{FFN}

```

**MoE Model FLOPs:**

```math
\text{FLOPs}_{MoE} = \text{FLOPs}_{attention} + K \cdot \text{FLOPs}_{expert}

```

**Efficiency Gain:**

```math
\text{Speedup} = \frac{N \cdot \text{FLOPs}_{expert}}{K \cdot \text{FLOPs}_{expert}} = \frac{N}{K}

```

For Mixtral (N=8, K=2): $4\times$ fewer FLOPs in FFN layers!

### 4. Load Balancing

**Problem:** Without balancing, router may collapse to using few experts.

**Auxiliary Loss (Switch Transformer):**

```math
\mathcal{L}_{balance} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i

```

Where:

- $f_i = \frac{\text{tokens routed to expert } i}{\text{total tokens}}$ (actual fraction)

- $P_i = \frac{1}{T}\sum_{t=1}^{T} g_i(x_t)$ (average routing probability)

**Intuition:** Minimizing $\sum f_i P_i$ encourages uniform distribution.

**Proof of Effectiveness:**

```math
\sum_i f_i P_i \geq \sum_i f_i^2 \geq \frac{1}{N}

```

Equality holds when $f_i = 1/N$ (perfectly balanced).

### 5. Expert Capacity

**Capacity Factor $C$:**

```math
\text{Capacity} = C \cdot \frac{B \times L}{N}

```

Tokens beyond capacity are dropped or sent to secondary expert.

**Token Dropping:**
If expert $i$ receives more than capacity tokens:

- Drop excess tokens (use identity residual)

- Or route to second-choice expert

### 6. Expert Choice Routing (Reverse Routing)

**Standard:** Tokens choose experts
**Expert Choice:** Experts choose tokens

```math
\text{TopK}_{tokens}(\text{softmax}(E_i^T X))

```

Each expert selects its top-$K$ tokens to process.

**Benefits:**
- Perfect load balance (each expert processes exactly $K$ tokens)

- No dropping

- Better gradient flow

### 7. Theoretical Analysis

**Capacity vs Quality Trade-off:**

```math
\mathcal{L}_{total} = \mathcal{L}_{task} + \alpha \mathcal{L}_{balance}

```

---

### 8. Rigorous Proofs

#### 8.1 Theorem: Load Balancing Loss Minimization

**Theorem:** The auxiliary loss $\mathcal{L}_{aux} = N \sum_{i=1}^N f_i P_i$ is minimized when $f_i = P_i = 1/N$ for all $i$.

**Proof:**

By Cauchy-Schwarz inequality:

```math
\left(\sum_i f_i P_i\right) \geq \frac{\left(\sum_i \sqrt{f_i P_i}\right)^2}{N}

```

Since $\sum_i f_i = 1$ and $\sum_i P_i = 1$ (probability constraints):

Using AM-GM on each term:

```math
f_i P_i \geq 0 \text{ with equality when } f_i = P_i

```

The minimum of $\sum_i f_i P_i$ subject to $\sum_i f_i = 1$ and $\sum_i P_i = 1$ occurs when:

```math
f_i = P_i = \frac{1}{N} \quad \forall i

```

At minimum:

```math
\mathcal{L}_{aux}^{min} = N \cdot N \cdot \frac{1}{N} \cdot \frac{1}{N} = 1

```

**Corollary:** $\mathcal{L}_{aux} \geq 1$ with equality iff perfect balance. ∎

#### 8.2 Theorem: Optimal Number of Experts

**Theorem:** For a fixed parameter budget $P$ and compute budget $C$, the optimal number of experts $N^*$ satisfies:

```math
N^* = \sqrt{\frac{P \cdot K}{C}}

```

where $K$ is the top-K routing parameter.

**Proof:**

Let each expert have $p$ parameters. Total parameters: $P = N \cdot p$, so $p = P/N$.

Compute per token (ignoring routing): $C = K \cdot c(p)$ where $c(p) \propto p$.

For FFN: $c(p) = 2 \cdot d \cdot (4d) = 8d^2$ where $p = 8d^2$.

Thus $c(p) = p$ (linear in parameters).

Compute: $C = K \cdot (P/N)$, so $N = KP/C$.

But larger $N$ means more routing overhead: $C_{router} = N \cdot d$.

Total compute: $C_{total} = K \cdot (P/N) + N \cdot d$

Minimizing w.r.t. $N$:

```math
\frac{dC_{total}}{dN} = -\frac{KP}{N^2} + d = 0
N^* = \sqrt{\frac{KP}{d}} \propto \sqrt{\frac{P \cdot K}{C}}

```

∎

#### 8.3 Theorem: Routing Entropy and Generalization

**Theorem:** Higher routing entropy leads to better generalization:

```math
H(g) = -\sum_i P_i \log P_i

```

Higher $H(g)$ correlates with lower generalization gap.

**Proof Sketch:**

1. Low entropy routing (few experts used) = less model capacity utilized

2. Equivalent to implicit L0 regularization on expert usage

3. By PAC-Bayes bounds, effective parameter count $\approx N_{active} \cdot p$

4. With high entropy, $N_{active} \approx N$, utilizing full capacity

**Formal bound:**

```math
\text{Generalization gap} \leq O\left(\sqrt{\frac{N_{active} \cdot p}{m}}\right)

```

where $m$ = training samples. ∎

#### 8.4 Lemma: Gradient Flow Through Router

**Lemma:** The gradient of task loss w.r.t. router parameters is:

```math
\frac{\partial \mathcal{L}}{\partial W_g} = \sum_{i \in \text{TopK}} \frac{\partial \mathcal{L}}{\partial y} \cdot E_i(x) \cdot \frac{\partial g_i}{\partial W_g}

```

**Proof:**

Forward: $y = \sum_{i \in \text{TopK}} g_i(x) E_i(x)$

By chain rule:

```math
\frac{\partial \mathcal{L}}{\partial W_g} = \frac{\partial \mathcal{L}}{\partial y} \cdot \frac{\partial y}{\partial g} \cdot \frac{\partial g}{\partial W_g}
\frac{\partial y}{\partial g_i} = E_i(x)
\frac{\partial g_i}{\partial W_g^{(j)}} = g_i(\delta_{ij} - g_j) \cdot x^T

```

Combining:

```math
\frac{\partial \mathcal{L}}{\partial W_g^{(j)}} = \sum_i \frac{\partial \mathcal{L}}{\partial y} E_i(x) g_i(\delta_{ij} - g_j) x^T

```

**Note:** The top-K operation has zero gradient for non-selected experts, requiring straight-through estimators or auxiliary losses for training. ∎

#### 8.5 Theorem: MoE Universal Approximation

**Theorem:** An MoE layer with $N$ experts, each being a 2-layer FFN with width $w$, can approximate any continuous function on compact domain $\mathcal{X}$ to arbitrary precision.

**Proof:**

1. Each expert $E_i$ with width $w \to \infty$ is a universal approximator (by Cybenko's theorem)

2. The gating network partitions input space

3. By Stone-Weierstrass, sum of localized approximators spans continuous functions

For any $\epsilon > 0$, there exists $N, w$ such that:

```math
\sup_{x \in \mathcal{X}} |f(x) - \text{MoE}(x)| < \epsilon

```

MoE is strictly more expressive than single FFN of same compute budget. ∎

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
