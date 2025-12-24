# MoE Routing

> **How tokens choose which experts to use**

---

## 📐 Top-K Routing

```
Input: x (token embedding)
Gate: G(x) = softmax(W_g · x)  # Scores for each expert
Selection: Select top-K experts

Output: y = Σᵢ∈top-K G(x)ᵢ · Expertᵢ(x)
```

---

## 🔑 Routing Mechanisms

| Type | Method | Used In |
|------|--------|---------|
| Top-1 | Use 1 best expert | Switch Transformer |
| Top-2 | Use 2 best experts | GShard, Mixtral |
| Expert Choice | Experts choose tokens | Expert Choice |
| Soft MoE | Weighted average | Soft MoE |

---

## ⚠️ Load Balancing

```
Problem: Tokens all go to same expert

Solutions:
1. Auxiliary loss: L_aux = CV(expert_loads)
2. Expert capacity: Max tokens per expert
3. Random routing: Add noise to gate scores
```

---

## 💻 Code

```python
class TopKRouter(nn.Module):
    def __init__(self, d_model, num_experts, k=2):
        super().__init__()
        self.gate = nn.Linear(d_model, num_experts)
        self.k = k
    
    def forward(self, x):
        # x: (batch, seq, d_model)
        logits = self.gate(x)  # (batch, seq, num_experts)
        
        # Top-K selection
        topk_vals, topk_idx = torch.topk(logits, self.k, dim=-1)
        topk_weights = F.softmax(topk_vals, dim=-1)
        
        return topk_idx, topk_weights
    
    def load_balance_loss(self, logits):
        """Encourage balanced expert usage"""
        probs = F.softmax(logits, dim=-1)
        # Fraction of tokens per expert
        f = probs.mean(dim=(0, 1))
        # Probability mass per expert
        P = probs.sum(dim=(0, 1)) / probs.sum()
        return (f * P).sum() * self.num_experts
```

---

<- [Back](./README.md)


