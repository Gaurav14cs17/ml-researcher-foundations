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

## 📐 MoE Layer

```
y = Σᵢ G(x)ᵢ × Eᵢ(x)

Router/Gate: G(x) = softmax(W_g × x)
Experts: E₁(x), E₂(x), ..., Eₙ(x)

Sparse MoE (top-k routing):
  Select only top-k experts per token
  → Reduces compute while keeping capacity
```

---

## 💻 Code Example

```python
class MoELayer(nn.Module):
    def __init__(self, dim, num_experts, top_k=2):
        super().__init__()
        self.gate = nn.Linear(dim, num_experts)
        self.experts = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(num_experts)
        ])
        self.top_k = top_k
    
    def forward(self, x):
        gate_scores = F.softmax(self.gate(x), dim=-1)
        top_k_scores, top_k_indices = gate_scores.topk(self.top_k)
        # Route to top-k experts...
```

---

## 🔗 Key Models

| Model | Experts | Parameters |
|-------|---------|------------|
| **Mixtral** | 8 | 47B (12B active) |
| **GPT-4** | ? | ~1.8T |
| **Switch** | 128 | 1.6T |

---

⬅️ [Back: Architectures](../)

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=80&section=footer" width="100%"/>
</p>
