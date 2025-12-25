<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=Graph Neural Networks&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---

# 🕸️ Graph Neural Networks

> **Learning on graph-structured data**

---

## 📐 Message Passing

```
GNN Layer:
  h_v^(l+1) = UPDATE(h_v^(l), AGGREGATE({h_u^(l) : u ∈ N(v)}))

GCN:
  H^(l+1) = σ(D̃^(-½) Ã D̃^(-½) H^(l) W^(l))
  where Ã = A + I (add self-loops)

GAT (with attention):
  h_v' = σ(Σ_{u∈N(v)} α_vu W h_u)
```

---

## 💻 Code Example

```python
import torch_geometric.nn as gnn

class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = gnn.GCNConv(in_dim, hidden_dim)
        self.conv2 = gnn.GCNConv(hidden_dim, out_dim)
    
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x
```

---

## 🔗 Applications

| Domain | Application |
|--------|-------------|
| **Chemistry** | Molecular property prediction |
| **Social** | Node classification |
| **Rec Systems** | User-item graphs |

---

⬅️ [Back: Architectures](../)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>

