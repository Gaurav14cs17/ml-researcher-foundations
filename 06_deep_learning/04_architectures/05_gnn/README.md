<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=120&section=header&text=Graph%20Neural%20Networks&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-06-45B7D1?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 📐 Mathematical Foundations

### Graph Notation

```math
G = (V, E, X)

```

Where:

- $V = \{v\_1, ..., v\_n\}$: nodes

- $E \subseteq V \times V$: edges

- $X \in \mathbb{R}^{n \times d}$: node features

- $A \in \{0,1\}^{n \times n}$: adjacency matrix

### Message Passing Framework

GNNs follow the message passing paradigm:

```math
h_v^{(l+1)} = \text{UPDATE}\left(h_v^{(l)}, \text{AGGREGATE}\left(\{h_u^{(l)} : u \in \mathcal{N}(v)\}\right)\right)

```

Where $\mathcal{N}(v)$ are neighbors of node $v$.

---

## 📐 Graph Convolutional Network (GCN)

### Spectral Foundation

Graph Laplacian:

```math
L = D - A = U\Lambda U^T

```

Where $D$ is the degree matrix.

Spectral convolution:

```math
g_\theta * x = Ug_\theta(\Lambda)U^T x

```

**Problem:** Computing eigenvectors is $O(n^3)$.

### GCN Layer (Simplified)

Using Chebyshev polynomial approximation:

```math
H^{(l+1)} = \sigma\left(\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} H^{(l)} W^{(l)}\right)

```

Where:

- $\tilde{A} = A + I\_n$ (add self-loops)

- $\tilde{D}\_{ii} = \sum\_j \tilde{A}\_{ij}$ (degree matrix)

- $W^{(l)} \in \mathbb{R}^{d\_l \times d\_{l+1}}$

### Per-Node Form

```math
h_v^{(l+1)} = \sigma\left(\sum_{u \in \mathcal{N}(v) \cup \{v\}} \frac{1}{\sqrt{\tilde{d}_u \tilde{d}_v}} h_u^{(l)} W^{(l)}\right)

```

The $\frac{1}{\sqrt{\tilde{d}\_u \tilde{d}\_v}}$ term normalizes by degrees.

---

## 📐 Graph Attention Network (GAT)

### Attention Mechanism

Compute attention coefficients:

```math
e_{vu} = \text{LeakyReLU}\left(a^T [Wh_v \| Wh_u]\right)

```

Normalize with softmax:

```math
\alpha_{vu} = \frac{\exp(e_{vu})}{\sum_{k \in \mathcal{N}(v)} \exp(e_{vk})}

```

### GAT Layer

```math
h_v^{(l+1)} = \sigma\left(\sum_{u \in \mathcal{N}(v)} \alpha_{vu} W h_u^{(l)}\right)

```

### Multi-Head Attention

```math
h_v^{(l+1)} = \Big\|_{k=1}^{K} \sigma\left(\sum_{u \in \mathcal{N}(v)} \alpha_{vu}^{(k)} W^{(k)} h_u^{(l)}\right)

```

---

## 📐 GraphSAGE

### Sampling-Based Aggregation

1. Sample fixed-size neighborhood

2. Aggregate neighbor features

3. Concatenate with self

```math
h_{\mathcal{N}(v)}^{(l)} = \text{AGGREGATE}^{(l)}\left(\{h_u^{(l-1)} : u \in \mathcal{N}(v)\}\right)
h_v^{(l)} = \sigma\left(W^{(l)} \cdot [h_v^{(l-1)} \| h_{\mathcal{N}(v)}^{(l)}]\right)

```

### Aggregation Functions

| Aggregator | Formula |
|------------|---------|
| **Mean** | $\frac{1}{|\mathcal{N}(v)|}\sum\_{u \in \mathcal{N}(v)} h\_u$ |
| **Max** | $\max\_{u \in \mathcal{N}(v)}(\text{ReLU}(Wh\_u))$ |
| **LSTM** | LSTM over randomly ordered neighbors |

---

## 📐 Message Passing Neural Network (MPNN)

### General Framework

**Message function:**

```math
m_v^{(l+1)} = \sum_{u \in \mathcal{N}(v)} M^{(l)}(h_v^{(l)}, h_u^{(l)}, e_{vu})

```

**Update function:**

```math
h_v^{(l+1)} = U^{(l)}(h_v^{(l)}, m_v^{(l+1)})

```

**Readout (graph-level):**

```math
\hat{y} = R(\{h_v^{(L)} : v \in V\})

```

---

## 📐 Over-Smoothing Problem

### The Issue

As layers increase, node representations converge:

```math
\lim_{l \rightarrow \infty} H^{(l)} \rightarrow \mathbf{1}\pi^T

```

Where $\pi$ is related to node degrees.

### Solutions

1. **Residual connections:** $h\_v^{(l+1)} = h\_v^{(l)} + \text{GNN}(h\_v^{(l)})$

2. **JK-Net:** Concatenate features from all layers

3. **DropEdge:** Randomly drop edges during training

4. **PairNorm:** Normalize pairwise distances

---

## 💻 Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Manual GCN Layer
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features) / in_features**0.5)
    
    def forward(self, x, adj):
        """
        x: (n_nodes, in_features)
        adj: (n_nodes, n_nodes) - normalized adjacency
        """
        support = x @ self.weight  # (n_nodes, out_features)
        output = adj @ support     # Aggregate neighbors
        return output

class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.5):
        super().__init__()
        self.gc1 = GCNLayer(in_dim, hidden_dim)
        self.gc2 = GCNLayer(hidden_dim, out_dim)
        self.dropout = dropout
    
    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

# Normalize adjacency matrix
def normalize_adj(adj):
    """Symmetric normalization: D^(-1/2) A D^(-1/2)"""
    adj = adj + torch.eye(adj.size(0), device=adj.device)  # Add self-loops
    degree = adj.sum(dim=1)
    d_inv_sqrt = degree.pow(-0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0
    return d_inv_sqrt.unsqueeze(1) * adj * d_inv_sqrt.unsqueeze(0)

# Using PyTorch Geometric
import torch_geometric.nn as gnn
from torch_geometric.nn import GCNConv, GATConv, SAGEConv

class GCN_PyG(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
    
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class GAT_PyG(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, heads=8):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, heads=heads)
        self.conv2 = GATConv(hidden_dim * heads, out_dim, heads=1)
    
    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class GraphSAGE_PyG(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)
    
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Graph-level prediction with pooling
from torch_geometric.nn import global_mean_pool, global_max_pool

class GraphClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, out_dim)
    
    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)  # Graph-level pooling
        x = self.fc(x)
        return x

```

---

## 📊 Comparison

| Model | Aggregation | Attention | Complexity |
|-------|-------------|-----------|------------|
| **GCN** | Mean (degree-weighted) | No | $O(|E|d)$ |
| **GAT** | Weighted sum | Yes | $O(|E|d + |V|d^2)$ |
| **GraphSAGE** | Mean/Max/LSTM | No | $O(|S|Kd^2)$ |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | GCN Paper | [arXiv](https://arxiv.org/abs/1609.02907) |
| 📄 | GAT Paper | [arXiv](https://arxiv.org/abs/1710.10903) |
| 📄 | GraphSAGE | [arXiv](https://arxiv.org/abs/1706.02216) |
| 📄 | MPNN Paper | [arXiv](https://arxiv.org/abs/1704.01212) |
| 💻 | PyTorch Geometric | [Docs](https://pytorch-geometric.readthedocs.io/) |
| 🇨🇳 | 图神经网络详解 | [知乎](https://zhuanlan.zhihu.com/p/75307407) |

---

## 🔗 Applications

| Domain | Application |
|--------|-------------|
| **Chemistry** | Molecular property prediction |
| **Social Networks** | Node classification, link prediction |
| **Recommendation** | User-item graphs |
| **Computer Vision** | Scene graphs, skeleton action recognition |
| **Traffic** | Traffic flow prediction |

---

⬅️ [Back: Generative](../04_generative/README.md) | ➡️ [Next: Language Models](../06_language_models/README.md)

---

⬅️ [Back: Architectures](../../README.md)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=80&section=footer" width="100%"/>
</p>
