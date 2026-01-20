<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=120&section=header&text=Federated%20Learning&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-06-45B7D1?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/federated-learning-complete.svg" width="100%">

*Caption: Federated learning trains on distributed data without centralizing it. Clients train locally, server aggregates model updates.*

---

## 📂 Overview

Federated Learning (FL) enables training ML models across decentralized data sources without sharing raw data. This preserves privacy while leveraging distributed computational resources.

---

## 📐 Mathematical Formulation

### Optimization Objective

**Global objective:**
```math
\min_w F(w) = \sum_{k=1}^{K} \frac{n_k}{n} F_k(w)
```

where:
- \(K\): number of clients
- \(n_k\): number of samples on client \(k\)
- \(n = \sum_k n_k\): total samples
- \(F_k(w) = \frac{1}{n_k} \sum_{i \in \mathcal{D}_k} \ell(w; x_i, y_i)\): local objective

### Comparison to Centralized Learning

**Centralized:**
```math
w^* = \arg\min_w \frac{1}{n} \sum_{i=1}^{n} \ell(w; x_i, y_i)
```

**Federated (equivalent if IID):**
```math
w^* = \arg\min_w \sum_{k=1}^{K} \frac{n_k}{n} \cdot \frac{1}{n_k} \sum_{i \in \mathcal{D}_k} \ell(w; x_i, y_i)
```

---

## 🔬 FedAvg Algorithm

### Complete Algorithm

```
FedAvg (Federated Averaging)
----------------------------
Input: K clients, learning rate η, rounds T, local epochs E

Initialize w₀
For round t = 0, 1, ..., T-1:
    1. Server selects subset S_t of clients (typically 10-100)
    2. Server broadcasts w_t to all clients in S_t
    3. For each client k ∈ S_t in parallel:
        w_k ← w_t
        For e = 1, ..., E:
            For batch b ∈ local_data_k:
                w_k ← w_k - η∇ℓ(w_k; b)
        Δw_k = w_k - w_t
        Send Δw_k to server
    4. Server aggregates:
        w_{t+1} = w_t + Σ_{k∈S_t} (n_k/Σ_{j∈S_t} n_j) Δw_k
```

### Convergence Analysis

**Theorem (FedAvg Convergence):** Under standard assumptions (L-smooth, μ-strongly convex), FedAvg converges with:

```math
\mathbb{E}[F(w_T)] - F(w^*) \leq O\left(\frac{1}{T} + \frac{E\eta^2L\sigma^2}{K} + E^2\eta^2L^2\Gamma\right)
```

where:
- \(\sigma^2\): variance of stochastic gradients
- \(\Gamma = F^* - \sum_k \frac{n_k}{n} F_k^*\): degree of non-IID-ness

**Key insight:** Non-IID data (\(\Gamma > 0\)) causes additional error!

### Proof Sketch

```
Per-round analysis:
w_{t+1} - w^* = w_t - w^* - η Σ_k (n_k/n) Σ_{e=1}^{E} ∇F_k(w_k^e)

Taking expectation and using smoothness:
E[||w_{t+1} - w^*||²] ≤ (1 - μη)E[||w_t - w^*||²]
                       + η²E||Σ_k ∇F_k - ∇F||²  (client drift)
                       + η²σ²/K                   (stochastic noise)

The "client drift" term is bounded by Γ (non-IID factor).
Summing over T rounds gives the convergence bound.
```

---

## 📊 FedAvg Variants

### 1. FedProx

Adds proximal term to prevent client drift:

```math
\min_w F_k(w) + \frac{\mu}{2}\|w - w_t\|^2
```

**Local update:**
```math
w_k^{e+1} = w_k^e - \eta(\nabla F_k(w_k^e) + \mu(w_k^e - w_t))
```

### 2. SCAFFOLD

Uses control variates to correct client drift:

```math
w_k^{e+1} = w_k^e - \eta(\nabla F_k(w_k^e) - c_k + c)
```

where:
- \(c_k\): client control variate (tracks \(\nabla F_k\))
- \(c\): server control variate (tracks \(\nabla F\))

**Update rule:**
```
c_k^{new} = c_k - c + (w_t - w_k)/(K·η)
c^{new} = c + (1/K) Σ_k (c_k^{new} - c_k)
```

### 3. FedNova

Normalizes updates by local computation:

```math
w_{t+1} = w_t - \tau_{eff} \cdot \frac{\sum_k \Delta w_k / \tau_k}{\sum_k n_k/n}
```

where \(\tau_k\) is the number of local steps on client \(k\).

---

## 🔐 Privacy in Federated Learning

### Differential Privacy

Add noise to gradients before aggregation:

```math
\tilde{g}_k = g_k + \mathcal{N}(0, \sigma^2 C^2 I)
```

where \(C\) is the clipping threshold.

**Privacy guarantee:**
```math
(\epsilon, \delta)\text{-DP with } \sigma \geq \frac{c \cdot C \sqrt{T \log(1/\delta)}}{n\epsilon}
```

### Secure Aggregation

Clients share secret keys to mask updates:
```
1. Client k generates random mask r_k such that Σ_k r_k = 0
2. Client sends masked update: Δw_k + r_k
3. Server computes: Σ_k (Δw_k + r_k) = Σ_k Δw_k
```
Server learns only the sum, not individual updates!

---

## 💻 Complete Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
from copy import deepcopy

class FederatedServer:
    """
    Federated Learning Server
    """
    def __init__(self, model: nn.Module, num_clients: int):
        self.global_model = model
        self.num_clients = num_clients
    
    def broadcast_model(self) -> Dict[str, torch.Tensor]:
        """Send global model to clients"""
        return deepcopy(self.global_model.state_dict())
    
    def aggregate(self, client_updates: List[Dict[str, torch.Tensor]], 
                  client_weights: List[float]) -> None:
        """
        FedAvg aggregation
        
        Args:
            client_updates: List of (model_state_dict) from clients
            client_weights: Weight for each client (typically n_k/n)
        """
        total_weight = sum(client_weights)
        
        # Weighted average of parameters
        aggregated = {}
        for key in self.global_model.state_dict().keys():
            aggregated[key] = sum(
                update[key] * (weight / total_weight)
                for update, weight in zip(client_updates, client_weights)
            )
        
        self.global_model.load_state_dict(aggregated)
    
    def aggregate_updates(self, client_deltas: List[Dict[str, torch.Tensor]], 
                         client_weights: List[float]) -> None:
        """
        Aggregate model deltas (Δw = w_k - w_t)
        """
        total_weight = sum(client_weights)
        
        with torch.no_grad():
            for key, param in self.global_model.named_parameters():
                delta = sum(
                    d[key] * (weight / total_weight)
                    for d, weight in zip(client_deltas, client_weights)
                )
                param.add_(delta)

class FederatedClient:
    """
    Federated Learning Client
    """
    def __init__(self, client_id: int, data_loader, model: nn.Module, 
                 lr: float = 0.01, local_epochs: int = 5):
        self.client_id = client_id
        self.data_loader = data_loader
        self.model = model
        self.lr = lr
        self.local_epochs = local_epochs
        self.num_samples = len(data_loader.dataset)
    
    def receive_model(self, global_state: Dict[str, torch.Tensor]) -> None:
        """Receive global model from server"""
        self.model.load_state_dict(deepcopy(global_state))
    
    def local_train(self) -> Dict[str, torch.Tensor]:
        """
        Train locally and return updated model
        """
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        
        for epoch in range(self.local_epochs):
            for batch_x, batch_y in self.data_loader:
                optimizer.zero_grad()
                output = self.model(batch_x)
                loss = F.cross_entropy(output, batch_y)
                loss.backward()
                optimizer.step()
        
        return deepcopy(self.model.state_dict())
    
    def compute_update(self, initial_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute Δw = w_local - w_global
        """
        current_state = self.model.state_dict()
        delta = {}
        for key in current_state.keys():
            delta[key] = current_state[key] - initial_state[key]
        return delta

class FedProxClient(FederatedClient):
    """
    FedProx Client with proximal term
    """
    def __init__(self, *args, mu: float = 0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.mu = mu
    
    def local_train(self) -> Dict[str, torch.Tensor]:
        """Train with proximal regularization"""
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        
        # Store global model for proximal term
        global_params = {name: param.clone() 
                        for name, param in self.model.named_parameters()}
        
        for epoch in range(self.local_epochs):
            for batch_x, batch_y in self.data_loader:
                optimizer.zero_grad()
                
                # Standard loss
                output = self.model(batch_x)
                loss = F.cross_entropy(output, batch_y)
                
                # Proximal term: (μ/2)||w - w_t||²
                prox_term = 0
                for name, param in self.model.named_parameters():
                    prox_term += ((param - global_params[name]) ** 2).sum()
                loss += (self.mu / 2) * prox_term
                
                loss.backward()
                optimizer.step()
        
        return deepcopy(self.model.state_dict())

def run_federated_training(
    server: FederatedServer,
    clients: List[FederatedClient],
    num_rounds: int = 100,
    clients_per_round: int = 10
) -> List[float]:
    """
    Run federated training loop
    """
    import random
    
    losses = []
    
    for round_idx in range(num_rounds):
        # Select random subset of clients
        selected_clients = random.sample(clients, min(clients_per_round, len(clients)))
        
        # Broadcast global model
        global_state = server.broadcast_model()
        
        # Collect client updates
        client_updates = []
        client_weights = []
        
        for client in selected_clients:
            client.receive_model(global_state)
            updated_state = client.local_train()
            client_updates.append(updated_state)
            client_weights.append(client.num_samples)
        
        # Aggregate updates
        server.aggregate(client_updates, client_weights)
        
        # Evaluate (optional)
        if round_idx % 10 == 0:
            print(f"Round {round_idx} completed")
    
    return losses

# Differential Privacy Helper
class DPFederatedClient(FederatedClient):
    """
    Federated client with differential privacy
    """
    def __init__(self, *args, clip_norm: float = 1.0, noise_scale: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.clip_norm = clip_norm
        self.noise_scale = noise_scale
    
    def compute_update(self, initial_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute DP-protected update"""
        current_state = self.model.state_dict()
        delta = {}
        
        for key in current_state.keys():
            d = current_state[key] - initial_state[key]
            
            # Clip gradient
            norm = d.norm()
            if norm > self.clip_norm:
                d = d * (self.clip_norm / norm)
            
            # Add noise
            noise = torch.randn_like(d) * self.noise_scale * self.clip_norm
            delta[key] = d + noise
        
        return delta

# Secure Aggregation (Simplified)
class SecureAggregationServer(FederatedServer):
    """
    Server with secure aggregation
    """
    def aggregate_secure(self, masked_updates: List[Dict[str, torch.Tensor]], 
                        client_weights: List[float]) -> None:
        """
        Aggregate masked updates
        Masks cancel out: Σ(Δw_k + r_k) = Σ Δw_k (since Σ r_k = 0)
        """
        # This is a simplified version - real implementation uses
        # Diffie-Hellman key exchange and pairwise masking
        self.aggregate(masked_updates, client_weights)

# Example usage
class SimpleModel(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Create synthetic federated data
def create_federated_data(num_clients: int, samples_per_client: int, non_iid: bool = False):
    """Create synthetic federated datasets"""
    from torch.utils.data import DataLoader, TensorDataset
    
    client_loaders = []
    
    for k in range(num_clients):
        if non_iid:
            # Non-IID: each client has different label distribution
            main_class = k % 10
            x = torch.randn(samples_per_client, 784)
            y = torch.full((samples_per_client,), main_class, dtype=torch.long)
            # Add some noise to labels
            noise_idx = torch.randperm(samples_per_client)[:samples_per_client // 5]
            y[noise_idx] = torch.randint(0, 10, (len(noise_idx),))
        else:
            # IID: uniform distribution
            x = torch.randn(samples_per_client, 784)
            y = torch.randint(0, 10, (samples_per_client,))
        
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        client_loaders.append(loader)
    
    return client_loaders

# Run example
print("Creating federated setup...")
model = SimpleModel()
server = FederatedServer(model, num_clients=10)

client_loaders = create_federated_data(num_clients=10, samples_per_client=100, non_iid=True)
clients = [
    FederatedClient(k, loader, deepcopy(model), lr=0.01, local_epochs=5)
    for k, loader in enumerate(client_loaders)
]

print("Running federated training...")
# run_federated_training(server, clients, num_rounds=10, clients_per_round=5)
print("Federated learning complete!")
```

---

## 📊 Key Challenges & Solutions

| Challenge | Problem | Solution |
|-----------|---------|----------|
| **Non-IID Data** | \(\Gamma > 0\) causes drift | FedProx, SCAFFOLD |
| **Communication** | Large model updates | Compression, sparse updates |
| **Stragglers** | Slow clients delay rounds | Async FL, timeout |
| **Privacy** | Gradients leak data | DP, Secure Aggregation |
| **Heterogeneity** | Different client compute | Personalization |

---

## 📚 References

| Type | Resource | Link |
|------|----------|------|
| 📄 | FedAvg Paper | [McMahan et al., 2017](https://arxiv.org/abs/1602.05629) |
| 📄 | FedProx | [Li et al., 2020](https://arxiv.org/abs/1812.06127) |
| 📄 | SCAFFOLD | [Karimireddy et al., 2020](https://arxiv.org/abs/1910.06378) |
| 📄 | Secure Aggregation | [Bonawitz et al., 2017](https://arxiv.org/abs/1611.04482) |

---

## 🗺️ Navigation

| ⬅️ Previous | 🏠 Home | ➡️ Next |
|:-----------:|:-------:|:-------:|
| [Meta-Learning](../12_meta_learning/README.md) | [Deep Learning](../README.md) | [Continual Learning](../14_continual_learning/README.md) |

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=80&section=footer" width="100%"/>
</p>
