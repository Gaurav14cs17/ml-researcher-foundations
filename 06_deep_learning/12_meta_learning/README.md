<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=120&section=header&text=Meta-Learning&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-06-45B7D1?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/meta-learning-complete.svg" width="100%">

*Caption: Meta-learning trains on many tasks to learn good initialization or learning strategy. Few-shot learning is a key application.*

---

## 📂 Overview

Meta-learning, or "learning to learn," trains models across many tasks so they can quickly adapt to new tasks with minimal data. Unlike traditional learning which optimizes for a single task, meta-learning optimizes for fast adaptation across a distribution of tasks.

---

## 📐 Mathematical Framework

### Task Distribution

A meta-learning problem consists of:

- **Task distribution:** \(p(\mathcal{T})\)
- **Each task:** \(\mathcal{T}_i = \{D_i^{train}, D_i^{test}\}\)
- **Support set:** \(D_i^{train}\) (few examples for adaptation)
- **Query set:** \(D_i^{test}\) (for evaluation)

**Meta-objective:**
```math
\min_\theta \mathbb{E}_{\mathcal{T} \sim p(\mathcal{T})} \left[ \mathcal{L}(\mathcal{A}(\theta, D^{train}), D^{test}) \right]
```

where \(\mathcal{A}\) is the adaptation algorithm.

---

## 🔬 MAML: Model-Agnostic Meta-Learning

### Core Idea

Learn an initialization \(\theta\) that can be quickly adapted to any task with a few gradient steps.

### Algorithm

**Outer loop (meta-update):**
```math
\theta \leftarrow \theta - \alpha \nabla_\theta \sum_{\mathcal{T}_i} \mathcal{L}_{\mathcal{T}_i}(f_{\theta'_i})
```

**Inner loop (task-specific adaptation):**
```math
\theta'_i = \theta - \beta \nabla_\theta \mathcal{L}_{\mathcal{T}_i}(f_\theta)
```

### Complete Derivation

**One inner step MAML:**
```
Given: Initial parameters θ, task Tᵢ with support set Sᵢ

Inner loop:
θ'ᵢ = θ - β∇_θ L_Sᵢ(f_θ)

Outer loop objective:
min_θ Σᵢ L_Qᵢ(f_{θ'ᵢ})  where Qᵢ is query set

Gradient (requires second derivatives):
∂/∂θ L_Qᵢ(f_{θ'ᵢ}) = ∂L_Qᵢ/∂θ'ᵢ · ∂θ'ᵢ/∂θ

where:
∂θ'ᵢ/∂θ = I - β ∂²L_Sᵢ/∂θ²  (Hessian term)

This is why MAML needs second-order derivatives!
```

### First-Order MAML (FOMAML)

Approximate by ignoring second-order terms:
```math
\nabla_\theta \mathcal{L}(f_{\theta'}) \approx \nabla_{\theta'} \mathcal{L}(f_{\theta'})
```

```
Justification:
∂θ'/∂θ ≈ I  (ignore Hessian)

This works surprisingly well in practice!
Much faster: O(n) vs O(n²) per step
```

---

## 📊 Prototypical Networks

### Intuition

Classify by computing distance to class "prototypes" (mean embeddings).

### Algorithm

**1. Compute prototypes:**
```math
c_k = \frac{1}{|S_k|} \sum_{(x_i, y_i) \in S_k} f_\theta(x_i)
```

where \(S_k\) is the support set for class \(k\).

**2. Classify by distance:**
```math
p(y = k | x) = \frac{\exp(-d(f_\theta(x), c_k))}{\sum_{k'} \exp(-d(f_\theta(x), c_{k'}))}
```

### Mathematical Justification

**Bregman divergence connection:**
```
For Euclidean distance and regular Bregman divergence,
the prototype (cluster mean) is optimal.

Proof:
argmin_c Σᵢ ||xᵢ - c||² 

Taking derivative and setting to 0:
-2 Σᵢ (xᵢ - c) = 0
c = (1/n) Σᵢ xᵢ  ✓

The mean minimizes sum of squared distances.
```

**Why it works for few-shot:**
- Prototypes are robust to noise (averaging)
- Simple, non-parametric classifier
- Embedding network learns task-agnostic features

---

## 📐 Matching Networks

### Attention-based Classification

```math
p(y|x, S) = \sum_{(x_i, y_i) \in S} a(x, x_i) \cdot y_i
```

where attention weights:
```math
a(x, x_i) = \frac{\exp(c(f(x), g(x_i)))}{\sum_j \exp(c(f(x), g(x_j)))}
```

**Key differences from Prototypical:**
- Soft attention over all support examples
- Separate embedding functions \(f\) (query) and \(g\) (support)
- Uses cosine similarity instead of Euclidean distance

---

## 🎯 Reptile: Simplified Meta-Learning

### Algorithm

```
for each iteration:
    Sample task Tᵢ
    θ̃ = θ
    for k steps:
        θ̃ = θ̃ - α∇L_Tᵢ(θ̃)  # SGD on task
    θ = θ + ε(θ̃ - θ)  # Move toward adapted params
```

### Mathematical Interpretation

```
Reptile gradient ≈ MAML gradient (first-order):

∇_Reptile = θ̃ - θ = -α Σₖ ∇L_k

Taylor expansion shows connection to MAML:
θ̃ - θ ≈ -αg₁ - α²H₁g₁ + O(α³)

where g₁ = ∇L, H₁ = ∇²L

The second term (Hessian) provides curvature info
similar to MAML's second-order gradient.
```

---

## 💻 Complete Implementations

### MAML Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

class MAML:
    def __init__(self, model, inner_lr=0.01, outer_lr=0.001, inner_steps=5):
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        self.meta_optimizer = torch.optim.Adam(model.parameters(), lr=outer_lr)
    
    def inner_loop(self, support_x, support_y, train=True):
        """
        Adapt model parameters on support set
        
        Returns adapted parameters (not in-place)
        """
        # Clone parameters for task-specific adaptation
        fast_weights = {name: param.clone() for name, param in self.model.named_parameters()}
        
        for _ in range(self.inner_steps):
            # Forward with current fast weights
            logits = self.functional_forward(support_x, fast_weights)
            loss = F.cross_entropy(logits, support_y)
            
            # Compute gradients
            grads = torch.autograd.grad(
                loss, fast_weights.values(),
                create_graph=train  # Need graph for outer loop gradient
            )
            
            # Update fast weights
            fast_weights = {
                name: param - self.inner_lr * grad
                for (name, param), grad in zip(fast_weights.items(), grads)
            }
        
        return fast_weights
    
    def functional_forward(self, x, weights):
        """
        Forward pass using given weights (not model's parameters)
        """
        # Assuming simple MLP: Linear -> ReLU -> Linear
        x = F.linear(x, weights['fc1.weight'], weights['fc1.bias'])
        x = F.relu(x)
        x = F.linear(x, weights['fc2.weight'], weights['fc2.bias'])
        return x
    
    def meta_train_step(self, tasks):
        """
        One meta-training step on a batch of tasks
        
        tasks: list of (support_x, support_y, query_x, query_y)
        """
        meta_loss = 0
        
        for support_x, support_y, query_x, query_y in tasks:
            # Inner loop: adapt to task
            fast_weights = self.inner_loop(support_x, support_y, train=True)
            
            # Outer loop: evaluate on query set
            query_logits = self.functional_forward(query_x, fast_weights)
            task_loss = F.cross_entropy(query_logits, query_y)
            meta_loss += task_loss
        
        meta_loss /= len(tasks)
        
        # Meta-update
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
        
        return meta_loss.item()
    
    def adapt_and_evaluate(self, support_x, support_y, query_x, query_y):
        """
        Adapt to new task and evaluate
        """
        fast_weights = self.inner_loop(support_x, support_y, train=False)
        
        with torch.no_grad():
            query_logits = self.functional_forward(query_x, fast_weights)
            pred = query_logits.argmax(dim=1)
            acc = (pred == query_y).float().mean()
        
        return acc.item()

class MAMLModel(nn.Module):
    """Simple model for MAML"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Example usage
model = MAMLModel(784, 64, 5)  # 5-way classification
maml = MAML(model, inner_lr=0.01, outer_lr=0.001, inner_steps=5)

# Simulate a task batch
tasks = []
for _ in range(4):  # 4 tasks per meta-batch
    support_x = torch.randn(5, 784)   # 5-shot
    support_y = torch.randint(0, 5, (5,))
    query_x = torch.randn(15, 784)
    query_y = torch.randint(0, 5, (15,))
    tasks.append((support_x, support_y, query_x, query_y))

loss = maml.meta_train_step(tasks)
print(f"Meta-training loss: {loss:.4f}")
```

### Prototypical Networks Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvEncoder(nn.Module):
    """
    Embedding network for images
    """
    def __init__(self, in_channels=1, hidden_dim=64, embedding_dim=64):
        super().__init__()
        
        self.encoder = nn.Sequential(
            self._conv_block(in_channels, hidden_dim),
            self._conv_block(hidden_dim, hidden_dim),
            self._conv_block(hidden_dim, hidden_dim),
            self._conv_block(hidden_dim, embedding_dim),
        )
        
    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
    
    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)

class PrototypicalNetwork(nn.Module):
    """
    Prototypical Networks for few-shot classification
    """
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
    
    def compute_prototypes(self, support_embeddings, support_labels, n_way):
        """
        Compute class prototypes as mean of support embeddings
        
        Args:
            support_embeddings: (n_support, embedding_dim)
            support_labels: (n_support,)
            n_way: number of classes
        
        Returns:
            prototypes: (n_way, embedding_dim)
        """
        prototypes = torch.zeros(n_way, support_embeddings.size(1), 
                                  device=support_embeddings.device)
        
        for c in range(n_way):
            mask = (support_labels == c)
            prototypes[c] = support_embeddings[mask].mean(dim=0)
        
        return prototypes
    
    def forward(self, support_x, support_y, query_x, n_way):
        """
        Compute query predictions given support set
        
        Args:
            support_x: (n_support, *input_shape)
            support_y: (n_support,)
            query_x: (n_query, *input_shape)
            n_way: number of classes
        
        Returns:
            log_probs: (n_query, n_way)
        """
        # Embed support and query
        support_embeddings = self.encoder(support_x)
        query_embeddings = self.encoder(query_x)
        
        # Compute prototypes
        prototypes = self.compute_prototypes(support_embeddings, support_y, n_way)
        
        # Compute distances to prototypes
        # Using negative squared Euclidean distance
        dists = self.euclidean_dist(query_embeddings, prototypes)
        
        # Convert to probabilities
        log_probs = F.log_softmax(-dists, dim=1)
        
        return log_probs
    
    def euclidean_dist(self, x, y):
        """
        Compute pairwise Euclidean distances
        
        Args:
            x: (n, d)
            y: (m, d)
        
        Returns:
            dist: (n, m)
        """
        n = x.size(0)
        m = y.size(0)
        
        # ||x - y||² = ||x||² + ||y||² - 2<x, y>
        xx = (x ** 2).sum(dim=1, keepdim=True).expand(n, m)
        yy = (y ** 2).sum(dim=1, keepdim=True).expand(m, n).t()
        
        dist = xx + yy - 2 * torch.mm(x, y.t())
        
        return dist

def train_prototypical(model, train_loader, optimizer, n_way, n_shot, n_query):
    """
    Training loop for Prototypical Networks
    """
    model.train()
    total_loss = 0
    total_acc = 0
    
    for batch_idx, (x, y) in enumerate(train_loader):
        # Sample episode
        # x: (n_way * (n_shot + n_query), *input_shape)
        
        # Separate support and query
        support_x = x[:n_way * n_shot]
        support_y = torch.arange(n_way).repeat_interleave(n_shot)
        query_x = x[n_way * n_shot:]
        query_y = torch.arange(n_way).repeat_interleave(n_query)
        
        # Forward
        log_probs = model(support_x, support_y, query_x, n_way)
        
        # Loss
        loss = F.nll_loss(log_probs, query_y)
        
        # Accuracy
        pred = log_probs.argmax(dim=1)
        acc = (pred == query_y).float().mean()
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_acc += acc.item()
    
    return total_loss / len(train_loader), total_acc / len(train_loader)

# Example
encoder = ConvEncoder(in_channels=1, embedding_dim=64)
proto_net = PrototypicalNetwork(encoder)

# Simulate an episode
support_x = torch.randn(25, 1, 28, 28)  # 5-way 5-shot
support_y = torch.arange(5).repeat(5)
query_x = torch.randn(75, 1, 28, 28)    # 15 queries per class

log_probs = proto_net(support_x, support_y, query_x, n_way=5)
print(f"Log probabilities shape: {log_probs.shape}")  # (75, 5)
```

### Reptile Implementation

```python
import torch
import torch.nn as nn
from copy import deepcopy

class Reptile:
    """
    Reptile meta-learning algorithm
    """
    def __init__(self, model, inner_lr=0.01, outer_lr=1.0, inner_steps=5):
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr  # Reptile step size (ε)
        self.inner_steps = inner_steps
    
    def inner_loop(self, support_x, support_y):
        """
        Train model on a single task
        Returns: adapted model parameters
        """
        # Clone model
        adapted_model = deepcopy(self.model)
        optimizer = torch.optim.SGD(adapted_model.parameters(), lr=self.inner_lr)
        
        for _ in range(self.inner_steps):
            logits = adapted_model(support_x)
            loss = nn.functional.cross_entropy(logits, support_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return adapted_model
    
    def meta_train_step(self, tasks):
        """
        One meta-training step
        
        tasks: list of (support_x, support_y)
        """
        # Store original parameters
        original_params = {name: param.clone() 
                          for name, param in self.model.named_parameters()}
        
        # Accumulate parameter updates from all tasks
        accumulated_grads = {name: torch.zeros_like(param) 
                            for name, param in self.model.named_parameters()}
        
        for support_x, support_y in tasks:
            # Inner loop
            adapted_model = self.inner_loop(support_x, support_y)
            
            # Compute Reptile gradient: θ̃ - θ
            for name, param in adapted_model.named_parameters():
                accumulated_grads[name] += param - original_params[name]
        
        # Average over tasks
        for name in accumulated_grads:
            accumulated_grads[name] /= len(tasks)
        
        # Reptile update: θ = θ + ε(θ̃ - θ)
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.add_(self.outer_lr * accumulated_grads[name])

# Example
model = nn.Sequential(
    nn.Linear(784, 64),
    nn.ReLU(),
    nn.Linear(64, 5)
)

reptile = Reptile(model, inner_lr=0.01, outer_lr=0.1, inner_steps=10)

# Simulate tasks
tasks = [(torch.randn(10, 784), torch.randint(0, 5, (10,))) for _ in range(4)]
reptile.meta_train_step(tasks)
print("Reptile step completed")
```

---

## 📊 Comparison of Methods

| Method | Second-Order | Memory | Speed | Performance |
|--------|--------------|--------|-------|-------------|
| **MAML** | Yes | High | Slow | Best |
| **FOMAML** | No | Medium | Fast | Good |
| **Reptile** | No | Low | Fast | Good |
| **Prototypical** | No | Low | Fast | Good (metric) |
| **Matching** | No | Low | Fast | Good (metric) |

---

## 📚 References

| Type | Resource | Link |
|------|----------|------|
| 📄 | MAML | [Finn et al., ICML 2017](https://arxiv.org/abs/1703.03400) |
| 📄 | Prototypical Networks | [Snell et al., NeurIPS 2017](https://arxiv.org/abs/1703.05175) |
| 📄 | Matching Networks | [Vinyals et al., NeurIPS 2016](https://arxiv.org/abs/1606.04080) |
| 📄 | Reptile | [Nichol et al., 2018](https://arxiv.org/abs/1803.02999) |

---

## 🗺️ Navigation

| ⬅️ Previous | 🏠 Home | ➡️ Next |
|:-----------:|:-------:|:-------:|
| [Multi-Task Learning](../11_multi_task_learning/README.md) | [Deep Learning](../README.md) | [Federated Learning](../13_federated_learning/README.md) |

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=80&section=footer" width="100%"/>
</p>
