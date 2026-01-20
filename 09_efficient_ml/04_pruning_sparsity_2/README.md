<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=1ABC9C&height=120&section=header&text=Lecture%204%20Pruning%20%26%20Sparsity%20Part%20II&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-09-1ABC9C?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

# Lecture 4: Pruning & Sparsity (Part II)

[← Back to Course](../) | [← Previous](../03_pruning_sparsity_1/) | [Next: Quantization I →](../05_quantization_1/)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/ml-researcher-foundations/blob/main/09_efficient_ml/04_pruning_sparsity_2/demo.ipynb) ← **Try the code!**

---

## 📺 Video Lecture

| Resource | Link |
|----------|------|
| 🎥 **Lecture Video** | [Watch on YouTube](https://www.youtube.com/watch?v=sZzc6tAtTrM&list=PL80kAHvQbh-pT4lCkDT53zT8DKmhE0idB&index=4) |
| 📊 **Slides** | [MIT Course Page](https://hanlab.mit.edu/courses/2023-fall-65940) |
| ⏱️ **Duration** | ~90 minutes |

### 📝 Video Summary

This lecture covers **advanced pruning topics** and the famous Lottery Ticket Hypothesis:

- **Lottery Ticket Hypothesis**: Dense networks contain sparse "winning tickets" trainable in isolation
- **Sparse formats**: COO, CSR, CSC for efficient sparse storage
- **Hardware support**: NVIDIA 2:4 structured sparsity, AMD block sparsity
- **Dynamic sparsity**: RigL, SET for sparse training from scratch
- **LLM pruning**: SparseGPT and challenges with pruning large language models
- **Practical speedups**: When sparsity translates to actual performance gains

> 💡 *"The Lottery Ticket Hypothesis changed how we think about neural network training—winning tickets exist at initialization."* — Prof. Song Han

---

![Overview](overview.png)

## The Lottery Ticket Hypothesis

> "Dense neural networks contain sparse subnetworks (winning tickets) that can train in isolation to match the full network's accuracy."

### Key Findings

1. Random init → Train → Prune → **Reset to original init** → Retrain
2. The "winning ticket" trains just as well as the original!
3. But random reinitialization doesn't work

```
Finding Winning Tickets:
1. Train network to completion
2. Prune smallest weights
3. Reset remaining weights to ORIGINAL initialization
4. Retrain from scratch → Same accuracy!
```

---

## Structured vs Unstructured

| Aspect | Unstructured | Structured |
|--------|-------------|------------|
| Granularity | Individual weights | Channels/filters |
| Compression | Higher (10-100x) | Lower (2-4x) |
| Hardware | Needs sparse support | Any hardware |
| Real speedup | Often none on GPU | Actual speedup |

---

## Sparse Formats

### COO (Coordinate)
Store (row, col, value) for each non-zero:
```python

# Dense: [[1, 0, 2], [0, 0, 3], [4, 0, 0]]
# COO: rows=[0,0,1,2], cols=[0,2,2,0], vals=[1,2,3,4]
```

### CSR (Compressed Sparse Row)
More efficient for row-wise operations:
```python

# vals = [1, 2, 3, 4]
# col_idx = [0, 2, 2, 0]  
# row_ptr = [0, 2, 3, 4]  # Start of each row
```

---

## Hardware Support for Sparsity

| Hardware | Sparse Support |
|----------|---------------|
| NVIDIA Ampere (A100) | 2:4 structured sparsity |
| AMD MI250 | Block sparsity |
| Apple M1/M2 | Limited |
| Intel | Sparse tensor ops |

### NVIDIA 2:4 Sparsity
Every 4 consecutive elements, exactly 2 must be zero:
```
[1.0, 0.0, 0.5, 0.0]  ✓ Valid
[1.0, 0.5, 0.0, 0.0]  ✓ Valid
[1.0, 0.5, 0.3, 0.0]  ✗ Invalid
```
**Result:** 2x speedup with 50% sparsity!

---

## Dynamic Sparsity

Instead of fixed sparsity patterns, let them change during training:

1. **RigL** - Periodic weight regrowth
2. **SET** - Sparse evolutionary training
3. **SNIP** - Prune at initialization

```python

# RigL pseudo-code
for epoch in training:
    if epoch % update_freq == 0:

        # Drop smallest magnitude weights
        drop_mask = magnitude_threshold(weights)

        # Grow new weights based on gradient
        grow_mask = gradient_threshold(gradients)
        update_sparsity_mask(drop_mask, grow_mask)
```

---

## Pruning LLMs

Pruning large language models is different:

| Challenge | Why |
|-----------|-----|
| Emergent abilities | May disappear with pruning |
| Calibration data | Need representative samples |
| Layer sensitivity | Some layers can't be pruned |

### SparseGPT
- Prunes LLMs to 50%+ sparsity
- Uses approximate second-order information
- One-shot (no retraining needed)

---

## Key Papers

- 📄 [Lottery Ticket Hypothesis](https://arxiv.org/abs/1803.03635)
- 📄 [RigL: Dynamic Sparsity](https://arxiv.org/abs/1911.11134)
- 📄 [SparseGPT](https://arxiv.org/abs/2301.00774)

---

## 📐 Mathematical Foundations & Proofs

### Lottery Ticket Hypothesis Formalization

Let $f(x; \theta)$ be a neural network with parameters $\theta \in \mathbb{R}^d$.

**Lottery Ticket Conjecture:**

For a dense network $f(x; \theta_0)$ with random initialization $\theta_0$, there exists:
- A sparse mask $m \in \{0,1\}^d$ with $\|m\|_0 \ll d$
- Such that training $f(x; m \odot \theta_0)$ achieves comparable accuracy to training $f(x; \theta_0)$

**Formal Statement:**

$$
\text{Acc}(f(\cdot; m \odot \theta^*_{sparse})) \approx \text{Acc}(f(\cdot; \theta^*_{dense}))
$$

where:
- $\theta^*_{sparse}$ = trained from $m \odot \theta_0$
- $\theta^*_{dense}$ = trained from $\theta_0$
- $\|m\|_0 / d$ can be as low as 10-20%

**Key Insight:** Initialization matters—random reinitialization fails.

---

### Sparse Matrix Formats

#### COO (Coordinate) Storage

$$
\text{Memory}_{COO} = \text{nnz} \times (2 \times \text{idx\_size} + \text{val\_size})
$$

For nnz non-zeros with INT32 indices and FP32 values:

$$
\text{Memory}_{COO} = \text{nnz} \times (2 \times 4 + 4) = 12 \times \text{nnz} \text{ bytes}
$$

#### CSR (Compressed Sparse Row) Storage

$$
\text{Memory}_{CSR} = \text{nnz} \times (\text{idx\_size} + \text{val\_size}) + (n_{rows}+1) \times \text{ptr\_size}
$$

For matrix $A \in \mathbb{R}^{m \times n}$:

$$
\text{Memory}_{CSR} = 8 \times \text{nnz} + 4 \times (m+1) \text{ bytes}
$$

**CSR is more efficient for row-wise operations** (common in ML).

#### Crossover Point

Dense is better when:

$$
m \times n \times \text{val\_size} < \text{nnz} \times (\text{idx\_size} + \text{val\_size})
$$

For FP32 with INT32 indices:

$$
\frac{\text{nnz}}{m \times n} > \frac{4}{8} = 0.5
$$

**Conclusion:** Use dense format when >50% non-zeros.

---

### 2:4 Structured Sparsity Analysis

**Constraint:** In every group of 4 consecutive elements, exactly 2 are non-zero.

**Sparsity:** 50% exactly

**Representability:** How well can 2:4 approximate arbitrary weights?

For weights $W = [w_1, w_2, w_3, w_4]$, the 2:4 approximation keeps the 2 largest magnitude:

$$
\hat{W} = [w_1, 0, w_3, 0] \text{ if } |w_1|, |w_3| \geq |w_2|, |w_4|
$$

**Approximation Error:**

$$
\|\hat{W} - W\|_F^2 = w_2^2 + w_4^2
$$

**Theorem:** 2:4 pruning retains the top-50% magnitude weights within each group.

---

### RigL: Dynamic Sparse Training

**Algorithm:**

At each update step:
1. **Drop:** Remove fraction $\alpha$ of weights with smallest magnitude
2. **Grow:** Add same number of weights with largest gradient magnitude
3. **Train:** Update remaining weights

**Drop criterion:**

$$
\mathcal{D} = \{(i,j) : |w_{ij}| < \tau_{\text{drop}}\}
$$

**Grow criterion:**

$$
\mathcal{G} = \{(i,j) : |g_{ij}| > \tau_{\text{grow}} \land (i,j) \notin \text{active}\}
$$

where $g_{ij} = \frac{\partial \mathcal{L}}{\partial w_{ij}}$.

**Update schedule for drop fraction:**

$$
\alpha_t = \alpha_0 \left(1 - \frac{t}{T}\right)^3
$$

Cubic decay: More exploration early, more exploitation later.

---

### SparseGPT: One-Shot LLM Pruning

**Problem:** Prune weight matrix $W$ to minimize reconstruction error:

$$
\min_{\hat{W}} \|WX - \hat{W}X\|_F^2 \quad \text{s.t.} \quad \hat{W} \text{ is } s\text{-sparse}
$$

**Solution using Hessian information:**

For each column $j$, the optimal update when pruning $w_j$:

$$
\delta_{j:} = -\frac{w_j}{[H^{-1}]_{jj}} \cdot [H^{-1}]_{:j}
$$

where $H = XX^T$ is the Hessian approximation.

**Algorithm (column-by-column):**
```
for j in columns:
    if should_prune(w_j):

        # Compensate remaining weights
        w_{j+1:} += w_j * H^{-1}_{j+1:,j} / H^{-1}_{jj}
        w_j = 0
```

**Complexity:** $O(d^2)$ per layer (one-shot, no retraining!)

---

### Layer Sensitivity Analysis

**Sensitivity score for layer $l$:**

$$
S_l = \frac{\Delta \mathcal{L}}{\Delta s_l}
$$

where $\Delta \mathcal{L}$ is loss increase when pruning layer $l$ to sparsity $s_l$.

**Empirical finding:** First and last layers are most sensitive.

**Proof sketch:**
- First layer: Directly processes input features; losing weights loses input information
- Last layer: Directly produces output; critical for final prediction
- Middle layers: Redundant representations can compensate

**Practical rule:** Keep first/last layers denser, prune middle layers more aggressively.

---

## 🧮 Key Derivations

### Sparse Matrix-Vector Multiplication Speedup

For sparse matrix $A$ with nnz non-zeros, computing $y = Ax$:

**Dense:** $O(mn)$ operations
**Sparse:** $O(\text{nnz})$ operations

**Speedup ratio:**

$$
\text{Speedup} = \frac{mn}{\text{nnz}} = \frac{1}{1-s}
$$

At 90% sparsity: theoretical 10× speedup.

**Actual speedup is lower due to:**
- Sparse format overhead
- Irregular memory access
- Poor cache utilization

---

### Information-Theoretic View of Pruning

**Hypothesis:** Pruned networks preserve essential information.

$$
I(X; \hat{Y}) \approx I(X; Y)
$$

where:
- $Y$ = dense network output
- $\hat{Y}$ = pruned network output
- $I(\cdot; \cdot)$ = mutual information

**Data Processing Inequality:** $I(X; \hat{Y}) \leq I(X; Y)$

Equality (approximately) holds when pruned weights carry redundant information.

---

## 💻 Code Examples

### Lottery Ticket Finding

```python
import torch
import torch.nn as nn
import copy

def find_lottery_ticket(model, train_fn, prune_ratio=0.2, iterations=5):
    """
    Find winning ticket via iterative magnitude pruning
    """

    # Save original initialization
    original_state = copy.deepcopy(model.state_dict())
    
    for iteration in range(iterations):

        # Train the model
        train_fn(model)
        
        # Compute pruning mask based on magnitude
        masks = {}
        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                threshold = torch.quantile(param.abs(), prune_ratio)
                masks[name] = (param.abs() > threshold).float()
        
        # Reset to original init with mask applied
        for name, param in model.named_parameters():
            if name in masks:
                param.data = original_state[name] * masks[name]
    
    return model, masks

# Apply 2:4 structured sparsity
def apply_2_4_sparsity(weight):
    """
    Apply NVIDIA 2:4 structured sparsity pattern
    Keep 2 largest values in every group of 4
    """
    shape = weight.shape
    weight_flat = weight.view(-1, 4)
    
    # Find indices of 2 smallest in each group
    _, indices = weight_flat.abs().topk(2, dim=1, largest=False)
    
    # Create mask
    mask = torch.ones_like(weight_flat)
    mask.scatter_(1, indices, 0)
    
    return (weight.view(-1, 4) * mask).view(shape)

# SparseGPT-style one-shot pruning
def sparse_gpt_prune(W, X, sparsity=0.5):
    """
    One-shot pruning with Hessian-based weight update
    
    W: weight matrix (out_features, in_features)
    X: calibration inputs (batch, in_features)
    """

    # Compute Hessian approximation
    H = X.T @ X / X.shape[0]
    H_inv = torch.linalg.inv(H + 1e-4 * torch.eye(H.shape[0]))
    
    W_pruned = W.clone()
    
    for j in range(W.shape[1]):

        # Find weights to prune in this column
        col = W_pruned[:, j]
        threshold = torch.quantile(col.abs(), sparsity)
        prune_mask = col.abs() < threshold
        
        if prune_mask.any():

            # Compute compensation for remaining columns
            for i in torch.where(prune_mask)[0]:
                if j + 1 < W.shape[1]:
                    delta = -W_pruned[i, j] / H_inv[j, j] * H_inv[j, j+1:]
                    W_pruned[i, j+1:] += delta
                W_pruned[i, j] = 0
    
    return W_pruned
```

### RigL Dynamic Sparsity

```python
class RigLScheduler:
    """
    RigL: Rigging the Lottery - Dynamic sparse training
    """
    def __init__(self, model, sparsity=0.9, update_freq=100, 
                 drop_fraction=0.3, T_end=50000):
        self.model = model
        self.sparsity = sparsity
        self.update_freq = update_freq
        self.drop_fraction_init = drop_fraction
        self.T_end = T_end
        self.step_count = 0
        
        # Initialize sparse masks
        self.masks = {}
        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                mask = torch.rand_like(param) > sparsity
                self.masks[name] = mask.float()
                param.data *= self.masks[name]
    
    def step(self, gradients):
        """Call after computing gradients"""
        self.step_count += 1
        
        if self.step_count % self.update_freq != 0:
            return
        
        # Cosine decay for drop fraction
        drop_fraction = self.drop_fraction_init * (
            1 + math.cos(math.pi * self.step_count / self.T_end)
        ) / 2
        
        for name, param in self.model.named_parameters():
            if name not in self.masks:
                continue
            
            mask = self.masks[name]
            grad = gradients[name]
            
            # Number of connections to update
            n_active = mask.sum().int()
            n_update = int(drop_fraction * n_active)
            
            # DROP: Remove smallest magnitude active weights
            active_weights = param.data.abs() * mask
            _, drop_indices = active_weights.view(-1).topk(n_update, largest=False)
            
            # GROW: Activate largest gradient inactive positions
            inactive_grads = grad.abs() * (1 - mask)
            _, grow_indices = inactive_grads.view(-1).topk(n_update, largest=True)
            
            # Update mask
            flat_mask = mask.view(-1)
            flat_mask[drop_indices] = 0
            flat_mask[grow_indices] = 1
            self.masks[name] = flat_mask.view(mask.shape)
            
            # Zero out dropped weights
            param.data *= self.masks[name]
```

---

## 🎯 Where Used

| Concept | Applications |
|---------|-------------|
| Lottery Ticket | Understanding generalization |
| 2:4 Sparsity | NVIDIA Ampere GPU acceleration |
| RigL | Sparse training from scratch |
| SparseGPT | LLM compression |

---

## 🗺️ Navigation

| ⬅️ Previous | 🏠 Home | ➡️ Next |
|:-----------:|:-------:|:-------:|
| [← Pruning & Sparsity I](../03_pruning_sparsity_1/README.md) | [Efficient ML](../README.md) | [Quantization I →](../05_quantization_1/README.md) |

---

## 📚 References

| Type | Resource | Link |
|------|----------|------|
| 📄 | Lottery Ticket Hypothesis | [arXiv](https://arxiv.org/abs/1803.03635) |
| 📄 | RigL | [arXiv](https://arxiv.org/abs/1911.11134) |
| 📄 | SparseGPT | [arXiv](https://arxiv.org/abs/2301.00774) |
| 📄 | NVIDIA 2:4 Sparsity | [NVIDIA Blog](https://developer.nvidia.com/blog/accelerating-inference-with-sparsity-using-ampere-and-tensorrt/) |
| 🎥 | MIT 6.5940 TinyML | [Course](https://hanlab.mit.edu/courses/2024-fall-65940) |

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=1ABC9C&height=80&section=footer" width="100%"/>
</p>
