# Batch Normalization

> **Normalizing activations for faster training**

---

## 📐 Algorithm

```
For mini-batch B:
    μ_B = (1/m) Σᵢ xᵢ           # Batch mean
    σ²_B = (1/m) Σᵢ (xᵢ - μ_B)²  # Batch variance
    x̂ᵢ = (xᵢ - μ_B) / √(σ²_B + ε)  # Normalize
    yᵢ = γx̂ᵢ + β                 # Scale & shift (learnable)
```

---

## 🔑 Why It Works

```
1. Reduces internal covariate shift
2. Allows higher learning rates
3. Acts as regularization (batch noise)
4. Reduces sensitivity to initialization
```

---

## ⚠️ Issues

| Issue | Solution |
|-------|----------|
| Small batches | Group Norm |
| RNNs | Layer Norm |
| Test time | Running averages |
| Distributed | Synced BN |

---

## 💻 Code

```python
import torch.nn as nn

# In a CNN
model = nn.Sequential(
    nn.Conv2d(3, 64, 3),
    nn.BatchNorm2d(64),  # After conv, before activation
    nn.ReLU(),
)

# Training vs Eval mode matters!
model.train()  # Use batch statistics
model.eval()   # Use running averages

# PyTorch tracks running mean/var automatically
bn = nn.BatchNorm2d(64)
print(bn.running_mean.shape)  # (64,)
print(bn.running_var.shape)   # (64,)
```

---

## 📊 Normalization Comparison

| Type | Normalizes Over | Use Case |
|------|-----------------|----------|
| Batch Norm | (N, H, W) | CNNs |
| Layer Norm | (C, H, W) | Transformers |
| Instance Norm | (H, W) | Style transfer |
| Group Norm | (G, H, W) | Small batch |

---

---

➡️ [Next: Layer Norm](./layer-norm.md)
