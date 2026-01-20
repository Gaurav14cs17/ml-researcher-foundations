<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=E74C3C&height=120&section=header&text=Knowledge%20Distillation&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-08-E74C3C?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 📐 Mathematical Foundations

### 1. Hinton's Knowledge Distillation

**Complete Loss Function:**

```math
\mathcal{L}_{KD} = \alpha \cdot \mathcal{L}_{hard} + (1-\alpha) \cdot T^2 \cdot \mathcal{L}_{soft}
```

**Hard Loss (Ground Truth):**

```math
\mathcal{L}_{hard} = -\sum_i y_i \log p_i^s = H(y, p^s)
```

**Soft Loss (Teacher Knowledge):**

```math
\mathcal{L}_{soft} = D_{KL}(p^t_T \| p^s_T) = \sum_i p_i^{t,T} \log \frac{p_i^{t,T}}{p_i^{s,T}}
```

Where temperature-scaled softmax:

```math
p_i^T = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}
```

### 2. Temperature Analysis (Mathematical Proof)

**Theorem:** Higher temperature reveals more information about class relationships.

**Taylor Expansion for large $T$:**

Let $z\_i$ be logits and $\bar{z} = \frac{1}{n}\sum\_i z\_i$

```math
\exp(z_i/T) \approx 1 + z_i/T + O(1/T^2)
```

Therefore:

```math
p_i^T = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)} \approx \frac{1 + z_i/T}{n + \sum_j z_j/T}
\approx \frac{1}{n} + \frac{z_i - \bar{z}}{nT} + O(1/T^2)
```

**Interpretation:**
- $T \to \infty$: Uniform distribution (no information)
- $T = 1$: Standard softmax (hard labels)
- Optimal $T$: Balances information and noise

### 3. Gradient Analysis

**Gradient w.r.t. student logits $z\_s$:**

```math
\frac{\partial \mathcal{L}_{soft}}{\partial z_s^{(i)}} = \frac{1}{T}\left(p_i^{s,T} - p_i^{t,T}\right)
```

**Why $T^2$ scaling?**

The gradient scales as $1/T$, and we want gradients comparable to hard loss:

```math
\frac{\partial (T^2 \mathcal{L}_{soft})}{\partial z_s^{(i)}} = T\left(p_i^{s,T} - p_i^{t,T}\right)
```

This makes the soft loss contribution independent of temperature choice.

### 4. Information-Theoretic Perspective

**Mutual Information:**

```math
I(X; Y) = H(Y) - H(Y|X)
```

**Knowledge in Soft Labels:**
The soft labels contain more information than hard labels:

```math
I(X; p^t_T) > I(X; y)
```

Because:

```math
H(p^t_T) > H(y) \text{ (soft labels have higher entropy)}
```

**Dark Knowledge:** Information in $p^t\_T$ beyond $y$:

```math
\text{Dark Knowledge} = I(X; p^t_T) - I(X; y)
```

### 5. Feature-Based Distillation

**FitNets (Romero et al., 2015):**

```math
\mathcal{L}_{hint} = \|r(F_s^l) - F_t^l\|_2^2
```

Where:
- $F\_s^l, F\_t^l$: Features at layer $l$ (student, teacher)
- $r(\cdot)$: Regressor to match dimensions

**Attention Transfer (Zagoruyko & Komodakis, 2016):**

```math
\mathcal{L}_{AT} = \sum_l \left\|\frac{Q_s^l}{\|Q_s^l\|_2} - \frac{Q_t^l}{\|Q_t^l\|_2}\right\|_p
```

Where attention maps: $Q^l = \sum\_c |F^l\_c|^2$ (sum over channels)

### 6. Relation-Based Distillation

**Relational Knowledge Distillation (RKD):**

*Distance-wise:*

```math
\mathcal{L}_{RKD-D} = \sum_{(i,j)} l_\delta(\psi_D(t_i, t_j), \psi_D(s_i, s_j))
```

Where $\psi\_D(x\_i, x\_j) = \frac{1}{\mu}\|x\_i - x\_j\|\_2$ (normalized distance)

*Angle-wise:*

```math
\mathcal{L}_{RKD-A} = \sum_{(i,j,k)} l_\delta(\psi_A(t_i, t_j, t_k), \psi_A(s_i, s_j, s_k))
```

Where $\psi\_A$ computes angle between vectors.

### 7. Contrastive Distillation

**CRD (Contrastive Representation Distillation):**

```math
\mathcal{L}_{CRD} = -\mathbb{E}_{(x,y^+)} \log \frac{h(T(x), S(y^+))}{h(T(x), S(y^+)) + \sum_{y^-} h(T(x), S(y^-))}
```

Where:
- $y^+$: Positive sample (same class)
- $y^-$: Negative samples
- $h$: Critic function

### 8. Self-Distillation

**Born-Again Networks:**

```math
S^{(k+1)} \leftarrow \text{train}(S^{(k)}, D_{KL}(p^{(k)} \| p^{(k+1)}))
```

Generation $k+1$ learns from generation $k$ with same architecture.

**Theorem:** Self-distillation improves accuracy through regularization and implicit ensemble effects.

---

## 🎯 The Core Idea

```
Teacher (Large):                     Student (Small):
+----------------+                   +----------------+
|                |                   |                |
|   BERT-Large   |   Knowledge       |   DistilBERT   |
|   340M params  | ------------>     |   66M params   |
|   Expert       |                   |   Learns from  |
|                |                   |   teacher      |
+----------------+                   +----------------+

Student learns:
1. Hard labels (ground truth)          → What's correct
2. Soft labels (teacher probs)         → Class relationships
3. Intermediate features (optional)    → Internal representations
```

---

## 💻 Complete Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    """
    Complete Knowledge Distillation Loss
    
    L = α·L_hard + (1-α)·T²·L_soft
    """
    
    def __init__(self, temperature=4.0, alpha=0.7):
        super().__init__()
        self.T = temperature
        self.alpha = alpha
    
    def forward(self, student_logits, teacher_logits, labels):

        # Hard loss (cross-entropy with ground truth)
        hard_loss = F.cross_entropy(student_logits, labels)
        
        # Soft loss (KL divergence with teacher)
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / self.T, dim=1),
            F.softmax(teacher_logits / self.T, dim=1),
            reduction='batchmean'
        ) * (self.T ** 2)
        
        # Combined loss
        return self.alpha * hard_loss + (1 - self.alpha) * soft_loss

class FeatureDistillationLoss(nn.Module):
    """Feature-based distillation (FitNets style)"""
    
    def __init__(self, student_dim, teacher_dim):
        super().__init__()

        # Regressor to match dimensions
        self.regressor = nn.Linear(student_dim, teacher_dim)
    
    def forward(self, student_features, teacher_features):

        # Project student features
        projected = self.regressor(student_features)
        
        # L2 loss
        return F.mse_loss(projected, teacher_features)

class AttentionTransferLoss(nn.Module):
    """Attention map distillation"""
    
    def __init__(self, p=2):
        super().__init__()
        self.p = p
    
    def forward(self, student_activations, teacher_activations):

        # Compute attention maps (spatial sum of squared activations)
        def attention_map(x):

            # x: [B, C, H, W]
            return x.pow(2).sum(dim=1)  # [B, H, W]
        
        s_attn = attention_map(student_activations)
        t_attn = attention_map(teacher_activations)
        
        # Normalize
        s_attn = s_attn / s_attn.norm(p=2, dim=(1,2), keepdim=True)
        t_attn = t_attn / t_attn.norm(p=2, dim=(1,2), keepdim=True)
        
        return (s_attn - t_attn).pow(self.p).mean()

# ========== Training Loop ==========
def train_with_distillation(student, teacher, train_loader, epochs=10):
    """Train student with knowledge distillation"""
    
    teacher.eval()  # Teacher is frozen
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)
    criterion = DistillationLoss(temperature=4.0, alpha=0.5)
    
    for epoch in range(epochs):
        student.train()
        total_loss = 0
        
        for batch in train_loader:
            inputs, labels = batch
            
            # Get teacher predictions (no grad)
            with torch.no_grad():
                teacher_logits = teacher(inputs)
            
            # Get student predictions
            student_logits = student(inputs)
            
            # Compute distillation loss
            loss = criterion(student_logits, teacher_logits, labels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")
    
    return student
```

---

## 📐 Distillation Loss Visualization

<img src="./images/distillation-math.svg" width="100%">

---

## 🌍 Famous Distilled Models

| Teacher | Student | Size Reduction | Accuracy Retention |
|---------|---------|---------------|-------------------|
| **BERT-Base** | DistilBERT | 40% smaller | 97% |
| **GPT-3/4** | Alpaca, Vicuna | 100×+ smaller | Variable |
| **Whisper Large** | Distil-Whisper | 2× smaller | 98% |
| **CLIP** | TinyCLIP | 20× smaller | 93% |
| **Stable Diffusion** | SD Turbo | 4× faster | ~Same |

---

## 📚 References & Resources

### 📄 Key Papers

| Paper | Authors | Year | Key Contribution |
|-------|---------|------|------------------|
| [Distilling Knowledge](https://arxiv.org/abs/1503.02531) | Hinton et al. | 2015 | Original KD (dark knowledge) |
| [FitNets](https://arxiv.org/abs/1412.6550) | Romero et al. | 2015 | Intermediate representations |
| [DistilBERT](https://arxiv.org/abs/1910.01108) | Sanh et al. | 2019 | BERT distillation |
| [TinyBERT](https://arxiv.org/abs/1909.10351) | Jiao et al. | 2019 | Two-stage distillation |
| [MiniLM](https://arxiv.org/abs/2002.10957) | Wang et al. | 2020 | Self-attention distillation |
| [CRD](https://arxiv.org/abs/1910.10699) | Tian et al. | 2019 | Contrastive distillation |
| [RKD](https://arxiv.org/abs/1904.05068) | Park et al. | 2019 | Relational knowledge |

### 🎓 Courses

| Course | Description | Link |
|--------|-------------|------|
| 🔥 MIT 6.5940 | TinyML: Lecture 9 Knowledge Distillation | [Course](https://hanlab.mit.edu/courses/2024-fall-65940) |

---

## 📁 Sub-Topics

| # | Topic | Description | Link |
|:-:|-------|-------------|:----:|
| 1 | **Response Distillation** | Hinton's KD, temperature, soft labels | [📁 Open](./01_response_distillation/README.md) |
| 2 | **Feature Distillation** | FitNets, attention transfer, FSP | [📁 Open](./02_feature_distillation/README.md) |
| 3 | **Self Distillation** | Born-Again, DML, online distillation | [📁 Open](./03_self_distillation/README.md) |

---

⬅️ [Back: Pruning](../03_pruning/README.md) | ➡️ [Next: Weight Sharing](../05_weight_sharing/README.md)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=E74C3C&height=80&section=footer" width="100%"/>
</p>
