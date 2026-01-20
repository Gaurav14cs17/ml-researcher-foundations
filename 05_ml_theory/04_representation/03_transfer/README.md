<!-- Navigation -->
<p align="center">
  <a href="../02_feature_learning/">⬅️ Prev: Feature Learning</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../">📚 Representation</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../../05_risk_minimization/">Next: Risk Minimization ➡️</a>
</p>

---

<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=120&section=header&text=Transfer%20Learning&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-05-4ECDC4?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/transfer.svg" width="100%">

*Caption: Transfer learning leverages knowledge from source tasks to improve target task performance.*

---

## 📂 Overview

**Transfer learning** reuses learned representations from source tasks for target tasks, dramatically reducing data requirements and training time. It is the foundation of modern foundation models.

---

## 📐 Mathematical Framework

### Problem Setting

**Source domain:** \(\mathcal{D}_S = \{(x_i^s, y_i^s)\}_{i=1}^{n_s}\) from \(P_S(X, Y)\)

**Target domain:** \(\mathcal{D}_T = \{(x_i^t, y_i^t)\}_{i=1}^{n_t}\) from \(P_T(X, Y)\)

**Goal:** Learn \(f_T: X \to Y\) using both \(\mathcal{D}_S\) and \(\mathcal{D}_T\).

### Types of Transfer

| Setting | \(P_S(X)\) | \(P_S(Y|X)\) | \(P_T(X)\) | \(P_T(Y|X)\) |
|---------|-----------|-------------|-----------|-------------|
| Covariate Shift | ≠ | = | - | - |
| Label Shift | = | ≠ | - | - |
| Domain Adaptation | ≠ | = (approx) | - | - |

---

## 📐 Feature Transfer

### Pre-training + Fine-tuning

**Stage 1 (Pre-training):** Learn encoder \(\phi_\theta\) on source task:
```math
\theta^* = \arg\min_\theta \mathcal{L}_S(\phi_\theta; \mathcal{D}_S)
```

**Stage 2 (Fine-tuning):** Adapt to target with small learning rate:
```math
\theta^{**} = \arg\min_\theta \mathcal{L}_T(\phi_\theta; \mathcal{D}_T)
```

starting from \(\theta = \theta^*\).

### Why Transfer Works

**Hypothesis:** Features learned on source task are useful for target task.

**Theorem (Ben-David et al.):** For domain adaptation:
```math
\epsilon_T(h) \leq \epsilon_S(h) + d_{\mathcal{H}}(S, T) + \lambda
```

where:
- \(\epsilon_T, \epsilon_S\) are target/source errors
- \(d_{\mathcal{H}}(S, T)\) is the \(\mathcal{H}\)-divergence between domains
- \(\lambda\) is the optimal joint error

---

## 📐 Low-Rank Adaptation (LoRA)

### Key Idea

Instead of updating all parameters, learn low-rank updates:

```math
W' = W + \Delta W = W + BA
```

where \(B \in \mathbb{R}^{d \times r}\), \(A \in \mathbb{R}^{r \times k}\), and \(r \ll \min(d, k)\).

### Parameter Efficiency

| Method | Trainable Params | Memory |
|--------|-----------------|--------|
| Full Fine-tuning | 100% | High |
| LoRA (r=16) | ~0.1% | Low |
| Adapters | ~1-5% | Medium |
| Prompt Tuning | ~0.01% | Very Low |

---

## 💻 Code Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation layer.
    
    W' = W + BA where B ∈ ℝ^{d×r}, A ∈ ℝ^{r×k}
    """
    
    def __init__(self, original_layer, rank=16, alpha=32):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        
        # Freeze original weights
        for param in self.original_layer.parameters():
            param.requires_grad = False
        
        # Initialize low-rank matrices
        d, k = original_layer.weight.shape
        self.lora_A = nn.Parameter(torch.randn(rank, k) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(d, rank))
    
    def forward(self, x):
        # Original output
        original_output = self.original_layer(x)
        
        # LoRA contribution: x @ A^T @ B^T * (alpha/rank)
        lora_output = F.linear(F.linear(x, self.lora_A), self.lora_B)
        lora_output = lora_output * (self.alpha / self.rank)
        
        return original_output + lora_output

class AdapterLayer(nn.Module):
    """
    Adapter module for transfer learning.
    
    Insert between transformer layers:
    h' = h + Adapter(h)
    Adapter(h) = W_up(GELU(W_down(h)))
    """
    
    def __init__(self, hidden_size, adapter_size=64):
        super().__init__()
        self.down_proj = nn.Linear(hidden_size, adapter_size)
        self.up_proj = nn.Linear(adapter_size, hidden_size)
        
        # Initialize for near-identity at start
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)
    
    def forward(self, x):
        down = F.gelu(self.down_proj(x))
        up = self.up_proj(down)
        return x + up

class TransferLearningModel(nn.Module):
    """
    General transfer learning framework.
    """
    
    def __init__(self, backbone, num_classes, freeze_backbone=True):
        super().__init__()
        self.backbone = backbone
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Get backbone output dimension
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            out = self.backbone(dummy)
            if isinstance(out, tuple):
                out = out[0]
            feature_dim = out.shape[-1]
        
        self.classifier = nn.Linear(feature_dim, num_classes)
    
    def forward(self, x):
        features = self.backbone(x)
        if isinstance(features, tuple):
            features = features[0]
        return self.classifier(features)
    
    def unfreeze_layers(self, num_layers):
        """Gradually unfreeze layers for fine-tuning."""
        layers = list(self.backbone.children())
        for layer in layers[-num_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

def create_transfer_model(pretrained_name, num_classes, strategy='feature_extraction'):
    """
    Create a transfer learning model.
    
    Args:
        pretrained_name: Name of pretrained model
        num_classes: Number of target classes
        strategy: 'feature_extraction' or 'fine_tuning'
    """
    import torchvision.models as models
    
    # Load pretrained model
    if pretrained_name == 'resnet50':
        backbone = models.resnet50(pretrained=True)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
    elif pretrained_name == 'vit_b_16':
        backbone = models.vit_b_16(pretrained=True)
        feature_dim = backbone.heads.head.in_features
        backbone.heads.head = nn.Identity()
    else:
        raise ValueError(f"Unknown model: {pretrained_name}")
    
    # Freeze based on strategy
    if strategy == 'feature_extraction':
        for param in backbone.parameters():
            param.requires_grad = False
    elif strategy == 'fine_tuning':
        # Freeze early layers
        for name, param in backbone.named_parameters():
            if 'layer4' not in name and 'encoder.layers.11' not in name:
                param.requires_grad = False
    
    # Add classifier
    model = nn.Sequential(
        backbone,
        nn.Linear(feature_dim, num_classes)
    )
    
    return model

# Domain adaptation with gradient reversal
class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None

class DomainAdaptationModel(nn.Module):
    """
    Domain Adversarial Neural Network (DANN).
    
    Learn domain-invariant features via adversarial training.
    """
    
    def __init__(self, feature_extractor, classifier, domain_classifier, alpha=1.0):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.domain_classifier = domain_classifier
        self.alpha = alpha
    
    def forward(self, x, domain_labels=None):
        features = self.feature_extractor(x)
        class_output = self.classifier(features)
        
        if domain_labels is not None:
            # Gradient reversal for domain classifier
            reversed_features = GradientReversal.apply(features, self.alpha)
            domain_output = self.domain_classifier(reversed_features)
            return class_output, domain_output
        
        return class_output
```

---

## 📊 Transfer Learning Strategies

| Strategy | When to Use | Data Required |
|----------|-------------|---------------|
| Feature Extraction | Small target data | Very little |
| Fine-tune last layers | Medium target data | Medium |
| Full fine-tuning | Large target data | Large |
| Domain Adaptation | Different distributions | Unlabeled target |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | LoRA | [Hu et al.](https://arxiv.org/abs/2106.09685) |
| 📄 | Foundation Models | [Bommasani et al.](https://arxiv.org/abs/2108.07258) |
| 📄 | Domain Adaptation Theory | [Ben-David et al.](https://www.alexkulesza.com/pubs/adapt_mlj10.pdf) |

---

⬅️ [Back: Feature Learning](../02_feature_learning/) | ➡️ [Back: Representation](../)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<!-- Navigation -->
<p align="center">
  <a href="../02_feature_learning/">⬅️ Prev: Feature Learning</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../">📚 Representation</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../../05_risk_minimization/">Next: Risk Minimization ➡️</a>
</p>

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=80&section=footer" width="100%"/>
</p>
