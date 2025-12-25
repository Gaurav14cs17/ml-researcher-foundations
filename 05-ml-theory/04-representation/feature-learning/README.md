<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=120&section=header&text=Feature%20Learning&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-05-4ECDC4?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/feature-learning.svg" width="100%">

*Caption: Deep networks learn hierarchical features automatically: edges → textures → parts → objects. This replaced hand-crafted features (SIFT, HOG) that required domain expertise and couldn't adapt to new tasks.*

---

## 📂 Overview

Feature learning is the ability of deep networks to automatically discover useful representations from raw data. This was a key breakthrough that enabled modern deep learning success.

---

## 📐 Mathematical Framework

### Representation Learning Objective

```
Learn function h: X → Z such that:
    - Z = h(x) captures meaningful structure
    - Downstream tasks are easier in Z space
    
Formally:
    h* = argmin_h E_x[L(f ∘ h(x), y)]
    
Where h is the feature extractor, f is the task head
```

### Information Bottleneck View

```
Feature learning as information trade-off:

    max I(Z; Y) - β I(Z; X)
    
Where:
    I(Z; Y) = information about labels (task-relevant)
    I(Z; X) = information from input (compression)
    
Good features: High I(Z;Y), Low I(Z;X) = discard noise
```

### Hierarchical Feature Decomposition

```
Image: x ∈ ℝ^(H×W×3)

Layer 1: h₁(x) = σ(W₁ * x)       → Edges, colors
Layer 2: h₂(h₁) = σ(W₂ * h₁)    → Textures, patterns  
Layer 3: h₃(h₂) = σ(W₃ * h₂)    → Parts, shapes
Layer N: hₙ(hₙ₋₁) = σ(Wₙ * hₙ₋₁) → Objects, concepts

Each layer builds on the previous, creating abstraction hierarchy
```

---

## 🔑 Key Concepts

| Concept | Description | Example |
|---------|-------------|---------|
| **Hierarchical Features** | Low → high level abstractions | Edges → Objects |
| **Transfer Learning** | Reuse features for new tasks | ImageNet → Medical |
| **Disentanglement** | Separate independent factors | Position vs. Identity |
| **Invariance** | Same output for transformed inputs | Rotation invariance |
| **Equivariance** | Output transforms with input | Translation equivariance |

---

## 📊 Evolution of Feature Engineering

```
Before Deep Learning (Manual Features):
+----------------------------------------------+
|  Image → SIFT/HOG/SURF → Classifier → Output |
|           Hand-crafted                        |
|           (2000s)                             |
+----------------------------------------------+

With Deep Learning (Learned Features):
+----------------------------------------------+
|  Image → [CNN Layers] → Classifier → Output  |
|           Learned                             |
|           (2012+)                             |
+----------------------------------------------+
```

---

## 💻 Code Examples

### Extracting Learned Features

```python
import torch
import torchvision.models as models

# Load pretrained model
resnet = models.resnet50(pretrained=True)

# Remove classification head to get features
feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])

# Extract features
def get_features(image):
    """Extract 2048-dim features from image"""
    with torch.no_grad():
        features = feature_extractor(image)
    return features.squeeze()  # Shape: (2048,)

# Features for transfer learning
features = get_features(my_image)
```

### Visualizing Learned Features

```python
import matplotlib.pyplot as plt
import torch

def visualize_first_layer_filters(model):
    """Visualize what the first conv layer learned"""
    # Get first conv layer weights
    filters = model.conv1.weight.data.clone()
    
    # Normalize for visualization
    filters = (filters - filters.min()) / (filters.max() - filters.min())
    
    # Plot
    fig, axes = plt.subplots(8, 8, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        if i < filters.shape[0]:
            # Show RGB filter as image
            ax.imshow(filters[i].permute(1, 2, 0).cpu())
        ax.axis('off')
    plt.suptitle('Learned First Layer Filters (Edge Detectors)')
    plt.show()
```

### Self-Supervised Feature Learning

```python
import torch.nn as nn

class ContrastiveLearning(nn.Module):
    """SimCLR-style self-supervised features"""
    
    def __init__(self, encoder, projection_dim=128):
        super().__init__()
        self.encoder = encoder
        self.projector = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )
    
    def forward(self, x1, x2):
        """x1, x2 are different augmentations of same image"""
        h1 = self.encoder(x1)  # Learned features
        h2 = self.encoder(x2)
        
        z1 = self.projector(h1)  # Projection for contrastive loss
        z2 = self.projector(h2)
        
        return z1, z2
    
    def get_features(self, x):
        """Use encoder output as features (not projection)"""
        return self.encoder(x)
```

---

## 🔗 Connection to Other Topics

```
Feature Learning
    +-- Transfer Learning (reuse features)
    +-- Embeddings (discrete → continuous)
    +-- Self-Supervised Learning (labels from data)
    |   +-- Contrastive (SimCLR, MoCo)
    |   +-- Masked (BERT, MAE)
    +-- Autoencoders (reconstruction objective)
    +-- Disentanglement (VAE, β-VAE)
```

---

## 📊 What Features Look Like

| Layer | CNN Features | Transformer Features |
|-------|--------------|---------------------|
| Early | Edges, colors | Token embeddings |
| Middle | Textures, patterns | Syntax, local context |
| Late | Parts, objects | Semantics, concepts |
| Final | Categories | Task-specific |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | Embeddings | [../embeddings/](../embeddings/) |
| 📖 | Transfer Learning | [../transfer/](../transfer/) |
| 🎥 | 3Blue1Brown: Neural Networks | [YouTube](https://www.youtube.com/watch?v=aircAruvnKk) |
| 📄 | AlexNet Paper | [NeurIPS](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) |
| 🇨🇳 | 深度学习特征学习详解 | [知乎](https://zhuanlan.zhihu.com/p/27642620) |
| 🇨🇳 | 特征提取与迁移学习 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88956438) |
| 🇨🇳 | CNN特征可视化 | [B站](https://www.bilibili.com/video/BV1Lq4y1k7j6) |
| 🇨🇳 | 自监督表示学习综述 | [机器之心](https://www.jiqizhixin.com/articles/2020-09-22-11)


## 🔗 Where This Topic Is Used

| Application | Usage |
|-------------|-------|
| **Machine Learning** | Core concept for ML systems |
| **Deep Learning** | Foundation for neural networks |
| **Research** | Important for understanding papers |

---

⬅️ [Back: Representation](../)

---

⬅️ [Back: Embeddings](../embeddings/) | ➡️ [Next: Transfer](../transfer/)

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=80&section=footer" width="100%"/>
</p>
