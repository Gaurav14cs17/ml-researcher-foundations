<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=120&section=header&text=Data%20Augmentation&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-06-45B7D1?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/data-augmentation-complete.svg" width="100%">

*Caption: Data augmentation creates new training samples through transformations. Image augmentations include flips, rotations, crops. Mixup and CutMix blend multiple images.*

---

## 📐 Mathematical Foundations

### Goal

Expand training set $\mathcal{D}$ with transformed samples:

```math
\mathcal{D}_{aug} = \{(T(x), y) : (x, y) \in \mathcal{D}, T \sim \mathcal{T}\}

```

Where $\mathcal{T}$ is a distribution over transformations.

### Theoretical Justification

Data augmentation is equivalent to adding a prior:

```math
p_{aug}(x|y) = \int p(x|T)p(T|y) dT

```

Or regularizing the loss:

```math
\mathcal{L}_{aug} = \mathbb{E}_{T \sim \mathcal{T}}[\mathcal{L}(f(T(x)), y)]

```

---

## 📐 Image Augmentation

### Geometric Transformations

**Random Crop:**

```math
x_{crop} = x[i:i+h, j:j+w]

```

Where $(i,j)$ is random position, $(h,w)$ is crop size.

**Random Flip:**

```math
x_{flip}[i,j] = x[i, W-1-j]

```

**Rotation:**

```math
\begin{pmatrix} x' \\ y' \end{pmatrix} = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix} \begin{pmatrix} x \\ y \end{pmatrix}

```

### Color Transformations

**Brightness:**

```math
x_{bright} = x + \delta, \quad \delta \sim U(-\beta, \beta)

```

**Contrast:**

```math
x_{contrast} = \alpha(x - \mu) + \mu, \quad \alpha \sim U(1-\gamma, 1+\gamma)

```

**Saturation (HSV):**

```math
S' = S \cdot \alpha, \quad \alpha \sim U(1-\delta, 1+\delta)

```

---

## 📐 Mixup

### Algorithm

For samples $(x_i, y_i)$ and $(x_j, y_j)$:

```math
\tilde{x} = \lambda x_i + (1-\lambda) x_j
\tilde{y} = \lambda y_i + (1-\lambda) y_j

```

Where $\lambda \sim \text{Beta}(\alpha, \alpha)$ (typically $\alpha=0.2$).

### Why It Works

Mixup encourages:

1. **Linear behavior between samples:** Smoother decision boundaries

2. **Regularization:** Reduces overconfidence

3. **Data efficiency:** Creates infinite virtual samples

### Loss with Mixup

```math
\mathcal{L}_{mixup} = \lambda \mathcal{L}(f(\tilde{x}), y_i) + (1-\lambda) \mathcal{L}(f(\tilde{x}), y_j)

```

---

## 📐 CutMix

### Algorithm

1. Sample $\lambda \sim \text{Beta}(\alpha, \alpha)$

2. Sample bounding box $B = (r_x, r_y, r_w, r_h)$ where:

```math
r_w = W\sqrt{1-\lambda}, \quad r_h = H\sqrt{1-\lambda}

```math

3. Combine:

```

\tilde{x} = M \odot x_i + (1-M) \odot x_j

```

Where $M$ is binary mask (1 inside $B$, 0 outside).

### Label

```math
\tilde{y} = \lambda y_i + (1-\lambda) y_j

```

Where $\lambda = 1 - \frac{r_w \cdot r_h}{W \cdot H}$ (fraction of image from $x_i$).

---

## 📐 AutoAugment

### Search Space

Policy = sequence of (operation, probability, magnitude)

Operations: Rotate, Shear, TranslateX, Color, etc.

### Search Algorithm

Use reinforcement learning to find optimal policy:

```math
\pi^* = \arg\max_\pi \mathbb{E}_{T \sim \pi}[\text{Accuracy}(\mathcal{D}_{val})]

```

### RandAugment (Simplified)

No search required:

1. Sample $N$ random transforms

2. Apply with magnitude $M$

```math
T = T_N \circ T_{N-1} \circ ... \circ T_1

```

---

## 📐 Text Augmentation

### Synonym Replacement

Replace random words with synonyms:

```math
x = [w_1, ..., w_i, ..., w_n] \rightarrow [w_1, ..., \text{syn}(w_i), ..., w_n]

```

### Back-Translation

```math
x \xrightarrow{\text{translate}} x_{foreign} \xrightarrow{\text{translate back}} \tilde{x}

```

### EDA (Easy Data Augmentation)

1. Synonym replacement

2. Random insertion

3. Random swap

4. Random deletion

---

## 💻 Implementation

```python
import torch
import torchvision.transforms as T
from torchvision.transforms import RandAugment
import numpy as np

# ============ Standard Augmentations ============

train_transform = T.Compose([
    T.RandomResizedCrop(224, scale=(0.8, 1.0)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(
        brightness=0.4, 
        contrast=0.4, 
        saturation=0.4, 
        hue=0.1
    ),
    T.RandomRotation(15),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225])
])

# ============ RandAugment ============

randaug_transform = T.Compose([
    T.RandomResizedCrop(224),
    T.RandomHorizontalFlip(),
    RandAugment(num_ops=2, magnitude=9),  # N=2, M=9
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ============ Mixup ============

def mixup_data(x, y, alpha=0.2):
    """
    Mixup: Linear interpolation of samples
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Loss for mixup"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# Training with Mixup
for x, y in dataloader:
    x, y_a, y_b, lam = mixup_data(x, y, alpha=0.2)
    
    output = model(x)
    loss = mixup_criterion(criterion, output, y_a, y_b, lam)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# ============ CutMix ============

def cutmix_data(x, y, alpha=1.0):
    """
    CutMix: Cut and paste patches between images
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size, _, H, W = x.size()
    index = torch.randperm(batch_size).to(x.device)
    
    # Sample box
    cut_ratio = np.sqrt(1 - lam)
    cut_w = int(W * cut_ratio)
    cut_h = int(H * cut_ratio)
    
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    # Clamp to image bounds
    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    
    # Apply cutmix
    x_mixed = x.clone()
    x_mixed[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    
    # Adjust lambda for actual box size
    lam = 1 - (x2 - x1) * (y2 - y1) / (W * H)
    
    y_a, y_b = y, y[index]
    
    return x_mixed, y_a, y_b, lam

# ============ Cutout ============

class Cutout:
    """Randomly mask out square regions"""
    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length
    
    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = torch.ones_like(img)
        
        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            
            mask[:, y1:y2, x1:x2] = 0
        
        return img * mask

# ============ Text Augmentation ============

import random

def eda_augment(sentence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1):
    """Easy Data Augmentation for text"""
    words = sentence.split()
    n = len(words)
    
    # Synonym replacement (using simple example)
    n_sr = max(1, int(alpha_sr * n))
    # ... (requires WordNet or similar)
    
    # Random insertion
    n_ri = max(1, int(alpha_ri * n))
    for _ in range(n_ri):
        random_word = random.choice(words)
        words.insert(random.randint(0, len(words)), random_word)
    
    # Random swap
    n_rs = max(1, int(alpha_rs * n))
    for _ in range(n_rs):
        idx1, idx2 = random.sample(range(len(words)), 2)
        words[idx1], words[idx2] = words[idx2], words[idx1]
    
    # Random deletion
    words = [w for w in words if random.random() > p_rd]
    
    return ' '.join(words)

```

---

## 📊 Comparison

| Method | Type | Regularization | Accuracy Boost |
|--------|------|----------------|----------------|
| **Random Crop** | Geometric | Mild | +1-2% |
| **Color Jitter** | Color | Mild | +0.5-1% |
| **Mixup** | Sample mixing | Strong | +1-2% |
| **CutMix** | Sample mixing | Strong | +1-3% |
| **RandAugment** | Auto policy | Strong | +1-2% |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | Mixup Paper | [arXiv](https://arxiv.org/abs/1710.09412) |
| 📄 | CutMix Paper | [arXiv](https://arxiv.org/abs/1905.04899) |
| 📄 | AutoAugment | [arXiv](https://arxiv.org/abs/1805.09501) |
| 📄 | RandAugment | [arXiv](https://arxiv.org/abs/1909.13719) |
| 📄 | EDA Paper | [arXiv](https://arxiv.org/abs/1901.11196) |
| 🇨🇳 | 数据增强详解 | [知乎](https://zhuanlan.zhihu.com/p/41679153) |

---

## 🔗 When to Use

| Augmentation | Best For |
|--------------|----------|
| **Geometric** | Images (always) |
| **Color** | Natural images |
| **Mixup/CutMix** | Classification |
| **RandAugment** | General image tasks |
| **Back-translation** | NLP |

---

⬅️ [Back: Transfer Learning](../07_transfer_learning/README.md) | ➡️ [Next: Self-Supervised Learning](../09_self_supervised/README.md)

---

⬅️ [Back: Main](../README.md)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=80&section=footer" width="100%"/>
</p>
