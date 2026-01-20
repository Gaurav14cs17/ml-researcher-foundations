<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=120&section=header&text=Residual%20Networks%20(ResNet)&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-06-45B7D1?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 📐 Mathematical Foundations

### The Problem: Degradation

**Observation:** Deeper networks perform *worse* than shallower ones (not just overfitting).

Even if a deeper network could learn identity mappings for extra layers, training fails to find this solution.

### Residual Learning

**Key Insight:** Instead of learning $H(x)$, learn the residual $F(x) = H(x) - x$

**Residual Block:**

```math
y = F(x) + x

```

Where $F(x)$ is the residual function (typically 2-3 conv layers).

**Why it works:**
- If optimal $H(x) = x$ (identity), network only needs $F(x) = 0$

- Learning to output zero is easier than learning identity through multiple nonlinear layers

- Skip connection provides "information highway"

---

## 📐 Gradient Flow Analysis

### Without Skip Connections

For $L$ layers: $h\_L = f\_L(f\_{L-1}(...f\_1(x)))$

```math
\frac{\partial \mathcal{L}}{\partial h_1} = \frac{\partial \mathcal{L}}{\partial h_L} \prod_{l=2}^{L} \frac{\partial h_l}{\partial h_{l-1}}

```

If $\left|\frac{\partial h\_l}{\partial h\_{l-1}}\right| < 1$: vanishing gradients.

### With Skip Connections

```math
h_{l+1} = h_l + F(h_l; W_l)
\frac{\partial h_{l+1}}{\partial h_l} = 1 + \frac{\partial F}{\partial h_l}

```

**Key:** The $+1$ ensures gradient of at least 1 flows backward!

```math
\frac{\partial h_L}{\partial h_1} = \prod_{l=1}^{L-1} \left(1 + \frac{\partial F_l}{\partial h_l}\right)

```

Expanding (ignoring higher-order terms):

```math
\approx 1 + \sum_l \frac{\partial F_l}{\partial h_l} + \text{higher order terms}

```

The $1$ provides a direct gradient path from loss to early layers!

---

## 📐 Residual Block Variants

### Basic Block (ResNet-18/34)

```math
F(x) = W_2 \cdot \sigma(W_1 \cdot x)

```

```
x → Conv3×3 → BN → ReLU → Conv3×3 → BN → (+x) → ReLU → y

```

### Bottleneck Block (ResNet-50/101/152)

```math
F(x) = W_3 \cdot \sigma(W_2 \cdot \sigma(W_1 \cdot x))

```

```
x → Conv1×1 → BN → ReLU → Conv3×3 → BN → ReLU → Conv1×1 → BN → (+x) → ReLU → y

```

**Bottleneck advantage:** 1×1 convs reduce/expand channels cheaply.

For 256 channels:

- Basic: $256 \cdot 3^2 \cdot 256 \times 2 = 1.2M$ params

- Bottleneck (64 intermediate): $256 \cdot 64 + 64 \cdot 3^2 \cdot 64 + 64 \cdot 256 = 70K$ params

### Pre-activation ResNet

```math
y = x + F(\text{BN}(\text{ReLU}(x)))

```

BN and ReLU before convolution. Slightly better gradient flow.

---

## 📐 Dimension Matching

When input/output dimensions differ:

**Option A: Zero Padding**

```math
y = F(x) + \text{pad}(x)

```

**Option B: Projection (1×1 Conv)**

```math
y = F(x) + W_s x

```

Where $W\_s$ is a 1×1 convolution matching dimensions.

---

## 📊 Architecture Comparison

| Model | Layers | Parameters | Top-1 Acc (ImageNet) |
|-------|--------|------------|---------------------|
| **ResNet-18** | 18 | 11.7M | 69.8% |
| **ResNet-34** | 34 | 21.8M | 73.3% |
| **ResNet-50** | 50 | 25.6M | 76.1% |
| **ResNet-101** | 101 | 44.5M | 77.4% |
| **ResNet-152** | 152 | 60.2M | 78.3% |

---

## 💻 Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """Basic residual block for ResNet-18/34"""
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity  # Skip connection!
        out = F.relu(out)
        
        return out

class Bottleneck(nn.Module):
    """Bottleneck block for ResNet-50/101/152"""
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity  # Skip connection!
        out = F.relu(out)
        
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super().__init__()
        self.in_channels = 64
        
        # Stem
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # Head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # Initialize
        self._initialize_weights()
    
    def _make_layer(self, block, out_channels, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        
        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * block.expansion
        
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

# Model constructors
def resnet18(num_classes=1000):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

def resnet34(num_classes=1000):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)

def resnet50(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)

def resnet101(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes)

def resnet152(num_classes=1000):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes)

# Using pretrained from torchvision
import torchvision.models as models
model = models.resnet50(pretrained=True)

```

---

## 📐 Extensions and Variants

### ResNeXt

Aggregated residual transformations:

```math
F(x) = \sum_{i=1}^{C} T_i(x)

```

Where $C$ is cardinality (number of parallel paths).

### DenseNet

Dense connections: each layer receives all preceding features:

```math
x_l = H_l([x_0, x_1, ..., x_{l-1}])

```

### SE-ResNet

Squeeze-and-Excitation: channel attention:

```math
y = F(x) \cdot \sigma(W_2 \text{ReLU}(W_1 \text{GAP}(F(x))))

```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | Deep Residual Learning | [arXiv](https://arxiv.org/abs/1512.03385) |
| 📄 | Identity Mappings in ResNets | [arXiv](https://arxiv.org/abs/1603.05027) |
| 📄 | ResNeXt | [arXiv](https://arxiv.org/abs/1611.05431) |
| 📄 | DenseNet | [arXiv](https://arxiv.org/abs/1608.06993) |
| 🇨🇳 | ResNet详解 | [知乎](https://zhuanlan.zhihu.com/p/31852747) |

---

## 🔗 Where ResNet Is Used

| Application | Usage |
|-------------|-------|
| **Image Classification** | ImageNet, CIFAR |
| **Object Detection** | Backbone for Faster R-CNN, YOLO |
| **Semantic Segmentation** | Backbone for DeepLab, UNet |
| **Feature Extraction** | Transfer learning |
| **Medical Imaging** | Disease classification |

---

⬅️ [Back: MoE](../07_mixture_of_experts/README.md) | ➡️ [Next: RNN](../09_rnn/README.md)

---

⬅️ [Back: Architectures](../../README.md)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=80&section=footer" width="100%"/>
</p>
