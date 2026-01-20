<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=120&section=header&text=Convolutional%20Neural%20Networks&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-06-45B7D1?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 📂 Overview

Convolutional Neural Networks (CNNs) are specialized neural networks for processing grid-like data (images, time series). They exploit spatial locality and translation invariance through convolution operations.

---

## 📐 Convolution Operation

### 2D Convolution

For input \(I\) and kernel \(K\):

```math
(I * K)[i,j] = \sum_{m=0}^{k_h-1} \sum_{n=0}^{k_w-1} I[i+m, j+n] \cdot K[m, n]
```

**With multiple channels (cross-correlation):**
```math
Y[c_{out}, i, j] = \sum_{c_{in}=0}^{C_{in}-1} \sum_{m=0}^{k_h-1} \sum_{n=0}^{k_w-1} X[c_{in}, i+m, j+n] \cdot W[c_{out}, c_{in}, m, n] + b[c_{out}]
```

### Output Size Formula

```math
H_{out} = \left\lfloor \frac{H_{in} + 2P - K}{S} \right\rfloor + 1
```

where:
- \(H_{in}\): input height
- \(P\): padding
- \(K\): kernel size
- \(S\): stride

### Parameter Count

For Conv2d(C_in, C_out, K×K):
```math
\text{Parameters} = C_{out} \times (C_{in} \times K \times K + 1)
```

The "+1" is for the bias per output channel.

---

## 🔬 Key Properties

### 1. Local Connectivity

Each output depends only on a local region (receptive field):
```
Unlike fully connected:
FC: y = Wx + b  (each output depends on ALL inputs)
    Parameters: H_in × W_in × C_in × H_out × W_out × C_out

Conv: Each output depends on K×K×C_in inputs
    Parameters: C_out × C_in × K × K

For 224×224×3 input with 64 output channels:
FC: 224 × 224 × 3 × 224 × 224 × 64 ≈ 483 billion!
Conv (3×3): 64 × 3 × 3 × 3 = 1,728

Massive parameter reduction!
```

### 2. Weight Sharing

Same kernel applied at all positions:
```
Traditional: Different weights for each position
Conv: Same weights everywhere

Benefits:
- Fewer parameters
- Translation equivariance
- Better generalization
```

### 3. Translation Equivariance

If input shifts, output shifts by the same amount:
```math
f(T_x \cdot I) = T_x \cdot f(I)
```

**Proof:**
```
Let I' = I shifted by (a, b)
I'[i, j] = I[i-a, j-b]

(I' * K)[i, j] = Σ_m Σ_n I'[i+m, j+n] · K[m, n]
               = Σ_m Σ_n I[i+m-a, j+n-b] · K[m, n]
               = (I * K)[i-a, j-b]

Output is shifted by same (a, b)! ✓
```

---

## 📊 Receptive Field

### Definition

The receptive field is the region of input that affects a single output neuron.

### Computing Receptive Field

For a stack of convolutions:
```math
R_{out} = R_{in} + (K - 1) \times \prod_{i=1}^{l-1} S_i
```

**Example (VGG-style):**
```
Layer 1: Conv 3×3, stride 1 → RF = 3
Layer 2: Conv 3×3, stride 1 → RF = 3 + (3-1)×1 = 5
Layer 3: MaxPool 2×2        → RF = 5 + (2-1)×1 = 6
Layer 4: Conv 3×3, stride 1 → RF = 6 + (3-1)×2 = 10
...

After 5 Conv + Pool layers: RF ≈ 180 pixels!
```

### Why Deep > Wide?

```
Two 3×3 layers: RF = 5×5, Params = 2 × 3² × C² = 18C²
One 5×5 layer:  RF = 5×5, Params = 5² × C² = 25C²

Same receptive field, fewer parameters, more nonlinearity!
```

---

## 📐 Common Layers

### 1. Pooling

**Max Pooling:**
```math
Y[i,j] = \max_{m,n \in \text{window}} X[i \cdot s + m, j \cdot s + n]
```

**Average Pooling:**
```math
Y[i,j] = \frac{1}{k^2} \sum_{m,n \in \text{window}} X[i \cdot s + m, j \cdot s + n]
```

**Global Average Pooling (GAP):**
```math
Y[c] = \frac{1}{H \times W} \sum_{i,j} X[c, i, j]
```

### 2. Strided Convolution

Alternative to pooling - use stride > 1:
```
Benefits:
- Learnable downsampling (vs fixed pooling)
- More computation efficient
- Used in ResNet (stride-2 conv instead of pool)
```

### 3. Dilated (Atrous) Convolution

Insert zeros between kernel elements:
```math
Y[i,j] = \sum_{m,n} X[i + r \cdot m, j + r \cdot n] \cdot K[m,n]
```

where \(r\) is the dilation rate.

**Benefit:** Larger receptive field without more parameters.

---

## 💻 Complete Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2dFromScratch(nn.Module):
    """
    2D Convolution implemented from scratch
    """
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        
        # Initialize weights (He initialization)
        k = 1 / (in_channels * self.kernel_size[0] * self.kernel_size[1])
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, *self.kernel_size) * (k ** 0.5)
        )
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x):
        """
        Args:
            x: (batch, in_channels, H, W)
        
        Returns:
            output: (batch, out_channels, H_out, W_out)
        """
        batch_size, _, H, W = x.shape
        kh, kw = self.kernel_size
        
        # Add padding
        if self.padding > 0:
            x = F.pad(x, [self.padding] * 4)
        
        # Compute output size
        H_out = (H + 2 * self.padding - kh) // self.stride + 1
        W_out = (W + 2 * self.padding - kw) // self.stride + 1
        
        # Unfold input into patches
        # This creates a tensor of all (kh × kw) patches
        patches = x.unfold(2, kh, self.stride).unfold(3, kw, self.stride)
        # Shape: (batch, in_channels, H_out, W_out, kh, kw)
        
        patches = patches.contiguous().view(batch_size, self.in_channels, H_out, W_out, -1)
        # Shape: (batch, in_channels, H_out, W_out, kh*kw)
        
        weight = self.weight.view(self.out_channels, self.in_channels, -1)
        # Shape: (out_channels, in_channels, kh*kw)
        
        # Compute convolution via einsum
        output = torch.einsum('bihwk,oik->bohw', patches, weight)
        
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)
        
        return output

class ResidualBlock(nn.Module):
    """
    Basic Residual Block
    
    x → Conv → BN → ReLU → Conv → BN → + → ReLU
    ↑_________________________________↓
    """
    def __init__(self, channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        return F.relu(out)

class BottleneckBlock(nn.Module):
    """
    Bottleneck Residual Block (used in ResNet-50+)
    
    1×1 → 3×3 → 1×1 (expansion)
    """
    expansion = 4
    
    def __init__(self, in_channels, channels, stride=1, downsample=None):
        super().__init__()
        # Bottleneck reduces channels, processes, then expands
        self.conv1 = nn.Conv2d(in_channels, channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, channels * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels * self.expansion)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        return F.relu(out)

class SimpleCNN(nn.Module):
    """
    Simple CNN for classification
    """
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        
        # Feature extractor
        self.features = nn.Sequential(
            # Block 1: 32 → 16
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2: 16 → 8
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3: 8 → 4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

class MiniResNet(nn.Module):
    """
    Small ResNet-style network
    """
    def __init__(self, num_classes=10, channels=[64, 128, 256, 512]):
        super().__init__()
        
        # Initial conv
        self.conv1 = nn.Conv2d(3, channels[0], 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(channels[0], channels[0], 2)
        self.layer2 = self._make_layer(channels[0], channels[1], 2, stride=2)
        self.layer3 = self._make_layer(channels[1], channels[2], 2, stride=2)
        self.layer4 = self._make_layer(channels[2], channels[3], 2, stride=2)
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels[3], num_classes)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        layers = [ResidualBlock(in_channels, stride, downsample)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Visualization: Feature maps
def visualize_features(model, image):
    """
    Extract and visualize intermediate feature maps
    """
    import matplotlib.pyplot as plt
    
    activations = []
    
    def hook_fn(module, input, output):
        activations.append(output.detach())
    
    # Register hooks on conv layers
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(hook_fn))
    
    # Forward pass
    with torch.no_grad():
        _ = model(image.unsqueeze(0))
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Plot
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for idx, (ax, act) in enumerate(zip(axes.flat, activations[:8])):
        # Show first channel of each layer
        ax.imshow(act[0, 0].cpu().numpy(), cmap='viridis')
        ax.set_title(f'Layer {idx+1}')
        ax.axis('off')
    
    plt.tight_layout()
    return fig

# Example usage
model = SimpleCNN(in_channels=3, num_classes=10)
x = torch.randn(4, 3, 32, 32)
y = model(x)
print(f"Input: {x.shape}")
print(f"Output: {y.shape}")

# Parameter count
n_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n_params:,}")
```

---

## 📊 Architecture Evolution

| Model | Year | Key Innovation | Top-1 Error |
|-------|------|----------------|-------------|
| **LeNet-5** | 1998 | First CNN | - |
| **AlexNet** | 2012 | Deep, ReLU, Dropout, GPU | 16.4% |
| **VGG-16** | 2014 | 3×3 convolutions | 7.3% |
| **GoogLeNet** | 2014 | Inception modules | 6.7% |
| **ResNet-152** | 2015 | Skip connections | 3.6% |
| **DenseNet** | 2017 | Dense connections | 3.5% |
| **EfficientNet** | 2019 | Compound scaling | 2.9% |

---

## 📚 Key Insights

| Insight | Details |
|---------|---------|
| **Small kernels** | 3×3 is optimal (VGG finding) |
| **Depth matters** | Skip connections enable very deep nets |
| **BN placement** | Before ReLU for Pre-LN, after for Post-LN |
| **Stride vs Pool** | Strided conv more expressive than pooling |

---

## 📚 References

| Type | Resource | Link |
|------|----------|------|
| 📄 | AlexNet | [Krizhevsky et al., 2012](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) |
| 📄 | VGGNet | [Simonyan & Zisserman, 2014](https://arxiv.org/abs/1409.1556) |
| 📄 | ResNet | [He et al., 2016](https://arxiv.org/abs/1512.03385) |
| 📄 | EfficientNet | [Tan & Le, 2019](https://arxiv.org/abs/1905.11946) |

---

## 🗺️ Navigation

| ⬅️ Previous | 🏠 Home | ➡️ Next |
|:-----------:|:-------:|:-------:|
| [Attention](../01_attention/README.md) | [Architectures](../README.md) | [Diffusion](../03_diffusion/README.md) |

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=80&section=footer" width="100%"/>
</p>
