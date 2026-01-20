<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=E67E22&height=100&section=header&text=Tensor%20Decomposition&fontSize=28&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-08.06.02-E67E22?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

<p align="center">
<img src="./images/tensor.svg" width="100%">
</p>

# Tensor Decomposition

## 📐 Mathematical Theory

### 1. CP Decomposition (CANDECOMP/PARAFAC)

#### 1.1 Definition

**Tensor $\mathcal{T} \in \mathbb{R}^{I\_1 \times I\_2 \times ... \times I\_N}$:**

```math
\mathcal{T} \approx \sum_{r=1}^{R} \lambda_r \cdot a_r^{(1)} \otimes a_r^{(2)} \otimes ... \otimes a_r^{(N)}
```

where:
- $\otimes$ = outer product
- $a\_r^{(n)} \in \mathbb{R}^{I\_n}$ = factor vectors
- $\lambda\_r$ = scalar weights
- $R$ = rank

#### 1.2 Element-wise Form

```math
\mathcal{T}_{i_1, i_2, ..., i_N} \approx \sum_{r=1}^{R} \lambda_r \cdot a_{r,i_1}^{(1)} \cdot a_{r,i_2}^{(2)} \cdot ... \cdot a_{r,i_N}^{(N)}
```

#### 1.3 Parameter Reduction

**Original:** $\prod\_{n=1}^N I\_n$

**CP:** $R \cdot \sum\_{n=1}^N I\_n$

**Example (3D tensor $100 \times 100 \times 100$, R=10):**
- Original: 1,000,000
- CP: 10 × 300 = 3,000
- Compression: 333×!

---

### 2. Tucker Decomposition

#### 2.1 Definition

```math
\mathcal{T} \approx \mathcal{G} \times_1 A^{(1)} \times_2 A^{(2)} \times_3 ... \times_N A^{(N)}
```

where:
- $\mathcal{G} \in \mathbb{R}^{R\_1 \times R\_2 \times ... \times R\_N}$ = core tensor
- $A^{(n)} \in \mathbb{R}^{I\_n \times R\_n}$ = factor matrices
- $\times\_n$ = n-mode product

#### 2.2 N-Mode Product

```math
(T \times_n A)_{i_1...i_{n-1},j,i_{n+1}...i_N} = \sum_{i_n} T_{i_1...i_N} A_{j, i_n}
```

#### 2.3 Parameter Count

**Core:** $\prod\_{n=1}^N R\_n$

**Factors:** $\sum\_{n=1}^N I\_n \cdot R\_n$

**Total:** $\prod\_{n=1}^N R\_n + \sum\_{n=1}^N I\_n \cdot R\_n$

---

### 3. Tensor Train (TT) Decomposition

#### 3.1 Definition

```math
\mathcal{T}_{i_1, i_2, ..., i_N} = G_1[i_1] \cdot G_2[i_2] \cdot ... \cdot G_N[i_N]
```

where:
- $G\_k[i\_k] \in \mathbb{R}^{r\_{k-1} \times r\_k}$ = matrix slice
- $r\_0 = r\_N = 1$ (boundary conditions)
- $r\_1, ..., r\_{N-1}$ = TT-ranks

#### 3.2 Core Tensors

Each core $G\_k \in \mathbb{R}^{r\_{k-1} \times I\_k \times r\_k}$ is a 3D tensor.

#### 3.3 Parameter Count

```math
\text{Params} = \sum_{k=1}^{N} r_{k-1} \cdot I_k \cdot r_k
```

For constant rank $r$:

```math
\text{Params} = O(N \cdot I \cdot r^2)
```

vs. original $O(I^N)$ - exponential reduction!

---

### 4. Application to Neural Networks

#### 4.1 Convolution Weight Tensor

**4D Conv weight:** $W \in \mathbb{R}^{C\_{out} \times C\_{in} \times H \times W}$

**Tucker decomposition:**

```math
W \approx G \times_1 A_{out} \times_2 A_{in} \times_3 A_H \times_4 A_W
```

**Implementation as 4 convolutions:**
1. 1×1 conv: $C\_{in} \to R\_2$
2. $R\_3 \times R\_4$ conv: $R\_2 \to R\_1$
3. 1×1 conv: $R\_1 \to C\_{out}$

#### 4.2 Embedding Matrix

**Large embedding:** $E \in \mathbb{R}^{V \times d}$

**Reshape to tensor:** $\mathcal{E} \in \mathbb{R}^{V\_1 \times V\_2 \times ... \times d}$

where $V = V\_1 \cdot V\_2 \cdot ...$

**Apply TT decomposition for massive compression!**

---

### 5. Implementation

```python
import numpy as np
import torch
import torch.nn as nn
import tensorly as tl
from tensorly.decomposition import parafac, tucker, tensor_train

tl.set_backend('pytorch')

class CPDecomposition:
    """CANDECOMP/PARAFAC decomposition."""
    
    def __init__(self, rank: int):
        self.rank = rank
    
    def decompose(self, tensor: torch.Tensor) -> tuple:
        """
        Decompose tensor into CP format.
        
        Returns:
            weights: [R] scalar weights
            factors: list of [I_n, R] factor matrices
        """
        weights, factors = parafac(tensor, rank=self.rank)
        return weights, factors
    
    def reconstruct(self, weights: torch.Tensor, 
                   factors: list) -> torch.Tensor:
        """Reconstruct tensor from CP format."""
        return tl.cp_to_tensor((weights, factors))
    
    def compression_ratio(self, original_shape: tuple) -> float:
        original = np.prod(original_shape)
        compressed = self.rank * sum(original_shape) + self.rank
        return original / compressed

class TuckerDecomposition:
    """Tucker decomposition with automatic rank selection."""
    
    def __init__(self, ranks: tuple = None, energy_threshold: float = 0.95):
        self.ranks = ranks
        self.energy_threshold = energy_threshold
    
    def decompose(self, tensor: torch.Tensor) -> tuple:
        """
        Decompose tensor into Tucker format.
        
        Returns:
            core: Core tensor
            factors: list of factor matrices
        """
        if self.ranks is None:
            # HOSVD with truncation
            core, factors = tucker(tensor, rank='same')
        else:
            core, factors = tucker(tensor, rank=self.ranks)
        
        return core, factors
    
    def reconstruct(self, core: torch.Tensor, 
                   factors: list) -> torch.Tensor:
        """Reconstruct tensor from Tucker format."""
        return tl.tucker_to_tensor((core, factors))

class TensorTrainLayer(nn.Module):
    """Neural network layer using Tensor Train decomposition."""
    
    def __init__(self, input_shape: tuple, output_shape: tuple,
                 tt_ranks: list, bias: bool = True):
        super().__init__()
        
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.tt_ranks = [1] + list(tt_ranks) + [1]
        
        # Create TT cores
        self.n_dims = len(input_shape)
        self.cores = nn.ParameterList()
        
        for k in range(self.n_dims):
            core = nn.Parameter(torch.randn(
                self.tt_ranks[k],
                input_shape[k] * output_shape[k],
                self.tt_ranks[k + 1]
            ))
            nn.init.xavier_normal_(core)
            self.cores.append(core)
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(np.prod(output_shape)))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with TT contraction.
        
        Args:
            x: [batch, *input_shape]
        """
        batch_size = x.size(0)
        
        # Reshape input
        x = x.view(batch_size, -1)
        
        # Contract with TT cores
        result = self._tt_contract(x)
        
        if self.bias is not None:
            result = result + self.bias
        
        # Reshape output
        result = result.view(batch_size, *self.output_shape)
        
        return result
    
    def _tt_contract(self, x: torch.Tensor) -> torch.Tensor:
        """Contract input with TT weight matrix."""
        batch_size = x.size(0)
        
        # Reconstruct weight matrix (expensive but clear)
        weight = self._reconstruct_weight()
        
        return x @ weight.T
    
    def _reconstruct_weight(self) -> torch.Tensor:
        """Reconstruct full weight matrix from TT cores."""
        # Start with first core
        result = self.cores[0].squeeze(0)  # [I_0 * O_0, r_1]
        
        for k in range(1, self.n_dims):
            core_k = self.cores[k]  # [r_k, I_k * O_k, r_{k+1}]
            
            # Contract: result @ core_k
            result = result @ core_k.view(core_k.size(0), -1)
            result = result.view(-1, core_k.size(2))
        
        # Final squeeze
        result = result.squeeze(-1)
        
        # Reshape to weight matrix
        in_size = np.prod(self.input_shape)
        out_size = np.prod(self.output_shape)
        result = result.view(in_size, out_size).T
        
        return result
    
    def parameter_count(self) -> int:
        """Count parameters in TT format."""
        count = sum(c.numel() for c in self.cores)
        if self.bias is not None:
            count += self.bias.numel()
        return count

class TuckerConv2d(nn.Module):
    """Convolution layer using Tucker decomposition."""
    
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, ranks: tuple,
                 stride: int = 1, padding: int = 0):
        super().__init__()
        
        r_out, r_in, r_h, r_w = ranks
        
        # Factor convolutions
        self.conv1 = nn.Conv2d(in_channels, r_in, 1, bias=False)
        self.conv2 = nn.Conv2d(r_in, r_out, kernel_size, 
                               stride=stride, padding=padding, bias=False)
        self.conv3 = nn.Conv2d(r_out, out_channels, 1, bias=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x
    
    @classmethod
    def from_conv2d(cls, conv: nn.Conv2d, ranks: tuple):
        """Create from existing Conv2d using Tucker decomposition."""
        weight = conv.weight.data  # [C_out, C_in, H, W]
        
        # Tucker decomposition
        core, factors = tucker(weight, rank=ranks)
        
        layer = cls(
            conv.in_channels, conv.out_channels,
            conv.kernel_size[0], ranks,
            conv.stride[0], conv.padding[0]
        )
        
        # Set weights from decomposition
        # This is approximate - would need more careful implementation
        
        return layer

def estimate_tt_ranks(tensor: torch.Tensor, energy_threshold: float = 0.95) -> list:
    """Estimate good TT ranks using SVD."""
    shape = tensor.shape
    n_dims = len(shape)
    ranks = []
    
    # Unfold and compute SVD for each mode
    tensor_copy = tensor.clone()
    
    for k in range(n_dims - 1):
        # Reshape to matrix
        left_size = int(np.prod(shape[:k+1]))
        right_size = int(np.prod(shape[k+1:]))
        matrix = tensor_copy.reshape(left_size, right_size)
        
        # SVD
        U, S, Vt = torch.linalg.svd(matrix, full_matrices=False)
        
        # Select rank based on energy
        total_energy = (S ** 2).sum()
        cumsum = torch.cumsum(S ** 2, dim=0)
        rank = (cumsum / total_energy < energy_threshold).sum().item() + 1
        rank = max(1, min(rank, min(left_size, right_size)))
        
        ranks.append(rank)
    
    return ranks
```

---

### 6. Comparison

| Method | Parameters | Computation | Flexibility |
|--------|------------|-------------|-------------|
| **CP** | $R \sum I\_n$ | $O(R \prod I\_n)$ | Low |
| **Tucker** | $\prod R\_n + \sum I\_n R\_n$ | $O(\prod R\_n)$ | High |
| **TT** | $\sum r^2 I\_n$ | $O(r^2 \sum I\_n)$ | Medium |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | Tensorizing Neural Networks | [arXiv](https://arxiv.org/abs/1509.06569) |
| 📄 | Compression of CNNs | [arXiv](https://arxiv.org/abs/1412.6553) |
| 📄 | Tensor Decomposition Survey | [arXiv](https://arxiv.org/abs/1711.10781) |
| 🇨🇳 | 张量分解入门 | [知乎](https://zhuanlan.zhihu.com/p/24798389) |
| 🇨🇳 | CP/Tucker分解详解 | [CSDN](https://blog.csdn.net/weixin_41965898/article/details/104584320) |
| 🇨🇳 | Tensor Train压缩 | [B站](https://www.bilibili.com/video/BV1mK4y1P7rP) |

---

⬅️ [Back: SVD Compression](../01_svd_compression/README.md) | ➡️ [Next: Depthwise Separable](../03_depthwise_separable/README.md)

