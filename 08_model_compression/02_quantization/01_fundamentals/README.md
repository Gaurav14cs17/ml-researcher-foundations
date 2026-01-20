<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=3498DB&height=100&section=header&text=Quantization%20Fundamentals&fontSize=28&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-08.02.01-3498DB?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

<p align="center">
<img src="./images/fundamentals.svg" width="100%">
</p>

## 📐 Mathematical Theory

### 1. Definition of Quantization

**Quantization** is a mapping $Q: \mathbb{R} \to \mathcal{Q}$ where $\mathcal{Q}$ is a finite set of discrete values.

**General Form:**

```math
Q(x) = \Delta \cdot \text{round}\left(\frac{x}{\Delta}\right)

```

where $\Delta$ is the quantization step size.

---

### 2. Uniform Quantization

#### 2.1 Symmetric Quantization

**Definition:** For $b$-bit signed quantization with symmetric range $[-\alpha, \alpha]$:

```math
Q_{sym}(x) = \text{clamp}\left(\text{round}\left(\frac{x}{s}\right), -2^{b-1}, 2^{b-1}-1\right) \cdot s

```

where the scale factor is:

```math
s = \frac{\alpha}{2^{b-1} - 1}

```

**Properties:**
- Zero is exactly representable

- Symmetric around zero

- Range: $[-\alpha, \alpha]$

#### 2.2 Asymmetric (Affine) Quantization

**Definition:** For range $[\beta, \alpha]$:

```math
Q_{asym}(x) = \text{clamp}\left(\text{round}\left(\frac{x - z}{s}\right), 0, 2^b - 1\right) \cdot s + z

```

where:

```math
s = \frac{\alpha - \beta}{2^b - 1}, \quad z = \beta

```

**Properties:**
- Can represent asymmetric distributions

- Zero-point $z$ may not be exactly representable

- More complex computation

---

### 3. Quantization Error Analysis

#### 3.1 Error Model

**Theorem (Quantization Noise Model):**
For uniform quantization with step size $\Delta$, the quantization error $\epsilon = x - Q(x)$ follows approximately:

```math
\epsilon \sim \text{Uniform}\left(-\frac{\Delta}{2}, \frac{\Delta}{2}\right)

```

**Proof:**
Let $x = k\Delta + r$ where $k \in \mathbb{Z}$ and $r \in [-\Delta/2, \Delta/2]$.

Then $Q(x) = k\Delta$, so:

```math
\epsilon = x - Q(x) = r

```

If $x$ is uniformly distributed within each quantization interval, then $r \sim \text{Uniform}(-\Delta/2, \Delta/2)$.

#### 3.2 Mean Squared Error

**Theorem:** The MSE of uniform quantization is:

```math
\text{MSE} = \mathbb{E}[\epsilon^2] = \frac{\Delta^2}{12}

```

**Proof:**

```math
\text{MSE} = \int_{-\Delta/2}^{\Delta/2} \epsilon^2 \cdot \frac{1}{\Delta} d\epsilon = \frac{1}{\Delta}\left[\frac{\epsilon^3}{3}\right]_{-\Delta/2}^{\Delta/2}
= \frac{1}{\Delta} \cdot \frac{2}{3} \cdot \frac{\Delta^3}{8} = \frac{\Delta^2}{12}

```

#### 3.3 Signal-to-Quantization-Noise Ratio (SQNR)

**Definition:**

```math
\text{SQNR} = \frac{\sigma_x^2}{\sigma_\epsilon^2} = \frac{\sigma_x^2}{\Delta^2/12} = \frac{12\sigma_x^2}{\Delta^2}

```

**In decibels:**

```math
\text{SQNR}_{dB} = 10\log_{10}\left(\frac{12\sigma_x^2}{\Delta^2}\right)

```

**For full-range quantization** with $\Delta = \frac{2\alpha}{2^b}$ and $\sigma\_x^2 \approx \alpha^2/3$:

```math
\text{SQNR}_{dB} \approx 6.02b + 4.77 \text{ dB}

```

**Proof:**

```math
\text{SQNR}_{dB} = 10\log_{10}\left(\frac{12 \cdot \alpha^2/3}{(2\alpha/2^b)^2}\right) = 10\log_{10}\left(\frac{4\alpha^2 \cdot 2^{2b}}{4\alpha^2}\right)
= 10\log_{10}(2^{2b}) = 20b \cdot \log_{10}(2) \approx 6.02b

```

The +4.77 dB comes from the factor of 12/3 = 4.

---

### 3.4 Bit-Width and Error Relationship

**Theorem (Bit-Width Scaling):** For $b$-bit uniform quantization over range $[-\alpha, \alpha]$:

```math
\text{MSE}(b) = \frac{\alpha^2}{3 \cdot 4^b}

```

**Proof:**

Step 1: The step size is:

```math
\Delta = \frac{2\alpha}{2^b} = \frac{\alpha}{2^{b-1}}

```

Step 2: Substituting into the MSE formula:

```math
\text{MSE} = \frac{\Delta^2}{12} = \frac{1}{12} \cdot \frac{\alpha^2}{4^{b-1}} = \frac{\alpha^2}{3 \cdot 4^b}

```

**Corollary:** Each additional bit reduces MSE by factor of 4 (6 dB improvement).

```math
\frac{\text{MSE}(b)}{\text{MSE}(b+1)} = \frac{\alpha^2 / (3 \cdot 4^b)}{\alpha^2 / (3 \cdot 4^{b+1})} = 4

```

---

### 3.5 Information-Theoretic Lower Bound

**Theorem (Rate-Distortion for Gaussian):**
For $X \sim \mathcal{N}(0, \sigma^2)$ with MSE distortion $D$:

```math
R(D) = \frac{1}{2}\log_2\left(\frac{\sigma^2}{D}\right) \text{ bits/sample}

```

**Proof:**

The rate-distortion function is defined as:

```math
R(D) = \min_{p(\hat{X}|X): \mathbb{E}[(X-\hat{X})^2] \leq D} I(X; \hat{X})

```

For Gaussian source with MSE distortion, the optimal test channel is:

```math
\hat{X} = X + N, \quad N \sim \mathcal{N}(0, D) \text{ independent of } X

```

The mutual information:

```math
I(X; \hat{X}) = h(\hat{X}) - h(\hat{X}|X) = h(\hat{X}) - h(N)

```

Since $\hat{X} = X + N$ where both are Gaussian:

```math
\text{Var}(\hat{X}) = \sigma^2 + D
I(X; \hat{X}) = \frac{1}{2}\log_2(2\pi e(\sigma^2 + D)) - \frac{1}{2}\log_2(2\pi e D)
= \frac{1}{2}\log_2\left(\frac{\sigma^2 + D}{D}\right) = \frac{1}{2}\log_2\left(1 + \frac{\sigma^2}{D}\right)

```

For small $D$: $R(D) \approx \frac{1}{2}\log\_2(\sigma^2/D)$

**Implication for Quantization:**
With $b$ bits, minimum achievable distortion is:

```math
D_{min} = \sigma^2 \cdot 2^{-2b}

```

This shows 4-bit quantization achieves $D \approx \sigma^2/65536$, often sufficient for neural networks!

---

### 4. Optimal Quantization (Lloyd-Max)

#### 4.1 Problem Formulation

**Goal:** Find quantization levels $\{q\_i\}$ and decision boundaries $\{d\_i\}$ that minimize MSE:

```math
\min_{\{q_i\}, \{d_i\}} \int_{-\infty}^{\infty} (x - Q(x))^2 p(x) dx

```

#### 4.2 Lloyd-Max Conditions

**Theorem (Necessary Conditions):** The optimal quantizer satisfies:

1. **Centroid Condition:** Each quantization level is the centroid of its region:

```math
q_i = \frac{\int_{d_{i-1}}^{d_i} x \cdot p(x) dx}{\int_{d_{i-1}}^{d_i} p(x) dx} = \mathbb{E}[X | d_{i-1} < X \leq d_i]

```

2. **Nearest Neighbor Condition:** Each decision boundary is the midpoint:

```math
d_i = \frac{q_i + q_{i+1}}{2}

```

**Proof of Centroid Condition:**
Taking derivative of MSE w.r.t. $q\_i$:

```math
\frac{\partial}{\partial q_i} \int_{d_{i-1}}^{d_i} (x - q_i)^2 p(x) dx = -2\int_{d_{i-1}}^{d_i} (x - q_i) p(x) dx = 0

```

Solving: $q\_i = \frac{\int\_{d\_{i-1}}^{d\_i} x \cdot p(x) dx}{\int\_{d\_{i-1}}^{d\_i} p(x) dx}$

#### 4.3 Lloyd-Max Algorithm

```
1. Initialize quantization levels {q_i}

2. Repeat until convergence:
   a. Update boundaries: d_i = (q_i + q_{i+1})/2
   b. Update levels: q_i = E[X | d_{i-1} < X ≤ d_i]

3. Return {q_i}, {d_i}

```

---

### 5. Clipping and Calibration

#### 5.1 Clipping Range Selection

**Problem:** Choose clipping range $[-\alpha, \alpha]$ to minimize total error.

**Total Error = Clipping Error + Quantization Error**

```math
\mathcal{L}(\alpha) = \underbrace{\int_{|x|>\alpha} (x - \text{sign}(x)\alpha)^2 p(x) dx}_{\text{clipping}} + \underbrace{\frac{\Delta^2}{12} \cdot P(|x| \leq \alpha)}_{\text{quantization}}

```

#### 5.2 Optimal Clipping (Gaussian Case)

**Theorem:** For Gaussian $X \sim \mathcal{N}(0, \sigma^2)$, the optimal clipping factor $c = \alpha/\sigma$ satisfies:

```math
2\phi(c) = c \cdot \Phi(-c)

```

where $\phi$ is the PDF and $\Phi$ is the CDF.

**Numerical solution:** $c \approx 2.83$ for high bit-widths.

---

### 6. Calibration Methods

| Method | Formula | Pros | Cons |
|--------|---------|------|------|
| **Min-Max** | $\alpha = \max(\|x\|)$ | Simple | Outlier sensitive |
| **Percentile** | $\alpha = P\_{99.9}(\|x\|)$ | Robust | May clip |
| **MSE** | $\alpha = \arg\min\_\alpha \mathbb{E}[(x-Q(x))^2]$ | Optimal | Expensive |
| **Entropy** | $\alpha = \arg\min\_\alpha D\_{KL}(P \| Q)$ | Distribution-aware | Complex |

---

### 7. Code Implementation

```python
import torch
import numpy as np

class UniformQuantizer:
    """Uniform symmetric quantizer with configurable bit-width."""
    
    def __init__(self, bits: int = 8):
        self.bits = bits
        self.qmin = -2**(bits-1)
        self.qmax = 2**(bits-1) - 1
        self.scale = None
    
    def calibrate(self, x: torch.Tensor, method: str = 'minmax'):
        """Compute scale factor from calibration data."""
        if method == 'minmax':
            alpha = x.abs().max()
        elif method == 'percentile':
            alpha = torch.quantile(x.abs(), 0.999)
        elif method == 'mse':
            alpha = self._mse_calibrate(x)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.scale = alpha / self.qmax
        return self.scale
    
    def _mse_calibrate(self, x: torch.Tensor, num_points: int = 100):
        """Find alpha that minimizes MSE."""
        max_val = x.abs().max()
        best_alpha, best_mse = max_val, float('inf')
        
        for alpha in torch.linspace(0.5 * max_val, max_val, num_points):
            scale = alpha / self.qmax
            x_q = self.quantize(x, scale)
            x_dq = self.dequantize(x_q, scale)
            mse = ((x - x_dq) ** 2).mean()
            
            if mse < best_mse:
                best_mse = mse
                best_alpha = alpha
        
        return best_alpha
    
    def quantize(self, x: torch.Tensor, scale: float = None) -> torch.Tensor:
        """Quantize tensor to integers."""
        s = scale if scale else self.scale
        return torch.clamp(torch.round(x / s), self.qmin, self.qmax).to(torch.int8)
    
    def dequantize(self, x_q: torch.Tensor, scale: float = None) -> torch.Tensor:
        """Dequantize integers back to floats."""
        s = scale if scale else self.scale
        return x_q.float() * s
    
    def compute_sqnr(self, x: torch.Tensor) -> float:
        """Compute Signal-to-Quantization-Noise Ratio in dB."""
        x_q = self.quantize(x)
        x_dq = self.dequantize(x_q)
        noise = x - x_dq
        
        signal_power = (x ** 2).mean()
        noise_power = (noise ** 2).mean()
        
        sqnr_db = 10 * torch.log10(signal_power / noise_power)
        return sqnr_db.item()

# Example usage
x = torch.randn(1000)
quantizer = UniformQuantizer(bits=8)
quantizer.calibrate(x, method='mse')

x_q = quantizer.quantize(x)
x_dq = quantizer.dequantize(x_q)

print(f"Scale: {quantizer.scale:.6f}")
print(f"MSE: {((x - x_dq) ** 2).mean():.6f}")
print(f"SQNR: {quantizer.compute_sqnr(x):.2f} dB")
print(f"Theoretical SQNR: {6.02 * 8 + 4.77:.2f} dB")

```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | Bennett (1948) | Quantization noise analysis |
| 📄 | Lloyd (1982) | Optimal scalar quantization |
| 📄 | Max (1960) | Lloyd-Max algorithm |
| 📄 | Gray & Neuhoff (1998) | Quantization survey |
| 🇨🇳 | 量化基础详解 | [知乎](https://zhuanlan.zhihu.com/p/132561405) |
| 🇨🇳 | 神经网络量化入门 | [CSDN](https://blog.csdn.net/weixin_44878336/article/details/124792645) |
| 🇨🇳 | 模型量化理论 | [B站](https://www.bilibili.com/video/BV1S34y1h7Fz) |

---

⬅️ [Back: Quantization](../README.md) | ➡️ [Next: PTQ](../02_ptq/README.md)

