<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=1ABC9C&height=120&section=header&text=Lecture%205%20Quantization%20Part%20I&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-09-1ABC9C?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

# Lecture 5: Quantization (Part I)

[← Back to Course](../) | [← Previous](../04_pruning_sparsity_2/) | [Next: Quantization II →](../06_quantization_2/)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/ml-researcher-foundations/blob/main/09_efficient_ml/05_quantization_1/demo.ipynb) ← **Try the code!**

---

## 📺 Video Lecture

| Resource | Link |
|----------|------|
| 🎥 **Lecture Video** | [Watch on YouTube](https://www.youtube.com/watch?v=xYDadmguObs&list=PL80kAHvQbh-pT4lCkDT53zT8DKmhE0idB&index=5) |
| 📊 **Slides** | [MIT Course Page](https://hanlab.mit.edu/courses/2023-fall-65940) |
| ⏱️ **Duration** | ~90 minutes |

### 📝 Video Summary

This lecture introduces **quantization fundamentals** for neural network compression:

- **What is quantization**: Reducing precision from FP32 to INT8/INT4
- **Quantization formula**: Scale, zero-point, and the quantization function
- **Symmetric vs Asymmetric**: Trade-offs in quantization schemes
- **Granularity levels**: Per-tensor, per-channel, per-group quantization
- **Post-Training Quantization (PTQ)**: Quantizing pre-trained models
- **Calibration methods**: Min-Max, percentile, MSE, KL-divergence

> 💡 *"Quantization provides a 4× memory reduction with INT8 and 8× with INT4—often with minimal accuracy loss."* — Prof. Song Han

---

![Overview](overview.png)

## What is Quantization?

**Quantization** reduces the precision of weights and activations from FP32 to INT8/INT4.

```
FP32 (32 bits) → INT8 (8 bits) = 4x memory reduction
FP32 (32 bits) → INT4 (4 bits) = 8x memory reduction
```

---

## Data Types

| Type | Bits | Range | Use Case |
|------|------|-------|----------|
| FP32 | 32 | ±3.4e38 | Training |
| FP16 | 16 | ±65504 | Mixed precision |
| BF16 | 16 | ±3.4e38 | Training (wider range) |
| INT8 | 8 | -128 to 127 | Inference |
| INT4 | 4 | -8 to 7 | LLM inference |

---

## Quantization Formula

Map floating point to integer:

```
q = round(x / scale) + zero_point

# Dequantize:
x_approx = (q - zero_point) * scale
```

### Example
```python
# FP32 weights: [0.1, 0.5, 0.9, 1.2]
# Scale = 1.2 / 127 ≈ 0.0094
# INT8: [11, 53, 96, 127]
```

---

## Symmetric vs Asymmetric

### Symmetric Quantization
- Zero point = 0
- Range: [-α, α]
- Simpler computation

```
q = round(x / scale)
```

### Asymmetric Quantization
- Zero point ≠ 0
- Range: [β, α] (not centered)
- Better for ReLU outputs

```
q = round(x / scale) + zero_point
```

---

## Quantization Granularity

| Level | Description | Accuracy | Speed |
|-------|-------------|----------|-------|
| Per-tensor | One scale for entire tensor | Lower | Fastest |
| Per-channel | One scale per output channel | Higher | Fast |
| Per-group | One scale per N weights | Highest | Slower |

---

## Post-Training Quantization (PTQ)

Quantize a pre-trained model without retraining:

```python
# 1. Calibrate on sample data
model.eval()
with torch.no_grad():
    for batch in calibration_data:
        model(batch)  # Collect activation statistics

# 2. Compute scales from min/max values
scale = (max_val - min_val) / 255
zero_point = round(-min_val / scale)

# 3. Quantize weights
quantized_weights = round(weights / scale) + zero_point
```

---

## 📐 Mathematical Foundations & Proofs

### Uniform Quantization

Map continuous values \( x \in [x_{min}, x_{max}] \) to discrete integers \( q \in [0, 2^b - 1] \):

$$q = \text{round}\left(\frac{x - x_{min}}{x_{max} - x_{min}} \cdot (2^b - 1)\right)$$

**Scale factor:**
$$s = \frac{x_{max} - x_{min}}{2^b - 1}$$

**Zero-point:**
$$z = \text{round}\left(-\frac{x_{min}}{s}\right)$$

**Quantization function:**
$$Q(x) = \text{clamp}\left(\text{round}\left(\frac{x}{s}\right) + z, 0, 2^b-1\right)$$

**Dequantization:**
$$\hat{x} = s \cdot (q - z)$$

---

### Quantization Error Analysis

**Quantization error:**
$$\epsilon = x - \hat{x} = x - s \cdot (Q(x) - z)$$

**Mean Squared Error (for uniform distribution):**

Assuming uniform quantization with step size \( \Delta = s \):
$$\mathbb{E}[\epsilon^2] = \frac{\Delta^2}{12} = \frac{s^2}{12}$$

**Proof:**
For uniform rounding error \( \epsilon \sim \text{Uniform}(-\Delta/2, \Delta/2) \):
$$\mathbb{E}[\epsilon^2] = \int_{-\Delta/2}^{\Delta/2} \epsilon^2 \cdot \frac{1}{\Delta} d\epsilon = \frac{1}{\Delta} \cdot \frac{\epsilon^3}{3}\Big|_{-\Delta/2}^{\Delta/2} = \frac{\Delta^2}{12}$$

---

### Symmetric vs Asymmetric Quantization

**Symmetric (zero-point = 0):**
$$q = \text{round}\left(\frac{x}{s}\right), \quad s = \frac{\max(|x_{max}|, |x_{min}|)}{2^{b-1}-1}$$

**Asymmetric (zero-point ≠ 0):**
$$q = \text{round}\left(\frac{x}{s}\right) + z, \quad s = \frac{x_{max} - x_{min}}{2^b - 1}$$

**Trade-off:**
- Symmetric: Simpler hardware (no zero-point addition), but wastes range if data is skewed
- Asymmetric: Full range utilization, better for ReLU activations (all positive)

---

### Per-Channel Quantization

For weight tensor \( W \in \mathbb{R}^{C_{out} \times C_{in} \times k \times k} \):

**Per-tensor:** One scale \( s \) for all of \( W \)
$$W_q = Q(W; s, z)$$

**Per-channel:** Separate scale \( s_c \) for each output channel
$$W_q[c,:,:,:] = Q(W[c,:,:,:]; s_c, z_c)$$

**Why per-channel is better:**

Different channels can have very different weight distributions:
$$\text{Var}(W[c_1]) \gg \text{Var}(W[c_2])$$

Per-channel adapts to each distribution → lower quantization error.

---

### Calibration Methods

#### 1. Min-Max Calibration
$$s = \frac{x_{max} - x_{min}}{2^b - 1}$$

Simple but sensitive to outliers.

#### 2. Percentile Calibration
Use 99.9th percentile instead of max:
$$s = \frac{P_{99.9}(|x|)}{2^{b-1} - 1}$$

Robust to outliers but may clip extreme values.

#### 3. MSE Calibration
Find scale that minimizes reconstruction error:
$$s^* = \arg\min_s \|x - \hat{x}(s)\|_2^2$$

Solved by grid search over candidate scales.

#### 4. KL-Divergence Calibration (TensorRT)
Minimize information loss:
$$s^* = \arg\min_s D_{KL}(P_x \| P_{\hat{x}})$$

where \( P_x \) and \( P_{\hat{x}} \) are histograms of original and quantized values.

---

### Quantized Matrix Multiplication

For \( Y = XW \) with quantized inputs:

$$Y = (X_q - z_x) \cdot s_x \cdot (W_q - z_w) \cdot s_w$$

Expanding:
$$Y = s_x s_w \left[ X_q W_q - z_x \sum W_q - z_w \sum X_q + z_x z_w \cdot \text{const} \right]$$

**Optimization:** Pre-compute \( z_x \sum W_q \), \( z_w \sum X_q \), and \( z_x z_w \) terms.

For symmetric quantization (\( z_x = z_w = 0 \)):
$$Y = s_x s_w \cdot X_q W_q$$

Much simpler! Just integer matmul + single scaling.

---

### Clipping and Optimal Range

**Problem:** Choosing \( [x_{min}, x_{max}] \) involves trade-off:
- Wide range → large quantization step → large quantization error
- Narrow range → clipping outliers → clipping error

**Total error:**
$$\mathcal{L}(r) = \mathcal{L}_{quant}(r) + \mathcal{L}_{clip}(r)$$

**Optimal range minimizes total error:**
$$r^* = \arg\min_r \mathcal{L}(r)$$

For Gaussian-distributed weights with std \( \sigma \):
$$r^* \approx 2.5\sigma \text{ to } 3\sigma$$

This clips ~1% of values but significantly reduces quantization error.

---

## 🧮 Key Derivations

### Quantization-Aware Gradient

During QAT, the quantization function is:
$$y = Q(x) = s \cdot \text{round}\left(\frac{x}{s}\right)$$

The gradient is zero almost everywhere (round is step function).

**Straight-Through Estimator (STE):**
$$\frac{\partial y}{\partial x} \approx 1 \text{ (identity)}$$

**Proof of STE validity:**
In expectation, the gradient of round is 1:
$$\mathbb{E}\left[\frac{\partial \text{round}(x)}{\partial x}\right] = 1$$

because the quantization error has zero mean.

---

### Bit-Width and Model Size

For a model with \( N \) parameters:

| Precision | Size |
|-----------|------|
| FP32 | \( 4N \) bytes |
| FP16 | \( 2N \) bytes |
| INT8 | \( N \) bytes |
| INT4 | \( 0.5N \) bytes |

**Compression ratio FP32→INT4:** 8×

---

### Hardware Efficiency

**Compute:** INT8 operations are 2-4× faster than FP32 on modern hardware.

**Memory bandwidth:** 4× less data to move with INT8 vs FP32.

**Total speedup:** Often 2-4× for inference (memory-bound operations benefit most).

---

## 🎯 Where Used

| Concept | Applications |
|---------|-------------|
| INT8 Quantization | CNN inference on edge devices |
| INT4 Quantization | LLM inference (GPTQ, AWQ) |
| Per-channel Quant | Accurate conv layer quantization |
| Calibration | Post-training quantization |

---


## 🗺️ Navigation

| ⬅️ Previous | 🏠 Home | ➡️ Next |
|:-----------:|:-------:|:-------:|
| [← Pruning & Sparsity II](../04_pruning_sparsity_2/README.md) | [Efficient ML](../README.md) | [Quantization II →](../06_quantization_2/README.md) |

---
## 📚 References

| Type | Resource | Link |
|------|----------|------|
| 📄 | Google Quantization Paper | [arXiv](https://arxiv.org/abs/1712.05877) |
| 📄 | GPTQ: LLM Quantization | [arXiv](https://arxiv.org/abs/2210.17323) |
| 💻 | PyTorch Quantization | [PyTorch](https://pytorch.org/docs/stable/quantization.html) |
| 📄 | TensorRT Quantization | [NVIDIA](https://developer.nvidia.com/tensorrt) |
| 🎥 | MIT 6.5940 TinyML | [Course](https://hanlab.mit.edu/courses/2024-fall-65940) |

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=1ABC9C&height=80&section=footer" width="100%"/>
</p>
