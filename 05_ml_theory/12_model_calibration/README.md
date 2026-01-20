<!-- Navigation -->
<p align="center">
  <a href="../11_adversarial_robustness/">⬅️ Prev: Adversarial Robustness</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../">📚 ML Theory</a>
</p>

---

<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=120&section=header&text=Model%20Calibration&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-05-4ECDC4?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/model-calibration-complete.svg" width="100%">

*Caption: A calibrated model's confidence matches its accuracy. If a model says 80% confidence, it should be correct 80% of the time. Reliability diagrams visualize calibration.*

---

## 📐 Mathematical Foundations

### Calibration Definition

```
A model is perfectly calibrated if:
P(Y = y | P̂ = p) = p  for all p ∈ [0, 1]

"When the model says 80% confidence, 
 it's correct 80% of the time"

Modern neural networks are often overconfident!
```

### Calibration Metrics

```
Expected Calibration Error (ECE):
ECE = Σᵢ (nᵢ/n) |acc(Bᵢ) - conf(Bᵢ)|

Bin samples by confidence, compare accuracy vs confidence

Maximum Calibration Error (MCE):
MCE = max_i |acc(Bᵢ) - conf(Bᵢ)|

Worst-case calibration error

Brier Score:
BS = (1/n) Σᵢ (pᵢ - yᵢ)²

Measures both calibration and refinement
```

### Temperature Scaling

```
Post-hoc calibration method:

z_calibrated = z / T

Where:
• z = logits (before softmax)
• T = temperature (learned on validation set)
• T > 1: Softer probabilities (less confident)
• T < 1: Sharper probabilities (more confident)

Find T by minimizing NLL on validation set
```

---

## 🎯 Calibration Methods

| Method | Type | Complexity | Accuracy Impact |
|--------|------|------------|-----------------|
| **Temperature Scaling** | Post-hoc | 1 parameter | None |
| **Platt Scaling** | Post-hoc | 2 parameters | None |
| **Isotonic Regression** | Post-hoc | Non-parametric | None |
| **Label Smoothing** | Training | Regularization | Slight |
| **Mixup** | Training | Data augmentation | Often improves |

---

## 💻 Code Examples

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression

# Temperature Scaling
class TemperatureScaling(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
    
    def forward(self, logits):
        return logits / self.temperature
    
    def fit(self, logits, labels, lr=0.01, max_iter=50):
        """Fit temperature on validation set"""
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        
        def closure():
            optimizer.zero_grad()
            scaled_logits = self(logits)
            loss = F.cross_entropy(scaled_logits, labels)
            loss.backward()
            return loss
        
        optimizer.step(closure)
        return self.temperature.item()

# Calculate ECE
def expected_calibration_error(probs, labels, n_bins=10):
    """Calculate Expected Calibration Error"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0
    
    for i in range(n_bins):
        # Get samples in this bin
        in_bin = (probs > bin_boundaries[i]) & (probs <= bin_boundaries[i + 1])
        
        if in_bin.sum() > 0:
            bin_accuracy = labels[in_bin].mean()
            bin_confidence = probs[in_bin].mean()
            bin_size = in_bin.sum() / len(probs)
            
            ece += bin_size * np.abs(bin_accuracy - bin_confidence)
    
    return ece

# Reliability Diagram
def reliability_diagram(probs, labels, n_bins=10):
    """Create reliability diagram data"""
    fraction_positives, mean_predicted = calibration_curve(
        labels, probs, n_bins=n_bins, strategy='uniform'
    )
    return mean_predicted, fraction_positives

# Example usage
np.random.seed(42)
n_samples = 1000

# Simulated uncalibrated model (overconfident)
true_probs = np.random.beta(2, 5, n_samples)  # True probability
labels = (np.random.rand(n_samples) < true_probs).astype(int)
predicted_probs = np.clip(true_probs + 0.2, 0, 1)  # Overconfident

# Before calibration
ece_before = expected_calibration_error(predicted_probs, labels)
print(f"ECE before calibration: {ece_before:.4f}")

# Isotonic Regression calibration
ir = IsotonicRegression(out_of_bounds='clip')
calibrated_probs = ir.fit_transform(predicted_probs, labels)

# After calibration
ece_after = expected_calibration_error(calibrated_probs, labels)
print(f"ECE after calibration: {ece_after:.4f}")

# Label Smoothing during training
def smooth_labels(labels, num_classes, smoothing=0.1):
    """Apply label smoothing"""
    confidence = 1.0 - smoothing
    smooth_value = smoothing / (num_classes - 1)
    
    one_hot = torch.zeros(len(labels), num_classes)
    one_hot.fill_(smooth_value)
    one_hot.scatter_(1, labels.unsqueeze(1), confidence)
    
    return one_hot
```

---

## 🌍 ML Applications

| Application | Why Calibration Matters |
|-------------|------------------------|
| **Medical Diagnosis** | Confidence = risk assessment |
| **Autonomous Driving** | When to defer to human |
| **Weather Prediction** | Probability of rain |
| **Recommender Systems** | Confidence in recommendations |
| **Active Learning** | Uncertainty sampling |

---

## 📊 Common Issues

```
Overconfidence:
• Neural networks often too confident
• Softmax outputs ≠ true probabilities
• Training on cross-entropy pushes toward extremes

Underconfidence:
• Less common
• Often due to regularization

Solutions:
• Temperature scaling (simple, effective)
• Label smoothing during training
• Ensemble methods
• Bayesian neural networks
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | On Calibration of Modern NNs | [Paper](https://arxiv.org/abs/1706.04599) |
| 📄 | Temperature Scaling | [Paper](https://arxiv.org/abs/1706.04599) |
| 📄 | Mixup Calibration | [Paper](https://arxiv.org/abs/1905.11001) |
| 🇨🇳 | 模型校准详解 | [知乎](https://zhuanlan.zhihu.com/p/90479183) |
| 🇨🇳 | 概率校准方法 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88111111) |

---

## 🔗 Where This Topic Is Used

| Application | How Calibration Is Used |
|-------------|------------------------|
| **Uncertainty Quantification** | Reliable confidence estimates |
| **Decision Making** | Risk-aware predictions |
| **Model Ensembling** | Better probability averaging |
| **Active Learning** | Uncertainty-based sampling |
| **Production ML** | Trustworthy predictions |

---

⬅️ [Back: 11-Adversarial Robustness](../11_adversarial_robustness/) | ➡️ [Back: ML Theory](../)

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<!-- Navigation -->
<p align="center">
  <a href="../11_adversarial_robustness/">⬅️ Prev: Adversarial Robustness</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../">📚 ML Theory</a>
</p>

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=80&section=footer" width="100%"/>
</p>
