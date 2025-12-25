<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=10 Interpretability&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---


# 🔍 Model Interpretability

> **Understanding why models make predictions**

---

## 🎯 Visual Overview

<img src="./images/model-interpretability-complete.svg" width="100%">

*Caption: Interpretability methods help explain model decisions. SHAP provides consistent feature attributions, LIME explains locally, attention weights show what models focus on.*

---

## 📐 Mathematical Foundations

### SHAP (SHapley Additive exPlanations)

```
Based on Shapley values from game theory:

φᵢ = Σ_{S⊆N\{i}} |S|!(n-|S|-1)!/n! [f(S∪{i}) - f(S)]

Properties:
• Local accuracy: f(x) = φ₀ + Σᵢ φᵢ
• Consistency: If feature contribution increases, so does SHAP
• Missingness: Missing features get φ = 0

For tree models: TreeSHAP is O(TL²) instead of O(2ⁿ)
```

### LIME (Local Interpretable Model-agnostic Explanations)

```
Fit interpretable model g locally around x:

ξ(x) = argmin_{g∈G} L(f, g, πₓ) + Ω(g)

Where:
• L = loss between f and g weighted by πₓ
• πₓ = locality kernel (e.g., exponential)
• Ω(g) = complexity penalty (encourage sparse g)

g is typically a linear model or decision tree
```

### Integrated Gradients

```
IG(x)ᵢ = (xᵢ - x'ᵢ) × ∫₀¹ (∂f/∂xᵢ)(x' + α(x-x')) dα

Where x' is baseline (e.g., zero image)

Properties:
• Sensitivity: If feature matters, IG ≠ 0
• Completeness: Σᵢ IGᵢ = f(x) - f(x')
```

---

## 🎯 Method Comparison

| Method | Scope | Model-agnostic | Computation |
|--------|-------|----------------|-------------|
| **SHAP** | Global/Local | Yes | Expensive |
| **LIME** | Local | Yes | Moderate |
| **Integrated Gradients** | Local | No (differentiable) | Fast |
| **Attention** | Local | No (attention models) | Free |
| **Feature Importance** | Global | No (tree models) | Fast |

---

## 💻 Code Examples

```python
import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Create dataset and model
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
feature_names = [f'feature_{i}' for i in range(20)]
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X[:100])

# Summary plot (global importance)
shap.summary_plot(shap_values[1], X[:100], feature_names=feature_names)

# Force plot (local explanation)
shap.force_plot(
    explainer.expected_value[1], 
    shap_values[1][0], 
    X[0],
    feature_names=feature_names
)

# Feature importance from trees
importances = model.feature_importances_
top_features = np.argsort(importances)[-5:][::-1]
print(f"Top 5 features: {top_features}")

# LIME example
from lime.lime_tabular import LimeTabularExplainer

lime_explainer = LimeTabularExplainer(
    X,
    feature_names=feature_names,
    class_names=['class_0', 'class_1'],
    mode='classification'
)

# Explain single prediction
explanation = lime_explainer.explain_instance(
    X[0],
    model.predict_proba,
    num_features=10
)
explanation.show_in_notebook()

# Integrated Gradients (for neural networks)
# Using captum library
try:
    import torch
    import torch.nn as nn
    from captum.attr import IntegratedGradients
    
    class SimpleNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(20, 50),
                nn.ReLU(),
                nn.Linear(50, 2)
            )
        
        def forward(self, x):
            return self.fc(x)
    
    nn_model = SimpleNN()
    ig = IntegratedGradients(nn_model)
    
    x_tensor = torch.tensor(X[:1], dtype=torch.float32)
    baseline = torch.zeros_like(x_tensor)
    
    attributions = ig.attribute(x_tensor, baseline, target=1)
    print(f"IG attributions shape: {attributions.shape}")
except ImportError:
    print("Captum not installed")
```

---

## 🌍 ML Applications

| Application | Method | Why |
|-------------|--------|-----|
| **Healthcare** | SHAP, LIME | Explain diagnoses |
| **Finance** | SHAP | Regulatory compliance |
| **NLP** | Attention, IG | Token importance |
| **Computer Vision** | GradCAM, IG | Spatial attention |
| **Debugging** | All methods | Find model issues |

---

## 📊 Interpretability Spectrum

```
Inherently Interpretable    ←→    Black Box + Explanation
        
Linear Regression               Deep Learning + SHAP
Decision Trees                  Neural Networks + LIME
Rule Lists                      Transformers + Attention
GAMs                            CNNs + GradCAM
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | SHAP Paper | [Paper](https://arxiv.org/abs/1705.07874) |
| 📄 | LIME Paper | [Paper](https://arxiv.org/abs/1602.04938) |
| 📄 | Integrated Gradients | [Paper](https://arxiv.org/abs/1703.01365) |
| 📖 | Interpretable ML Book | [Book](https://christophm.github.io/interpretable-ml-book/) |
| 🇨🇳 | 可解释性机器学习 | [知乎](https://zhuanlan.zhihu.com/p/97629463) |

---

## 🔗 Where This Topic Is Used

| Application | How Interpretability Is Used |
|-------------|------------------------------|
| **Model Debugging** | Understand model errors |
| **Feature Engineering** | Identify important features |
| **Regulatory Compliance** | Explain decisions |
| **User Trust** | Build confidence in predictions |
| **Scientific Discovery** | Understand learned patterns |

---

⬅️ [Back: 09-Hyperparameter Tuning](../09-hyperparameter-tuning/) | ➡️ [Next: 11-Adversarial Robustness](../11-adversarial-robustness/)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>

