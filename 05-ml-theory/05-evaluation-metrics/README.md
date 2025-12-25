<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=05 Evaluation Metrics&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---


# 📊 Evaluation Metrics

> **Measuring model performance correctly**

---

## 🎯 Visual Overview

<img src="./images/precision-recall-roc-complete.svg" width="100%">

*Caption: Evaluation metrics measure different aspects of model performance. Precision-Recall for imbalanced data, ROC-AUC for ranking, F1 for balance between precision and recall.*

---

## 📐 Mathematical Foundations

### Confusion Matrix

```
                  Predicted
                  Pos    Neg
Actual  Pos       TP     FN
        Neg       FP     TN

TP = True Positive (correct positive)
FP = False Positive (Type I error)
FN = False Negative (Type II error)
TN = True Negative (correct negative)
```

### Classification Metrics

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)

Precision = TP / (TP + FP)
  "Of predicted positives, how many are correct?"

Recall (Sensitivity) = TP / (TP + FN)
  "Of actual positives, how many did we find?"

Specificity = TN / (TN + FP)
  "Of actual negatives, how many did we identify?"

F1 Score = 2 × (Precision × Recall) / (Precision + Recall)
  Harmonic mean of precision and recall
```

### ROC and AUC

```
ROC Curve: TPR vs FPR at different thresholds
  TPR = TP / (TP + FN) = Recall
  FPR = FP / (FP + TN) = 1 - Specificity

AUC (Area Under Curve):
  • AUC = 1.0: Perfect classifier
  • AUC = 0.5: Random guessing
  • AUC < 0.5: Worse than random
```

---

## 🎯 Metrics Comparison

| Metric | Formula | Use Case | Imbalanced Data |
|--------|---------|----------|-----------------|
| **Accuracy** | (TP+TN)/All | Balanced classes | ❌ Misleading |
| **Precision** | TP/(TP+FP) | Cost of FP high | ✅ Good |
| **Recall** | TP/(TP+FN) | Cost of FN high | ✅ Good |
| **F1** | 2PR/(P+R) | Balance P & R | ✅ Good |
| **AUC-ROC** | Area under ROC | Ranking ability | ⚠️ Can be misleading |
| **AUC-PR** | Area under PR | Imbalanced | ✅ Best |

---

## 💻 Code Examples

```python
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt

# Sample predictions
y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0])
y_prob = np.array([0.9, 0.1, 0.8, 0.4, 0.2, 0.7, 0.6, 0.3, 0.85, 0.15])

# Basic metrics
print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
print(f"Precision: {precision_score(y_true, y_pred):.4f}")
print(f"Recall: {recall_score(y_true, y_pred):.4f}")
print(f"F1: {f1_score(y_true, y_pred):.4f}")

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
print(f"Confusion Matrix:\n{cm}")

# Full classification report
print(classification_report(y_true, y_pred))

# AUC metrics (require probabilities)
roc_auc = roc_auc_score(y_true, y_prob)
pr_auc = average_precision_score(y_true, y_prob)
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"PR-AUC: {pr_auc:.4f}")

# ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_prob)

# Precision-Recall curve
precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

# Multi-class F1
y_true_multi = np.array([0, 1, 2, 0, 1, 2])
y_pred_multi = np.array([0, 2, 1, 0, 0, 2])
f1_macro = f1_score(y_true_multi, y_pred_multi, average='macro')
f1_micro = f1_score(y_true_multi, y_pred_multi, average='micro')
f1_weighted = f1_score(y_true_multi, y_pred_multi, average='weighted')
```

---

## 🌍 ML Applications

| Application | Primary Metric | Why |
|-------------|----------------|-----|
| **Spam Detection** | Precision | Don't want to block real emails |
| **Disease Screening** | Recall | Don't want to miss cases |
| **Search Ranking** | NDCG, MAP | Order matters |
| **Fraud Detection** | PR-AUC | Imbalanced data |
| **Object Detection** | mAP | Multi-class, multi-object |

---

## 📊 Regression Metrics

```
MSE = (1/n) Σᵢ(yᵢ - ŷᵢ)²
RMSE = √MSE
MAE = (1/n) Σᵢ|yᵢ - ŷᵢ|
R² = 1 - Σ(yᵢ - ŷᵢ)² / Σ(yᵢ - ȳ)²

MAPE = (100/n) Σᵢ|yᵢ - ŷᵢ|/|yᵢ|  (percentage error)
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | Scikit-learn Metrics | [Docs](https://scikit-learn.org/stable/modules/model_evaluation.html) |
| 📄 | Precision-Recall vs ROC | [Paper](https://www.biostat.wisc.edu/~page/rocpr.pdf) |
| 🎥 | Metrics Explained | [YouTube](https://www.youtube.com/watch?v=LbX4X71-TFI) |
| 🇨🇳 | 评估指标详解 | [知乎](https://zhuanlan.zhihu.com/p/30721429) |
| 🇨🇳 | 机器学习评估 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88666666) |

---

## 🔗 Where This Topic Is Used

| Application | How Metrics Are Used |
|-------------|---------------------|
| **Model Selection** | Compare models fairly |
| **Hyperparameter Tuning** | Optimize right objective |
| **Production Monitoring** | Track model performance |
| **A/B Testing** | Statistical significance |
| **Kaggle Competitions** | Leaderboard ranking |

---

⬅️ [Back: 04-Representation](../04-representation/) | ➡️ [Next: 06-Ensemble Methods](../06-ensemble-methods/)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>

