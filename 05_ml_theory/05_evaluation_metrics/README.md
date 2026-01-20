<!-- Navigation -->
<p align="center">
  <a href="../04_representation/">⬅️ Prev: Representation</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../">📚 ML Theory</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../05_risk_minimization/">Next: Risk Minimization ➡️</a>
</p>

---

<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=120&section=header&text=Evaluation%20Metrics&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-05-4ECDC4?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/precision-recall-roc-complete.svg" width="100%">

*Caption: Evaluation metrics quantify different aspects of model performance.*

---

## 📂 Overview

**Evaluation metrics** measure how well a model performs. Choosing the right metric is crucial—optimizing the wrong metric leads to models that fail in practice.

---

## 📐 Classification Metrics

### Confusion Matrix

|  | Predicted Positive | Predicted Negative |
|--|-------------------|--------------------|
| **Actual Positive** | TP (True Positive) | FN (False Negative) |
| **Actual Negative** | FP (False Positive) | TN (True Negative) |

### Primary Metrics

**Accuracy:**

```math
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}

```

**Precision (Positive Predictive Value):**

```math
\text{Precision} = \frac{TP}{TP + FP}

```

*"Of all predicted positives, how many are correct?"*

**Recall (Sensitivity, True Positive Rate):**

```math
\text{Recall} = \frac{TP}{TP + FN}

```

*"Of all actual positives, how many did we find?"*

**Specificity (True Negative Rate):**

```math
\text{Specificity} = \frac{TN}{TN + FP}

```

**F1 Score (Harmonic Mean):**

```math
F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2TP}{2TP + FP + FN}

```

**F-beta Score (Weighted):**

```math
F_\beta = (1 + \beta^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{\beta^2 \cdot \text{Precision} + \text{Recall}}

```

- \(\beta > 1\): Emphasize recall
- \(\beta < 1\): Emphasize precision

---

## 📐 ROC and AUC

### ROC Curve

**True Positive Rate (TPR):** \(\frac{TP}{TP + FN}\)

**False Positive Rate (FPR):** \(\frac{FP}{FP + TN}\)

ROC curve plots TPR vs FPR at all classification thresholds.

### AUC (Area Under ROC)

```math
\text{AUC} = \int_0^1 \text{TPR}(t) \, d\text{FPR}(t)

```

**Interpretation:**

```math
\text{AUC} = P(\text{score}(x^+) > \text{score}(x^-))

```

Probability that a random positive is ranked higher than a random negative.

| AUC | Interpretation |
|-----|----------------|
| 1.0 | Perfect |
| 0.9-1.0 | Excellent |
| 0.8-0.9 | Good |
| 0.7-0.8 | Fair |
| 0.5 | Random |

---

## 📐 Precision-Recall Curve

For imbalanced data, PR curve is more informative:

**Average Precision (AP):**

```math
\text{AP} = \sum_n (R_n - R_{n-1}) P_n

```

where \(P_n\) and \(R_n\) are precision and recall at threshold \(n\).

### Why PR > ROC for Imbalanced Data

When negatives dominate, FPR stays low even with many false positives (denominator is large). PR curve exposes this.

---

## 📐 Regression Metrics

**Mean Squared Error (MSE):**

```math
\text{MSE} = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2

```

**Root Mean Squared Error (RMSE):**

```math
\text{RMSE} = \sqrt{\text{MSE}}

```

**Mean Absolute Error (MAE):**

```math
\text{MAE} = \frac{1}{n}\sum_{i=1}^n |y_i - \hat{y}_i|

```

**R² (Coefficient of Determination):**

```math
R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2} = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}}

```

**MAPE (Mean Absolute Percentage Error):**

```math
\text{MAPE} = \frac{100\%}{n}\sum_{i=1}^n \left|\frac{y_i - \hat{y}_i}{y_i}\right|

```

---

## 💻 Code Implementation

```python
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve, mean_squared_error, r2_score
)

class ClassificationMetrics:
    """Comprehensive classification evaluation."""
    
    def __init__(self, y_true, y_pred, y_prob=None):
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_prob = y_prob
        self._compute_confusion_matrix()
    
    def _compute_confusion_matrix(self):
        cm = confusion_matrix(self.y_true, self.y_pred)
        if cm.shape == (2, 2):
            self.tn, self.fp, self.fn, self.tp = cm.ravel()
        else:
            self.tn = self.fp = self.fn = self.tp = None
    
    def accuracy(self):
        """(TP + TN) / Total"""
        return accuracy_score(self.y_true, self.y_pred)
    
    def precision(self):
        """TP / (TP + FP)"""
        return precision_score(self.y_true, self.y_pred, zero_division=0)
    
    def recall(self):
        """TP / (TP + FN)"""
        return recall_score(self.y_true, self.y_pred, zero_division=0)
    
    def specificity(self):
        """TN / (TN + FP)"""
        if self.tn is not None:
            return self.tn / (self.tn + self.fp) if (self.tn + self.fp) > 0 else 0
        return None
    
    def f1(self):
        """2 * P * R / (P + R)"""
        return f1_score(self.y_true, self.y_pred, zero_division=0)
    
    def f_beta(self, beta=1.0):
        """(1 + β²) * P * R / (β² * P + R)"""
        p = self.precision()
        r = self.recall()
        if p + r == 0:
            return 0
        return (1 + beta**2) * p * r / (beta**2 * p + r)
    
    def roc_auc(self):
        """Area under ROC curve"""
        if self.y_prob is None:
            raise ValueError("Need probability scores for AUC")
        return roc_auc_score(self.y_true, self.y_prob)
    
    def pr_auc(self):
        """Area under Precision-Recall curve"""
        if self.y_prob is None:
            raise ValueError("Need probability scores for PR-AUC")
        return average_precision_score(self.y_true, self.y_prob)
    
    def report(self):
        """Print comprehensive metrics report."""
        print("Classification Metrics Report")
        print("=" * 40)
        print(f"Accuracy:    {self.accuracy():.4f}")
        print(f"Precision:   {self.precision():.4f}")
        print(f"Recall:      {self.recall():.4f}")
        print(f"F1 Score:    {self.f1():.4f}")
        if self.specificity() is not None:
            print(f"Specificity: {self.specificity():.4f}")
        if self.y_prob is not None:
            print(f"ROC-AUC:     {self.roc_auc():.4f}")
            print(f"PR-AUC:      {self.pr_auc():.4f}")

class RegressionMetrics:
    """Comprehensive regression evaluation."""
    
    def __init__(self, y_true, y_pred):
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
    
    def mse(self):
        """Mean Squared Error"""
        return np.mean((self.y_true - self.y_pred) ** 2)
    
    def rmse(self):
        """Root Mean Squared Error"""
        return np.sqrt(self.mse())
    
    def mae(self):
        """Mean Absolute Error"""
        return np.mean(np.abs(self.y_true - self.y_pred))
    
    def r2(self):
        """R² (Coefficient of Determination)"""
        ss_res = np.sum((self.y_true - self.y_pred) ** 2)
        ss_tot = np.sum((self.y_true - np.mean(self.y_true)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    def mape(self):
        """Mean Absolute Percentage Error"""
        mask = self.y_true != 0
        return 100 * np.mean(np.abs(
            (self.y_true[mask] - self.y_pred[mask]) / self.y_true[mask]
        ))
    
    def report(self):
        """Print comprehensive metrics report."""
        print("Regression Metrics Report")
        print("=" * 40)
        print(f"MSE:  {self.mse():.4f}")
        print(f"RMSE: {self.rmse():.4f}")
        print(f"MAE:  {self.mae():.4f}")
        print(f"R²:   {self.r2():.4f}")
        try:
            print(f"MAPE: {self.mape():.2f}%")
        except:
            print("MAPE: N/A (zeros in y_true)")

def plot_roc_pr_curves(y_true, y_prob):
    """Plot ROC and PR curves side by side."""
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    ax1.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
    ax1.plot([0, 1], [0, 1], 'k--', label='Random')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend()
    
    # PR Curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    
    ax2.plot(recall, precision, label=f'AP = {pr_auc:.3f}')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Binary classification example
    np.random.seed(42)
    y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0])
    y_prob = np.array([0.9, 0.1, 0.8, 0.4, 0.2, 0.7, 0.6, 0.3, 0.85, 0.15])
    
    metrics = ClassificationMetrics(y_true, y_pred, y_prob)
    metrics.report()

```

---

## 📊 Metric Selection Guide

| Scenario | Recommended Metric | Reason |
|----------|-------------------|--------|
| Balanced classes | Accuracy, F1 | Fair comparison |
| Imbalanced classes | PR-AUC, F1 | Accuracy misleading |
| Cost of FP high | Precision | Minimize false alarms |
| Cost of FN high | Recall | Don't miss positives |
| Ranking | AUC-ROC, NDCG | Order matters |
| Regression | RMSE, MAE | Scale-dependent |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | Scikit-learn Metrics | [Docs](https://scikit-learn.org/stable/modules/model_evaluation.html) |
| 📄 | PR vs ROC | [Davis & Goadrich](https://www.biostat.wisc.edu/~page/rocpr.pdf) |
| 📄 | Class Imbalance | [He & Garcia](https://ieeexplore.ieee.org/document/5128907) |

---

⬅️ [Back: Representation](../04_representation/) | ➡️ [Next: Ensemble Methods](../06_ensemble_methods/)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<!-- Navigation -->
<p align="center">
  <a href="../04_representation/">⬅️ Prev: Representation</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../">📚 ML Theory</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../05_risk_minimization/">Next: Risk Minimization ➡️</a>
</p>

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=80&section=footer" width="100%"/>
</p>
