# SVM Margin

> **Finding the widest separating hyperplane**

---

## 📐 Definition

```
For hyperplane: wᵀx + b = 0

Functional margin: y(wᵀx + b)
Geometric margin: y(wᵀx + b) / ||w||

"Distance from point to decision boundary"
```

---

## 🎯 Maximum Margin

```
Optimization:
max_{w,b}  min_i [y_i(wᵀx_i + b) / ||w||]

Equivalent to:
min_{w,b}  ½||w||²
s.t.       y_i(wᵀx_i + b) ≥ 1  ∀i
```

---

## 📊 Visual

```
        ○ ○                    
      ○ ○ ○        margin      
        ○      ←------→        
    - - - - - - - - - - -   w·x + b = 1
    =================        w·x + b = 0
    - - - - - - - - - - -   w·x + b = -1
          ●                    
        ● ● ●                  
      ● ● ●                    
```

---

## 🔑 Why Maximize Margin?

```
1. Robustness: Larger margin = more room for error
2. Generalization: VC dimension scales as 1/margin²
3. Theory: Maximizes distance to worst-case points
```

---

## 💻 Code

```python
from sklearn.svm import SVC

# Hard margin SVM (only for linearly separable)
svm = SVC(kernel='linear', C=float('inf'))

# Soft margin SVM (allows violations)
svm = SVC(kernel='linear', C=1.0)

svm.fit(X_train, y_train)

# Access decision boundary
w = svm.coef_[0]
b = svm.intercept_[0]
margin = 2 / np.linalg.norm(w)
```

---

<- [Back](./README.md)


