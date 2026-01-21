<!-- Navigation -->
<p align="center">
  <a href="../04_regularization/">⬅️ Prev: Regularization</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../">📚 Generalization</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../../03_kernel_methods/">Next: Kernel Methods ➡️</a>
</p>

---

<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=120&section=header&text=VC%20Dimension&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-05-4ECDC4?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/vc-dimension.svg" width="100%">

*Caption: VC dimension measures how many points a hypothesis class can "shatter" (perfectly classify in all possible ways). Linear classifiers in 2D have VC=3 - they can shatter 3 non-collinear points but not 4.*

---

## 📂 Overview

The **Vapnik-Chervonenkis (VC) dimension** is a fundamental measure of the capacity (complexity) of a hypothesis class. It provides distribution-free bounds on generalization error and forms the theoretical foundation of statistical learning theory.

---

## 📐 Formal Definitions

### Shattering

**Definition:** A hypothesis class \(\mathcal{H}\) **shatters** a set of points \(S = \{x_1, \ldots, x_n\}\) if for every possible labeling \((y_1, \ldots, y_n) \in \{-1, +1\}^n\), there exists \(h \in \mathcal{H}\) such that \(h(x_i) = y_i\) for all \(i\).

In other words, \(\mathcal{H}\) can achieve all \(2^n\) possible binary labelings of \(S\).

### VC Dimension

**Definition:** The VC dimension of \(\mathcal{H}\), denoted \(\text{VC}(\mathcal{H})\), is:

$$\text{VC}(\mathcal{H}) = \max\{n : \exists S \text{ with } |S| = n \text{ that } \mathcal{H} \text{ shatters}\}$$

If \(\mathcal{H}\) can shatter arbitrarily large sets, \(\text{VC}(\mathcal{H}) = \infty\).

**Equivalently:**
- \(\text{VC}(\mathcal{H}) \geq d\) if there exists **some** set of \(d\) points that \(\mathcal{H}\) shatters

- \(\text{VC}(\mathcal{H}) < d\) if **no** set of \(d\) points can be shattered by \(\mathcal{H}\)

---

## 📊 Computing VC Dimension: Examples

### Example 1: Linear Classifiers in \(\mathbb{R}^d\)

**Hypothesis class:** \(\mathcal{H} = \{x \mapsto \text{sign}(w^\top x + b) : w \in \mathbb{R}^d, b \in \mathbb{R}\}\)

**Theorem:** \(\text{VC}(\mathcal{H}) = d + 1\)

**Proof of lower bound (\(\geq d+1\)):**

We need to show there exists a set of \(d+1\) points that can be shattered.

Consider points in general position:

$$x_0 = 0, \quad x_i = e_i \text{ for } i = 1, \ldots, d$$

where \(e_i\) is the \(i\)-th standard basis vector.

For any labeling \((y_0, y_1, \ldots, y_d)\), set:

$$w = \sum_{i=1}^d y_i e_i, \quad b = -\frac{1}{2}(1 + y_0)$$

Then:

- \(w^\top x_0 + b = b = -\frac{1}{2}(1 + y_0)\), which has sign \(y_0\)

- \(w^\top x_i + b = y_i + b\), which has sign \(y_i\) for appropriate \(b\)

**Proof of upper bound (\(\leq d+1\)):**

For any \(d+2\) points in \(\mathbb{R}^d\), they are linearly dependent. By Radon's theorem, they can be partitioned into two sets whose convex hulls intersect. This partitioning cannot be achieved by any hyperplane.

### Example 2: Intervals on \(\mathbb{R}\)

**Hypothesis class:** \(\mathcal{H} = \{x \mapsto \mathbb{1}[a \leq x \leq b] : a, b \in \mathbb{R}\}\)

**Theorem:** \(\text{VC}(\mathcal{H}) = 2\)

**Proof:**

*Lower bound:* Two points \(x_1 < x_2\) can be shattered:

- \((+, +)\): interval \([x_1, x_2]\)

- \((-, -)\): interval \([x_1 - 1, x_1 - 0.5]\) (excludes both)

- \((+, -)\): interval \([x_1 - 0.5, (x_1 + x_2)/2]\)

- \((-, +)\): interval \([(x_1 + x_2)/2, x_2 + 0.5]\)

*Upper bound:* For any three points \(x_1 < x_2 < x_3\), the labeling \((+, -, +)\) cannot be achieved by a single interval.

### Example 3: Axis-Aligned Rectangles in \(\mathbb{R}^2\)

**Theorem:** \(\text{VC}(\mathcal{H}) = 4\)

**Proof:**

*Lower bound:* Four points forming a "+" shape can be shattered (each can be included/excluded by adjusting rectangle boundaries).

*Upper bound:* For any 5 points, one is inside the convex hull of the others. The labeling with only that point positive cannot be achieved.

### Example 4: k-Nearest Neighbors

**Theorem:** \(\text{VC}(\mathcal{H}_{k\text{-NN}}) = \infty\)

**Proof:** For any \(n\) points and any labeling, 1-NN memorizes and achieves the labeling. Hence arbitrarily large sets can be shattered.

---

## 📐 The Growth Function and Sauer's Lemma

### Growth Function

**Definition:** The growth function of \(\mathcal{H}\) is:

$$m_\mathcal{H}(n) = \max_{x_1, \ldots, x_n} |\{(h(x_1), \ldots, h(x_n)) : h \in \mathcal{H}\}|$$

This counts the maximum number of distinct classifications on \(n\) points.

**Properties:**
- \(m_\mathcal{H}(n) \leq 2^n\) (at most all labelings)

- If \(\text{VC}(\mathcal{H}) = d\), then \(m_\mathcal{H}(d) = 2^d\) (some set is shattered)

- If \(\text{VC}(\mathcal{H}) = d\), then \(m_\mathcal{H}(n) < 2^n\) for \(n > d\)

### Sauer-Shelah Lemma

**Theorem (Sauer, 1972; Shelah, 1972):** If \(\text{VC}(\mathcal{H}) = d\), then:

$$m_\mathcal{H}(n) \leq \sum_{i=0}^{d} \binom{n}{i} \leq \left(\frac{en}{d}\right)^d$$

**Proof Sketch:**

By induction on \(n + d\). For the base case, if \(d = 0\), then \(m_\mathcal{H}(n) = 1\). For the induction step, partition hypotheses based on behavior on the \(n\)-th point and apply the inductive hypothesis to both subclasses.

**Implication:** For \(n > d\), the growth function is polynomial in \(n\), not exponential!

---

## 📐 Generalization Bounds via VC Dimension

### VC Generalization Theorem

**Theorem:** Let \(\mathcal{H}\) have VC dimension \(d < \infty\). For any distribution \(P\) over \(\mathcal{X} \times \{0, 1\}\), with probability \(\geq 1 - \delta\) over training set \(S\) of size \(n\):

$$\forall h \in \mathcal{H}: \quad R(h) \leq \hat{R}(h) + \sqrt{\frac{8d \ln(en/d) + 8\ln(4/\delta)}{n}}$$

where:

- \(R(h) = \mathbb{E}_{(x,y) \sim P}[\mathbb{1}[h(x) \neq y]]\) is the true risk

- \(\hat{R}(h) = \frac{1}{n}\sum_{i=1}^n \mathbb{1}[h(x_i) \neq y_i]\) is the empirical risk

### Proof Outline

**Step 1: Symmetrization**

The key insight is to relate uniform deviations to the growth function using a "ghost sample" technique.

$$\Pr\left[\sup_{h \in \mathcal{H}} |R(h) - \hat{R}(h)| > \epsilon\right] \leq 2\Pr\left[\sup_{h \in \mathcal{H}} |\hat{R}(h) - \hat{R}'(h)| > \epsilon/2\right]$$

where \(\hat{R}'(h)\) is the empirical risk on an independent "ghost" sample.

**Step 2: Finite Effective Hypothesis Class**

On \(2n\) points, \(\mathcal{H}\) induces at most \(m_\mathcal{H}(2n)\) distinct behaviors.

**Step 3: Union Bound + Hoeffding**

Apply Hoeffding's inequality to each effective hypothesis, then union bound:

$$\Pr\left[\sup_{h \in \mathcal{H}} |R(h) - \hat{R}(h)| > \epsilon\right] \leq 2 m_\mathcal{H}(2n) \cdot e^{-n\epsilon^2/2}$$

**Step 4: Apply Sauer's Lemma**

$$\leq 2 \left(\frac{2en}{d}\right)^d e^{-n\epsilon^2/2}$$

Setting this to \(\delta\) and solving for \(\epsilon\) gives the bound.

### Sample Complexity

**Corollary:** To achieve \(R(\hat{h}) - \hat{R}(\hat{h}) \leq \epsilon\) with probability \(\geq 1 - \delta\):

$$n = O\left(\frac{d + \ln(1/\delta)}{\epsilon^2}\right)$$

**Key insights:**
- Sample complexity is **linear** in VC dimension

- Sample complexity is **quadratic** in \(1/\epsilon\)

- No dependence on the distribution \(P\) (distribution-free bound)

---

## 📊 Common VC Dimensions

| Hypothesis Class | VC Dimension | Notes |
|------------------|--------------|-------|
| Linear classifiers in \(\mathbb{R}^d\) | \(d + 1\) | \(d\) weights + 1 bias |
| Polynomial of degree \(k\) in \(\mathbb{R}^d\) | \(\binom{d+k}{k}\) | Number of monomials |
| Axis-aligned rectangles in \(\mathbb{R}^d\) | \(2d\) | 2 boundaries per dimension |
| Intervals on \(\mathbb{R}\) | 2 | 2 endpoints |
| Half-spaces in \(\mathbb{R}^d\) | \(d + 1\) | Same as linear classifiers |
| Decision stumps | 2 | Single threshold |
| Union of \(k\) intervals | \(2k\) | \(2k\) endpoints |
| k-NN | \(\infty\) | Can memorize |
| Neural network with \(W\) weights | \(O(W \log W)\) | Upper bound |
| Neural network with ReLU | \(O(WL)\) | \(L\) = number of layers |

---

## 💻 Code Examples

### Computing VC Dimension Empirically

```python
import numpy as np
from itertools import product
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

def can_shatter(X, model_class, **model_params):
    """
    Check if a model class can shatter the given points.
    
    Shattering means achieving all 2^n possible labelings.
    
    Args:
        X: numpy array of shape (n, d) - the points to shatter
        model_class: sklearn-compatible classifier
        **model_params: parameters for the model
        
    Returns:
        bool: True if all labelings can be achieved
    """
    n = len(X)
    
    # Generate all 2^n possible labelings
    all_labelings = list(product([-1, 1], repeat=n))
    
    for y in all_labelings:
        y = np.array(y)
        
        try:
            # Try to fit the model
            model = model_class(**model_params)
            model.fit(X, y)
            
            # Check if it achieves this labeling
            pred = model.predict(X)
            
            if not np.array_equal(pred, y):
                return False
                
        except Exception:
            return False
    
    return True

def estimate_vc_dimension(model_class, d, max_points=10, n_trials=100, **model_params):
    """
    Empirically estimate VC dimension by finding largest shatterable set.
    
    Strategy:
    - For each n from 1 to max_points
    - Generate random point configurations
    - Check if any configuration can be shattered
    - VC dimension is largest n where shattering is possible
    
    Args:
        model_class: sklearn-compatible classifier
        d: input dimension
        max_points: maximum number of points to try
        n_trials: number of random configurations to try per n
        **model_params: parameters for the model
        
    Returns:
        int: estimated VC dimension
    """
    vc_dim = 0
    
    for n in range(1, max_points + 1):
        shattered = False
        
        for trial in range(n_trials):
            # Generate random points
            if d == 1:
                X = np.random.randn(n, 1)
            else:
                # Use points in general position
                X = np.random.randn(n, d)
            
            if can_shatter(X, model_class, **model_params):
                shattered = True
                break
        
        if shattered:
            vc_dim = n
            print(f"  n={n}: Can shatter ✓")
        else:
            print(f"  n={n}: Cannot shatter ✗")
            break
    
    return vc_dim

# Example: Verify VC dimension of linear classifiers
print("Estimating VC dimension of linear classifiers in R^2:")
vc = estimate_vc_dimension(
    SVC, 
    d=2, 
    max_points=5,
    kernel='linear', 
    C=1e10  # Hard margin
)
print(f"Estimated VC dimension: {vc}")
print(f"Theoretical: d+1 = 3")

```

### Visualizing Shattering

```python
import matplotlib.pyplot as plt
from itertools import product

def visualize_shattering_2d(points, title="Shattering Visualization"):
    """
    Visualize all possible labelings of 2D points and linear separators.
    """
    n = len(points)
    all_labelings = list(product([-1, 1], repeat=n))
    
    n_cols = 4
    n_rows = (len(all_labelings) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
    axes = axes.flatten()
    
    for idx, labeling in enumerate(all_labelings):
        ax = axes[idx]
        y = np.array(labeling)
        
        # Plot points
        colors = ['red' if yi == -1 else 'blue' for yi in y]
        ax.scatter(points[:, 0], points[:, 1], c=colors, s=100, edgecolors='black')
        
        # Try to find separating hyperplane
        try:
            svm = SVC(kernel='linear', C=1e10)
            svm.fit(points, y)
            
            # Plot decision boundary
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            
            xx = np.linspace(xlim[0]-1, xlim[1]+1, 100)
            yy = np.linspace(ylim[0]-1, ylim[1]+1, 100)
            XX, YY = np.meshgrid(xx, yy)
            Z = svm.decision_function(np.c_[XX.ravel(), YY.ravel()])
            Z = Z.reshape(XX.shape)
            
            ax.contour(XX, YY, Z, levels=[0], colors='green', linewidths=2)
            ax.contourf(XX, YY, Z, levels=[-100, 0, 100], alpha=0.1, colors=['red', 'blue'])
            
            success = np.array_equal(svm.predict(points), y)
            status = "✓" if success else "✗"
        except:
            status = "✗"
        
        ax.set_title(f'{labeling} {status}')
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect('equal')
    
    # Hide unused subplots
    for idx in range(len(all_labelings), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig('shattering_visualization.png', dpi=150)
    plt.show()

# 3 points can be shattered by linear classifier in 2D
points_3 = np.array([[0, 0], [1, 0], [0.5, 0.8]])
visualize_shattering_2d(points_3, "3 Points (Can Shatter - VC Dim ≥ 3)")

# 4 points cannot all be shattered (XOR configuration)
points_4 = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
visualize_shattering_2d(points_4, "4 Points (Cannot Shatter XOR - VC Dim < 4)")

```

### Generalization Bound Computation

```python
import numpy as np

def vc_generalization_bound(n, d, delta=0.05):
    """
    Compute VC generalization bound.
    
    With probability ≥ 1-δ:
    R(h) ≤ R̂(h) + sqrt(8(d*ln(2n/d) + ln(4/δ)) / n)
    
    Args:
        n: sample size
        d: VC dimension
        delta: confidence parameter
        
    Returns:
        float: generalization gap bound
    """
    if n < d:
        return float('inf')
    
    term1 = d * np.log(2 * n / d)
    term2 = np.log(4 / delta)
    
    bound = np.sqrt(8 * (term1 + term2) / n)
    return bound

def required_samples(d, epsilon, delta=0.05):
    """
    Compute sample complexity for desired accuracy.
    
    Returns n such that generalization gap ≤ ε with probability ≥ 1-δ.
    
    n = O(d/ε² + ln(1/δ)/ε²)
    """
    # Solve: sqrt(8(d*ln(2n/d) + ln(4/δ))/n) ≤ ε
    # Approximate: n ≈ 8*(d*ln(2n/d) + ln(4/δ))/ε²
    
    # Use binary search
    n_low, n_high = d, int(1e8)
    
    while n_high - n_low > 1:
        n_mid = (n_low + n_high) // 2
        if vc_generalization_bound(n_mid, d, delta) <= epsilon:
            n_high = n_mid
        else:
            n_low = n_mid
    
    return n_high

# Example computations
print("VC Dimension Analysis")
print("=" * 50)

# For linear classifiers in different dimensions
for d in [3, 10, 100, 1000]:
    print(f"\nVC Dimension d = {d} (linear classifier in R^{d-1}):")
    
    for n in [100, 1000, 10000]:
        bound = vc_generalization_bound(n, d)
        print(f"  n = {n:>5}: Generalization bound = {bound:.4f}")
    
    # Sample complexity for ε = 0.05
    n_required = required_samples(d, epsilon=0.05)
    print(f"  Samples needed for ε=0.05: {n_required}")

```

---

## 🌍 Modern Perspective: Beyond VC Dimension

### The Deep Learning Paradox

Classical VC theory predicts:

- High VC dimension → poor generalization

- Neural networks have VC dim \(\approx O(WL)\) where \(W\) = parameters, \(L\) = layers

**Reality:** Overparameterized networks generalize well despite huge VC dimension!

### Resolution: Other Complexity Measures

| Measure | Idea | Reference |
|---------|------|-----------|
| **Rademacher Complexity** | Data-dependent capacity | Bartlett & Mendelson (2002) |
| **PAC-Bayes Bounds** | Prior/posterior on weights | McAllester (1999) |
| **Flat Minima** | Sharpness of loss landscape | Keskar et al. (2017) |
| **Compression** | Information in weights | Arora et al. (2018) |
| **NTK Theory** | Infinite-width analysis | Jacot et al. (2018) |
| **Implicit Regularization** | SGD bias toward simple solutions | Neyshabur et al. (2017) |

### Rademacher Complexity

**Definition:** The empirical Rademacher complexity is:

$$\hat{\mathcal{R}}_S(\mathcal{H}) = \mathbb{E}_\sigma\left[\sup_{h \in \mathcal{H}} \frac{1}{n}\sum_{i=1}^n \sigma_i h(x_i)\right]$$

where \(\sigma_i \in \{-1, +1\}\) are uniform random signs.

**Advantage:** Data-dependent, can be tighter than VC bounds.

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | Vapnik & Chervonenkis (1971) | Original paper |
| 📖 | Understanding Machine Learning | [Shalev-Shwartz Ch. 6](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/) |
| 📖 | Statistical Learning Theory | Vapnik (1998) |
| 📄 | Rethinking Generalization | [Zhang et al., 2017](https://arxiv.org/abs/1611.03530) |
| 📄 | Reconciling Modern ML and VC | [Nagarajan & Kolter, 2019](https://arxiv.org/abs/1903.07571) |

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<!-- Navigation -->
<p align="center">
  <a href="../04_regularization/">⬅️ Prev: Regularization</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../">📚 Generalization</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../../03_kernel_methods/">Next: Kernel Methods ➡️</a>
</p>

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=80&section=footer" width="100%"/>
</p>
