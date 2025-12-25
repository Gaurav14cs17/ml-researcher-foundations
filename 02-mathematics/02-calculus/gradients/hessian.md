<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=00C853&height=120&section=header&text=Hessian%20Matrix&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-02-00C853?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 📐 Definition

```
For f: ℝⁿ → ℝ

H = ∇²f ∈ ℝⁿˣⁿ

H[i,j] = ∂²f/∂xᵢ∂xⱼ

Symmetric if f is C² (continuous second derivatives)
```

---

## 🔑 What It Tells Us

```
At critical point (∇f = 0):
• H ≻ 0 (positive definite): Local minimum
• H ≺ 0 (negative definite): Local maximum
• H indefinite: Saddle point

Eigenvalues of H = Curvature in each direction
```

---

## 📊 Condition Number

```
κ = λ_max / λ_min

High κ: Ill-conditioned, GD struggles
Low κ: Well-conditioned, fast convergence
```

---

## 💻 Code

```python
import torch

# Function f: R² → R
def f(x):
    return x[0]**2 + 10*x[1]**2  # Ill-conditioned!

x = torch.tensor([1., 1.], requires_grad=True)

# Compute Hessian
hessian = torch.autograd.functional.hessian(f, x)
print(hessian)
# [[2., 0.],
#  [0., 20.]]

# Condition number
eigvals = torch.linalg.eigvalsh(hessian)
condition_number = eigvals.max() / eigvals.min()  # 10
```

---

## ⚠️ Cost

```
Computing full Hessian: O(n²) space, O(n²) time
Hessian-vector products: O(n) time, O(n) space

For deep learning: Full Hessian is impractical
Use: Adam (diagonal approx), K-FAC (block approx)
```

---

---

➡️ [Next: Jacobian](./jacobian.md)

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=00C853&height=80&section=footer" width="100%"/>
</p>
