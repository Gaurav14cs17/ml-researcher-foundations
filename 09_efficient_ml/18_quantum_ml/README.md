<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=1ABC9C&height=120&section=header&text=Lecture%2018%20Quantum%20Machine%20Learning&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-09-1ABC9C?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

# Lecture 18: Quantum Machine Learning

[← Back to Course](../) | [← Previous](../17_efficient_diffusion_models/)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/ml-researcher-foundations/blob/main/09_efficient_ml/18_quantum_ml/demo.ipynb) ← **Try the code!**

---

## 📺 Video Lecture

| Resource | Link |
|----------|------|
| 🎥 **Lecture Video** | [Watch on YouTube](https://www.youtube.com/watch?v=oVeaRWP1DYg&list=PL80kAHvQbh-pT4lCkDT53zT8DKmhE0idB&index=18) |
| 📊 **Slides** | [MIT Course Page](https://hanlab.mit.edu/courses/2023-fall-65940) |
| ⏱️ **Duration** | ~90 minutes |

### 📝 Video Summary

This lecture provides an **introduction to quantum machine learning**:

- **Quantum basics**: Qubits, superposition, entanglement
- **Quantum gates**: Building blocks of quantum computation
- **Variational quantum circuits**: The quantum analog of neural networks
- **Quantum kernels**: Potentially powerful feature spaces
- **Current limitations**: NISQ devices and their constraints
- **Future outlook**: When might quantum ML become practical?

> 💡 *"Quantum computing offers exponential state space—but translating that into practical ML speedups remains an open challenge."* — Prof. Song Han

---

![Overview](overview.png)

## What is Quantum Computing?

Classical computers use **bits** (0 or 1).
Quantum computers use **qubits** (superposition of 0 AND 1).

```
Classical bit: 0 OR 1
Qubit: α|0⟩ + β|1⟩ (both simultaneously!)

n bits: 1 state
n qubits: 2^n states simultaneously
```

---

## 📐 Mathematical Foundations & Proofs

### Qubit State Space

**Single qubit:**

$$
|\psi\rangle = \alpha|0\rangle + \beta|1\rangle
$$

where $\alpha, \beta \in \mathbb{C}$ and $|\alpha|^2 + |\beta|^2 = 1$.

**Bloch sphere representation:**

$$
|\psi\rangle = \cos\frac{\theta}{2}|0\rangle + e^{i\phi}\sin\frac{\theta}{2}|1\rangle
$$

**n-qubit state:**

$$
|\psi\rangle = \sum_{i=0}^{2^n-1} \alpha_i |i\rangle
$$

$2^n$ complex amplitudes — exponential state space!

---

### Quantum Gates

**Pauli-X (NOT):**

$$
X = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}
X|0\rangle = |1\rangle, \quad X|1\rangle = |0\rangle
$$

**Hadamard (superposition):**

$$
H = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}
H|0\rangle = \frac{|0\rangle + |1\rangle}{\sqrt{2}}
$$

**Rotation gates:**

$$
R_Y(\theta) = \begin{pmatrix} \cos\frac{\theta}{2} & -\sin\frac{\theta}{2} \\ \sin\frac{\theta}{2} & \cos\frac{\theta}{2} \end{pmatrix}
$$

**CNOT (entanglement):**

$$
\text{CNOT}|00\rangle = |00\rangle, \quad \text{CNOT}|10\rangle = |11\rangle
$$

---

### Variational Quantum Circuit (VQC)

**Quantum neural network:**

$$
|\psi_{out}\rangle = U(\theta) |0\rangle^{\otimes n}
$$

where $U(\theta)$ is a parameterized unitary:

$$
U(\theta) = \prod_{l=1}^{L} \left( \prod_{i=1}^{n} R_Y(\theta_{l,i}) \cdot \text{Entangle} \right)
$$

**Measurement:**

$$
\langle O \rangle = \langle\psi_{out}| O |\psi_{out}\rangle
$$

Expectation value of observable $O$.

---

### Quantum Gradient (Parameter Shift Rule)

**Gradient of expectation:**

$$
\frac{\partial \langle O \rangle}{\partial \theta} = \frac{\langle O \rangle_{\theta+\pi/2} - \langle O \rangle_{\theta-\pi/2}}{2}
$$

**Proof:**
For $R_Y(\theta)$ rotation:

$$
\frac{\partial}{\partial \theta} e^{-i\theta Y/2} = \frac{e^{-i(\theta+\pi/2)Y/2} - e^{-i(\theta-\pi/2)Y/2}}{2}
$$

Allows exact gradients via circuit evaluation!

---

### Quantum Kernels

**Classical kernel:**

$$
k(x, y) = \phi(x)^T \phi(y)
$$

**Quantum kernel:**

$$
k(x, y) = |\langle\phi(x)|\phi(y)\rangle|^2
$$

where $|\phi(x)\rangle = U(x)|0\rangle^{\otimes n}$.

**Potential advantage:** $|\phi\rangle$ can represent exponentially complex features.

**Computation:**
1. Prepare $|\phi(x)\rangle$
2. Apply $U^\dagger(y)$
3. Measure probability of $|0\rangle^{\otimes n}$

$$
k(x,y) = |\langle 0^n | U^\dagger(y) U(x) | 0^n \rangle|^2
$$

---

### Barren Plateaus

**Problem:** Random VQCs have vanishing gradients.

**Theorem (McClean et al., 2018):**
For random circuit $U$ with $n$ qubits:

$$
\text{Var}\left[\frac{\partial \langle O \rangle}{\partial \theta}\right] \leq O(2^{-n})
$$

**Consequence:** Gradients vanish exponentially with circuit depth/width.

**Mitigation:**
- Shallow circuits
- Problem-specific ansätze
- Layer-wise training

---

### Quantum Advantage Conditions

For quantum speedup:
1. **Problem structure:** Must exploit quantum parallelism
2. **Data loading:** Encoding classical data is expensive $O(N)$
3. **Readout:** Need efficient way to extract answer

**Current status:**
- Theoretical speedups: Yes (some problems)
- Practical ML advantage: Not yet demonstrated

---

## 🧮 Key Derivations

### Expressibility vs Trainability

**More expressive circuit:**
- Can represent more functions
- Harder to train (barren plateaus)

**Trade-off:**

$$
\text{Expressibility} \times \text{Trainability} \leq \text{constant}
$$

For practical QML, need structured circuits matched to problem.

---

### NISQ Limitations

**Current devices (NISQ = Noisy Intermediate-Scale Quantum):**
- 50-1000 qubits
- ~0.1-1% error per gate
- ~100-1000 gate depth before decoherence

**Implications:**
- Can't run deep circuits
- Error correction not yet available
- Limited to variational algorithms

---

### Hybrid Classical-Quantum

**Practical approach:**
```
Classical preprocessing → Quantum circuit → Classical postprocessing
```

Quantum computer handles expensive subroutine; classical handles rest.

**Examples:**
- VQE: Quantum evaluates energy, classical optimizes
- QAOA: Quantum samples, classical analyzes

---

## 🎯 Where Used

| Concept | Applications |
|---------|-------------|
| Variational Circuits | Quantum neural networks |
| Quantum Kernels | Quantum SVM |
| QAOA | Combinatorial optimization |
| Hybrid Models | Quantum-classical learning |

---

## 🗺️ Navigation

| ⬅️ Previous | 🏠 Home | ➡️ Next |
|:-----------:|:-------:|:-------:|
| [← Efficient Diffusion](../17_efficient_diffusion_models/README.md) | [Efficient ML](../README.md) | [🎉 Course Complete!](../README.md) |

---

## 📚 References

| Type | Resource | Link |
|------|----------|------|
| 📄 | Quantum ML Survey | [arXiv](https://arxiv.org/abs/1611.09347) |
| 📄 | VQE | [arXiv](https://arxiv.org/abs/1304.3061) |
| 📄 | QAOA | [arXiv](https://arxiv.org/abs/1411.4028) |
| 📄 | Barren Plateaus | [arXiv](https://arxiv.org/abs/1803.11173) |
| 🌐 | IBM Quantum | [Website](https://quantum-computing.ibm.com/) |
| 🌐 | PennyLane | [Website](https://pennylane.ai/qml/) |
| 🎥 | MIT 6.5940 TinyML | [Course](https://hanlab.mit.edu/courses/2024-fall-65940) |

---

## Summary

| Aspect | Status |
|--------|--------|
| Theory | Promising |
| Hardware | Improving rapidly |
| Practical ML advantage | Not yet demonstrated |
| Worth learning | Yes, for the future |

> "Quantum computing for ML is like fusion power—tremendous potential, but practical applications remain years away."

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=1ABC9C&height=80&section=footer" width="100%"/>
</p>
