# Physics-Informed Neural Network for Voltage Magnitude Estimation  
*(IEEE-33 Bus Distribution System)*

---

## 📌 Project Overview

This project implements a **Physics-Informed Neural Network (PINN)** to estimate **bus voltage magnitudes** in a **radial distribution power system** using **active and reactive power injections** as inputs. The study is carried out on the **standard IEEE-33 bus distribution system** and compares a purely data-driven baseline neural network with a physics-informed learning approach.

The motivation behind this work is to enhance the **physical consistency and reliability** of machine learning-based voltage estimation models, especially when **complete power system information (topology, voltage angles)** is unavailable.

---

## ⚡ Problem Statement

Accurate voltage magnitude estimation is critical for the operation, monitoring, and planning of distribution power systems. Traditional power-flow techniques such as Newton–Raphson and Gauss–Seidel require complete system parameters, including:

- Network topology  
- Line parameters  
- Voltage angles  

However, real-world datasets often lack such information. Purely data-driven machine learning models can achieve high numerical accuracy but may violate fundamental power system laws.

**Objective:**  
To develop a **Physics-Informed Neural Network (PINN)** that predicts voltage magnitudes while enforcing **power-system physics** through a simplified reactive power balance constraint.

---

## 🏗️ System Description

- **Test System:** IEEE-33 Bus Distribution System (Baran–Wu)
- **Network Type:** Radial distribution feeder
- **Buses:** 33
- **Lines:** 32
- **Slack Bus:** Bus 1
- **Voltage Angles:** Not available / not modeled

---

## 📊 Dataset Description

- **Total Samples:** 35,135  
- **Nature:** Independent steady-state operating points  
- **Original Features (per bus):**
  - Active Power (P)
  - Reactive Power (Q)
  - Load components (PL, QL)
  - Voltage Magnitude (V)

### Final Dataset Used
- **Inputs:**  
  - Active Power: \( P_0 \dots P_{32} \)  
  - Reactive Power: \( Q_0 \dots Q_{32} \)  
  → **66 input features**

- **Outputs:**  
  - Voltage Magnitudes: \( V_0 \dots V_{32} \)  
  → **33 output targets**

- **Final Shape:** `(35135, 99)`

> Note: PL and QL were intentionally removed to maintain consistency.

---

## 🧠 Modeling Approach

### 1️⃣ Baseline Machine Learning Model

- **Model Type:** Multi-output Artificial Neural Network (ANN)
- **Inputs:** 66 (P, Q)
- **Outputs:** 33 (Voltage magnitudes)
- **Loss Function:** Mean Squared Error (MSE)
- **Optimizer:** Adam
- **Activation Functions:** ReLU (hidden layers), Linear (output)

**Observation:**  
The baseline model achieves very high numerical accuracy due to the low variance of voltage magnitudes but lacks physical interpretability.

---

### 2️⃣ Physics-Informed Neural Network (PINN)

## 🔬 Physics Constraint Used (PINN Formulation)

To embed power system physics into the learning process, a **voltage-only physics constraint** is incorporated into the Physics-Informed Neural Network (PINN).  
Since voltage angle information is unavailable, a simplified **reactive power balance constraint** based on the imaginary part of the Y-bus matrix is adopted.

### 🔹 Reactive Power Balance Constraint

The reactive power relationship is expressed as:

$$
\mathbf{Q} + \mathbf{V} \odot (\mathbf{B}\mathbf{V}) \approx \mathbf{0}
$$

where:

- $\mathbf{Q} \in \mathbb{R}^{33}$  
  Reactive power injection vector  
- $\mathbf{V} \in \mathbb{R}^{33}$  
  Voltage magnitude vector predicted by the neural network  
- $\mathbf{B} \in \mathbb{R}^{33 \times 33}$  
  Imaginary part (susceptance matrix) of the Y-bus  
- $\odot$  
  Element-wise (Hadamard) product  

This constraint enforces physical consistency between predicted voltages and reactive power behavior of the distribution network.

---

## ⚙️ PINN Loss Function

The PINN is trained using a **composite loss function** that combines data accuracy with physics enforcement:

$$
\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} \left\| \mathbf{V}_{\text{pred}}^{(i)} - \mathbf{V}_{\text{true}}^{(i)} \right\|_2^2 + \lambda \frac{1}{N} \sum_{i=1}^{N} \left\| \mathbf{Q}^{(i)} + \mathbf{V}_{\text{pred}}^{(i)} \odot \left( \mathbf{B}\mathbf{V}_{\text{pred}}^{(i)} \right) \right\|_2^2
$$

where:

- $N$ — number of training samples  
- $\lambda$ — physics weighting coefficient (typically $\lambda \approx 10^{-3}$)  
- $\mathbf{V}_{\text{pred}}$ — predicted voltage magnitudes  
- $\mathbf{V}_{\text{true}}$ — ground-truth voltage magnitudes  

---

## ✅ Interpretation

- **Data Loss** ensures high numerical accuracy with respect to measured voltages.  
- **Physics Loss** penalizes violations of the reactive power balance constraint.  
- Joint minimization guides the network toward a **physics-consistent solution manifold**, improving reliability and interpretability.

This formulation enables robust voltage magnitude estimation even in the absence of voltage angle information.

## 🔌 Y-Bus Matrix Extraction

- **Tool Used:** `pandapower`
- **Network Model:** Standard IEEE-33 bus (`case33bw`)
- **Process:**
  1. Load IEEE-33 bus network
  2. Run power flow
  3. Extract Y-bus from internal PYPOWER structure
  4. Use only the **imaginary part (B-matrix)**

```python
import pandapower as pp
import pandapower.networks as pn

net = pn.case33bw()
pp.runpp(net, numba=False)

Ybus = net._ppc["internal"]["Ybus"].toarray()
B = Ybus.imag
