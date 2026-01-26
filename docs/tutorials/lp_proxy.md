# ⚛️ Neural Network LP Proxy: Logic & Architecture

This repository implements a **Neural Network Surrogate** for Linear Programming (LP) problems. It uses a **Partial Mapping** approach designed for physical systems like Power Grids, where a network learns to predict optimal decision variables while adhering to a static set of physical constraints.

---

## 1. Problem Formulation

The system is modeled as a **Canonical Linear Program**. The goal is to minimize the operating cost of a system subject to linear inequalities.

### The Math
$$
\begin{aligned}
\min_{x} \quad & c^T x \\
\text{s.t.} \quad & A x \le b \\
& x \ge 0
\end{aligned}
$$

### Agnostic Variable Mapping
We partition the global state vector $x$ into two distinct subsets:
* **Inputs ($x_{in}$):** Independent system parameters (e.g., nodal demand, weather data).
* **Outputs ($x_{out}$):** Decision variables predicted by the NN (e.g., generator setpoints, battery dispatch).

The Neural Network functions as the mapping: $f_{\theta}(x_{in}) \to x_{out}$.

---

## 2. The Balance Constraint (Physics)

In many cost-minimization problems, the mathematical "cheapest" solution is simply to do nothing ($x=0$). To prevent this and simulate a real-world system, we enforce a **Demand Satisfaction** constraint:



> **Equation:** $\sum x_{out} \ge \sum x_{in}$

This forces the Neural Network to learn the "Load Following" logic—it must increase output variables whenever the input demand increases. This is injected into the global $A$ matrix and $b$ vector before training and verification.

---

## 3. Formal Verification Pipeline

This project goes beyond simple testing by using **Mixed-Integer Linear Programming (MILP)** to formally verify the model's behavior against its physical constraints.

### 🛡️ Feasibility Verification
We search the entire continuous input space to find the "Worst Case" input $x_{in}$ that causes the Neural Network to violate the system constraints ($Ax \le b$).

### 📉 Optimality (Regret) Analysis
We calculate the **Proven Max Regret**. This uses **KKT (Karush-Kuhn-Tucker) Conditions** to find the maximum possible gap between the cost of the NN's prediction and the true mathematical optimal solution.



---

## 4. Project Structure & Workflow

1.  **`generate_data.py`**: Creates the $A$, $b$, and $c$ matrices. It solves the LP for thousands of random input scenarios to create a training dataset.
2.  **`train_nn.py`**: A PyTorch implementation of a Feedforward ReLU network. It maps inputs to outputs using the generated data.
3.  **`verify.py`**: The verification engine. It translates the NN weights and the LP physics into a Pyomo model to prove the optimality gap.

---

## 5. Quick Start

```bash
# 1. Generate the LP physics and data
python usecase/optimization/lp/generate.py

# 2. Train the surrogate model
python usecase/optimization/lp/train.py

# 3. Prove the Max Regret Gap
python main.py usecase/optimization/lp/config.yaml