# 📄 Configuration & Model Metadata

The `config.yaml` file is the **Single Source of Truth** for the verification engine. It bridges the gap between the Machine Learning model (weights/biases) and the Mathematical Optimization problem (constraints/objectives) and it specifies the verification objective.

---

## 1. Problem Class: Linear Programming (LP)

It is important to note that the current `verification_spec` is designed specifically for **Linear Programming (LP)** structures. 

* **Why this matters:** Different optimization classes require different mathematical representations. For example, a **Quadratic Program (QP)** would require an additional Hessian matrix ($Q$) for the objective, while **Non-Linear Programs (NLP)** might require functional expressions.
* **Standard Form:** The verifier assumes the physics follow the canonical LP form:
  $$\min c^T x \quad \text{s.t.} \quad Ax \le b, \quad x \ge 0$$



---

## 2. Model Metadata (`model_meta`)

This section defines the identity and architecture of the neural network.
* **`pclass`**: Specifies the problem class, currently set to `optimization`. Other choices are for example control, or forecast.
* **`ptype`**: Specifies the type of problem. Currently set to `lp` (linear program). Other options are `qp` (quadratic program).
* **`architecture`**: Currently set to `feedforward`. this specifies the neural network architecture. 
* **`activation`**: Set to `relu`. This tells the verifier to use **Big-M** or **Indicator Constraints** to linearize the non-linear activation functions for the MILP solver.
* **`check`**: Specifies whether you want to check worst-case constraint violations, or the worst-case distance (`distance`) to the optimal solution. Currently set to `constraint`.


---

## 3. Verification Specification (`verification_spec`)

This section defines the "Rules of the Game" that the Neural Network must follow.

### A. Input Space & Indices
* **`input_bounds`**: Defines the valid search space (e.g., $0$ to $5$) for the input variables. The verifier looks for violations **only** within this range.
* **`input_indices`**: The indices in the state vector $x$ that represent independent parameters (e.g., Demand/Load).
* **`output_indices`**: The indices predicted by the NN (e.g., Generator Dispatch).

### B. Physical Constraints (Matrix $A$ and Vector $b$)
These define the boundaries of the feasible region.
* **Matrix A**: Each row is a linear inequality. 
* **The Balance Constraint**: The final row in your $A$ matrix (`[1, 1, -1, -1, -1, -1]`) represents the physical law of supply and demand: 
  $$\sum \text{Inputs} - \sum \text{Outputs} \le 0 \implies \sum \text{Outputs} \ge \sum \text{Inputs}$$
* **`b_static`**: The right-hand side limits for the inequalities.

### C. Objective Coefficients (`objective_c`)
The "Cost" of each variable. In this LP proxy:
* **Input costs** are `0.0` (we don't pay for the demand itself).
* **Output costs** are positive (we pay for the generation used to satisfy the demand).



---

## 4. Usage in the Pipeline

1. **The Trainer** uses this file to understand the architecture it needs to build.
2. **The Verifier** uses the `layers` to build the NN inside the optimization model and the `verification_spec` to build the "True Optimal" baseline using KKT conditions.

<details>
<summary>🔍 Click to view a sample Config structure</summary>

```yaml
model_meta:
  name: lp_proxy
  pclass: optimization
  ptype: lp
  architecture: feedforward
  activation: relu
  check: distance
layers:
  - weights: [[...]]
    biases: [...]
verification_spec:
  input_bounds: [{min: 0, max: 5}, {min: 0, max: 5}]
  constraints:
    A: [[...]]
    b_static: [...]
  objective_c: [0.0, 0.0, 0.5, 0.2, 0.9, 0.3]