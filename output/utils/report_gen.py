import nbformat as nbf
import os
from datetime import datetime
import numpy as np

def generate_report(results, config, output_path="output/"):
    """
    Unified generator that handles both 'distance' (regret) and 'constraint' checks.
    """
    nb = nbf.v4.new_notebook()
    check_type = config['model_meta'].get('check', 'distance')
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # --- 1. Header Section ---
    title = "# 🛡️ Sub-Optimality Analysis Report" if check_type == "distance" else "# ⚠️ Constraint Violation Report"
    nb['cells'].append(nbf.v4.new_markdown_cell(f"{title}\n**Generated on:** {timestamp}"))
    nb['cells'].append(nbf.v4.new_markdown_cell(
        f"## ⚙️ System Configuration\n"
        f"Model: `{config['model_meta']['name']}` | "
        f"Type: `{config['model_meta']['ptype'].upper()}` | "
        f"Check: `{check_type}`"
    ))

    # --- 2. Conditional Metrics Section ---
    if check_type == "distance":
        gap = results.get('optimality_gap', 0.0)
        summary_md = (
            "### 📊 Optimality Summary\n"
            "| Metric | Value |\n"
            "| :--- | :--- |\n"
            f"| **Status** | {'✅ PASSED' if gap < 1e-4 else '❌ FAILED'} |\n"
            f"| **NN Optimal Cost** | {results.get('nn_total_cost', 0.0):.6f} |\n"
            f"| **True Optimal Cost** | {results.get('true_optimal_cost', 0.0):.6f} |\n"
            f"| **Max Sub-Optimality (Gap)** | {gap:.6f} |\n"
        )
    else:
        max_viol = results.get('max_violation', 0.0)
        summary_md = (
            "### 🛡️ Safety Summary\n"
            "| Metric | Value |\n"
            "| :--- | :--- |\n"
            f"| **Status** | {'✅ FEASIBLE' if max_viol < 1e-4 else '❌ VIOLATED'} |\n"
            f"| **Worst Violation** | {max_viol:.6f} |\n"
        )
        
        A = np.array(results.get('constraints_A', []))
        b = np.array(results.get('b_static', []))
        x_full = np.array(results.get('full_x_vector', []))
        
        # Add the Constraint Table
        summary_md += "#### 📏 Detailed Constraint Analysis\n"
        summary_md += "| Row | Status | LHS Value | Limit (b) | Violation |\n"
        summary_md += "| :--- | :--- | :--- | :--- | :--- |\n"
        
        for i in range(len(A)):
            lhs_val = np.dot(A[i], x_full)
            viol = lhs_val - b[i]
            status_icon = "❌" if viol > 1e-5 else "✅"
            summary_md += f"| {i} | {status_icon} | {lhs_val:.4f} | {b[i]:.4f} | {viol:.4f} |\n"
    
    nb['cells'].append(nbf.v4.new_markdown_cell(summary_md))

    # --- 3. Conditional Visualizations ---
    if check_type == "distance":
        viz_code = f"""
import matplotlib.pyplot as plt
import numpy as np

indices = {list(range(len(results.get('nn_vals', []))))}
nn_vals = {results.get('nn_vals', [])}
opt_vals = {results.get('opt_vals', [])}

plt.figure(figsize=(10, 5))
plt.bar(np.array(indices) - 0.2, nn_vals, width=0.4, label='NN Prediction', color='skyblue')
plt.bar(np.array(indices) + 0.2, opt_vals, width=0.4, label='True Optimal', color='salmon')
plt.title('NN Prediction vs. True Optimality')
plt.legend(); plt.show()
        """
    else:
        viz_code = f"""
import matplotlib.pyplot as plt
violations = {results.get('violation_per_row', [])}
plt.figure(figsize=(10, 5))
plt.bar(range(len(violations)), violations, color=['red' if v > 1e-4 else 'green' for v in violations])
plt.axhline(y=0, color='black')
plt.title('Constraint Violations (Positive = Infeasible)')
plt.show()
        """
    
    nb['cells'].append(nbf.v4.new_markdown_cell("## 📈 Analysis Visualization"))
    nb['cells'].append(nbf.v4.new_code_cell(viz_code.strip()))

    # --- 4. Save Logic ---
    filename = f"report_{check_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.ipynb"
    os.makedirs(output_path, exist_ok=True)
    full_path = os.path.join(output_path, filename)
    
    with open(full_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
        
    return full_path