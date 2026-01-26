import numpy as np
import pandas as pd
from scipy.optimize import linprog
import yaml
from pathlib import Path

# Define the base directory where the script resides
BASE_DIR = Path(__file__).resolve().parent

def generate_agnostic_lp(n_vars=6, n_cons=4, num_samples=5000):
    """
    n_vars: Total variables in the canonical physics (x_full)
    """
    # 1. Generate Static Physics (A, b)
    A = np.random.uniform(-1.0, 1.0, size=(n_cons, n_vars))
    b = np.random.uniform(5.0, 10.0, size=n_cons)
    
    # Define the Mapping Logic early so we can mask the objective
    input_idxs = [0, 1]
    output_idxs = [2, 3, 4, 5]
    
    # --- ADDING THE BALANCE CONSTRAINT ---
    # We want: sum(x_out) >= sum(x_in)  =>  sum(x_in) - sum(x_out) <= 0
    balance_row = np.zeros((1, n_vars))
    balance_row[0, input_idxs] = 1.0    # Load side
    balance_row[0, output_idxs] = -1.0  # Generation side
    
    A = np.vstack([A, balance_row])    # Add row to A
    b = np.append(b, 0.0)              # Add 0 to b
    # -------------------------------------

    c = np.zeros(n_vars)
    # Assign random costs ONLY to the output variables
    c[output_idxs] = np.random.uniform(0.1, 1.0, size=len(output_idxs))

    data = []

    for i in range(num_samples):
        # Sample random values for the 'input' subset of x (e.g. Demand levels)
        x_in = np.random.uniform(1, 5, size=len(input_idxs))
        
        # Physics Step: A_in * x_in + A_out * x_out <= b
        # Rearranged:  A_out * x_out <= b - (A_in * x_in)
        A_input = A[:, input_idxs]
        A_output = A[:, output_idxs]
        b_dynamic = b - (A_input @ x_in)
        
        # Objective for the solver: minimize c_output^T * x_out
        c_output = c[output_idxs]
        
        # Solving the local LP for this specific input scenario
        res = linprog(c_output, A_ub=A_output, b_ub=b_dynamic, bounds=(0, 10), method='highs')
        
        if res.success:
            # Store [Inputs, Optimal Outputs]
            data.append(np.concatenate([x_in, res.x]))
        else:
            print("Infeasible!")

    # 3. Save Data and Metadata
    df = pd.DataFrame(data, columns=[f'in_{i}' for i in input_idxs] + [f'out_{j}' for j in output_idxs])
    df.to_csv(BASE_DIR / "lp_data.csv", index=False)

    metadata = {
        "physics": {
            "A": A.tolist(), 
            "b": b.tolist(), 
            "c": c.tolist() # This now reflects the zeroed-out input costs
        },
        "mapping": {
            "input_indices": input_idxs, 
            "output_indices": output_idxs
        },
        "input_bounds": [{"min": 0, "max": 5} for _ in input_idxs]
    }
    
    with open(BASE_DIR / "lp_metadata.yaml", "w") as f:
        yaml.dump(metadata, f, sort_keys=False)
    
    print(f"Successfully generated {len(data)} feasible samples.")

if __name__ == "__main__":
    generate_agnostic_lp()