import torch
import numpy as np
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

class CrownRunner:
    def __init__(self, loader):
        self.loader = loader
        self.config = loader.config
        # Move model to eval mode for bound propagation
        self.model = loader.model.eval()

    def __call__(self, loader):
        """
        Main execution entry point to match the MILP runner interface.
        """
        spec = loader.get_spec() # Assuming loader provides constraints A, b

        # 1. Prepare Inputs & Perturbations
        # CROWN needs a sample input (the center of the box we are checking)
        input_data = torch.tensor([spec.input_center], dtype=torch.float32)
        eps = torch.tensor([spec.input_radius], dtype=torch.float32)
        
        # Define the 'epsilon' (bounds) for the inputs
        # eps is the distance from the center to the bounds
        ptb = PerturbationLpNorm(norm=np.inf, eps=eps)
        x = BoundedTensor(input_data, ptb)

        # 2. Wrap the model for auto_LiRPA
        # 'method' can be 'CROWN', 'IBP', or 'CROWN-IBP'
        bounded_model = BoundedModule(self.model, input_data)
    
        # 3. Compute Bounds
        # 1. Get NN Bounds (4 outputs)
        lb, ub = bounded_model.compute_bounds(x=x, method="CROWN") 
        
        # 5. Now multiply by Matrix A (5x6)
        A = torch.tensor(spec.constraints_A, dtype=torch.float32)
        b = torch.tensor(spec.b_static, dtype=torch.float32)
        
        # 2. Identify indices from spec
        in_idx = spec.input_indices   # e.g., [0, 1, 2, 3, 4, 5]
        out_idx = spec.output_indices # e.g., [2, 3, 4, 5]
        
        # 1. Get the total number of variables from Matrix A (should be 6)
        A = torch.tensor(spec.constraints_A, dtype=torch.float32)
        num_total_vars = A.shape[1] 

        # 2. Create empty full-size tensors
        full_lb = torch.zeros(num_total_vars, dtype=torch.float32)
        full_ub = torch.zeros(num_total_vars, dtype=torch.float32)

        # 3. Fill in the INPUTS (size 2) into the correct slots
        # spec.input_center usually contains the values for indices [0, 1]
        input_vals = torch.tensor(spec.input_center, dtype=torch.float32)
        full_lb[spec.input_indices] = input_vals
        full_ub[spec.input_indices] = input_vals

        # 4. Fill in the NN OUTPUTS (size 4) into indices [2, 3, 4, 5]
        # This will NO LONGER crash because full_lb is size 6
        full_lb[out_idx] = lb.flatten()
        full_ub[out_idx] = ub.flatten()
        
        max_violation = self._calculate_worst_violation(full_lb, full_ub, A, b)
        
        
        # 1. Initialize full_center to the correct TOTAL size (e.g., 6)
        full_center = torch.zeros(num_total_vars, dtype=torch.float32)

        # 2. Fill inputs (constants)
        full_center[spec.input_indices] = torch.tensor(spec.input_center, dtype=torch.float32)

        # 3. Fill NN outputs (prediction centers)
        nn_center_vals = (lb + ub) / 2
        full_center[out_idx] = nn_center_vals.flatten()

        # 4. Calculate violations for the report
        # actual_ax is (5x6) @ (6x1) = (5x1)
        actual_ax = torch.matmul(A, full_center.unsqueeze(1)).flatten()
        violation_per_row = actual_ax - b.flatten()
        
        # --- Standardized Output Printing (Same as MILP) ---
        print(f"\n{'-'*20} CROWN Summary {'-'*20}")
        status_str = "FEASIBLE (Safe)" if max_violation <= 1e-6 else "VIOLATED (Unsafe)"
        print(f"[*] Engine Status: {status_str}")
        print(f"[*] Max Violation: {max_violation.item():.6f}")
        
        # Print first few NN output values for a quick sanity check
        print(f"[*] NN Outputs (Center): {nn_center_vals.flatten().detach().cpu().numpy()[:5]}")
        
        # Find which constraint is the "Worst"
        worst_idx = torch.argmax(violation_per_row)
        print(f"[*] Worst Constraint Index: {worst_idx.item()}")
        print(f"{'-'*55}\n")
        # --------------------------------------------------


        # 5. Return standardized results dictionary
        return {
            "status": "Success",
            "max_gap": float(max_violation.detach().cpu()), # Added detach() here
            "max_violation": float(max_violation.detach().cpu()),
            "violation_per_row": violation_per_row.detach().cpu().numpy().tolist(),
            "nn_vals": nn_center_vals.detach().cpu().numpy().flatten().tolist(),
            "at_input_val": input_data.detach().cpu().numpy().flatten().tolist(),
            "engine": "CROWN"
        }

    def _calculate_worst_violation(self, lb, ub, A, b):
        """
        Calculates the maximum possible value of Ax - b given output bounds [lb, ub].
        lb, ub: shape (1, num_outputs) or (num_outputs,)
        A: shape (num_constraints, num_outputs)
        b: shape (num_constraints,)
        """
        # Ensure lb and ub are 2D for consistent broadcasting: (1, num_outputs)
        lb = lb.view(1, -1)
        ub = ub.view(1, -1)

        # For each constraint (row in A):
        # If A_ij > 0, the maximum is reached at ub_j
        # If A_ij < 0, the maximum is reached at lb_j
        
        # We use torch.clamp to isolate positive and negative parts of A
        # A_plus contains only positive entries of A, A_minus only negative
        A_plus = torch.clamp(A, min=0)
        A_minus = torch.clamp(A, max=0)

        # Max(Ax) = A_plus @ ub + A_minus @ lb
        # Use .T on bounds to align for matrix multiplication (num_constraints, 1)
        max_ax = torch.matmul(A_plus, ub.T) + torch.matmul(A_minus, lb.T)

        # Subtract b (ensure b is reshaped to match max_ax)
        violation_per_row = max_ax.flatten() - b.flatten()

        # The worst violation is the maximum across all constraints
        return torch.max(violation_per_row)