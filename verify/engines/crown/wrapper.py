import torch
import numpy as np
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from verify.models.lp_augmented_model import LPAugmentedModel

class CrownRunner:
    def __init__(self, loader):
        self.loader = loader
        self.config = loader.config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = loader.model.to(self.device).eval()

    def __call__(self, loader):
        spec = loader.get_spec()
        A_phys = torch.tensor(spec.constraints_A, dtype=torch.float32, device=self.device)
        b_phys = torch.tensor(spec.b_static, dtype=torch.float32, device=self.device)
        
        # 1. Setup Augmented Model
        augmented_model = LPAugmentedModel(self.model, A_phys, b_phys, spec.input_indices, spec.output_indices)
        
        # 2. Prepare Bounded Inputs (Define x_bounded FIRST)
        input_center = torch.tensor(spec.input_center, dtype=torch.float32, device=self.device).unsqueeze(0)
        input_radius = torch.tensor(spec.input_radius, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        ptb = PerturbationLpNorm(norm=np.inf, eps=input_radius)
        x_bounded = BoundedTensor(input_center, ptb).to(self.device)

        # 3. Initialize BoundedModule with the BoundedTensor
        bounded_model = BoundedModule(augmented_model, x_bounded, device=self.device)
        
        # Now these names will be correctly registered in the graph
        output_name = bounded_model.output_name[0]
        input_node_name = bounded_model.input_name[0]

        # 4. Compute Bounds
        lb_viol, ub_viol, A_dict = bounded_model.compute_bounds(
            x=(x_bounded,), 
            method="alpha-CROWN", 
            return_A=True,
            needed_A_dict={output_name: [input_node_name]}
        )
        
        # 4. Extract "Worst-Case" Inputs
        max_violation, worst_row_idx_tensor = torch.max(ub_viol.flatten(), dim=0)
        worst_row_idx = int(worst_row_idx_tensor.item())

        # Extract slopes
        uA = A_dict[output_name][input_node_name]['uA']

        # WRONG HEURISTIC FOR GETTING WORST-CASE INPUTS! 
        if uA.shape[0] > 1:
            # If the first dimension is our constraints (size 5)
            slopes = uA[worst_row_idx, 0, :]
        else:
            # If auto_LiRPA collapsed the constraints or swapped dims
            # We slice the second dimension instead
            slopes = uA[0, worst_row_idx, :]

        # Pick corners based on slope sign
        in_lb = (input_center - input_radius).flatten()
        in_ub = (input_center + input_radius).flatten()
        worst_inputs = torch.where(slopes > 0, in_ub, in_lb)

        # 6. Run specific forward pass for the "Worst-Case" Counter-Example
        with torch.no_grad():
            nn_worst_outputs = self.model(worst_inputs.unsqueeze(0)).flatten()
            nn_nominal_outputs = self.model(input_center).flatten()

        # 7. Reconstruct Full Vectors for Reporting
        full_x_nominal = self._reconstruct_full_vector(spec, nn_nominal_outputs, A_phys.shape[1], input_center.flatten())
        full_x_worst = self._reconstruct_full_vector(spec, nn_worst_outputs, A_phys.shape[1], worst_inputs)

        result = {
            "status": "Success",
            "max_violation": float(max_violation.item()),
            "violation_per_row": ub_viol.flatten().detach().cpu().numpy().tolist(),
            "nn_vals": nn_nominal_outputs.tolist(),
            "full_x_vector": full_x_worst.tolist(), 
            "at_input_val": worst_inputs.tolist(),   
            "engine": "CROWN"
        }
        
        self._print_feasibility_report(spec, result)
        return result

    def _reconstruct_full_vector(self, spec, nn_out, total_dim, inputs):
        """Helper to assemble the 6D vector from specific inputs and outputs."""
        vec = torch.zeros(total_dim, device=self.device)
        vec[spec.input_indices] = inputs
        vec[spec.output_indices] = nn_out
        return vec
    

    def _print_feasibility_report(self, spec, result):
        """Standardized breakdown of constraints and variable contributions."""
        x_full = np.array(result['full_x_vector'])
        A = spec.constraints_A
        b = spec.b_static
        in_idx = spec.input_indices
        out_idx = spec.output_indices
        
        print(f"\n[!] CROWN ENGINE REPORT: Max Violation found: {result['max_violation']:.6f}, NOTE THIS IS A RELAXED UPPER BOUND")
        print("The below analysis is not fully correct yet.")
        print(f"{'='*80}")
        print(f"{'COMPONENT ANALYSIS':<20} | {'INDICES':<15} | {'VALUES'}")
        print(f"{'-'*80}")
        print(f"{'NN Inputs (Fixed)':<20} | {str(in_idx):<15} | {[round(x_full[i], 4) for i in in_idx]}")
        print(f"{'NN Outputs (Pred)':<20} | {str(out_idx):<15} | {[round(x_full[i], 4) for i in out_idx]}")
        
        # Identify auxiliary variables
        all_indices = set(range(len(x_full)))
        aux_idx = sorted(list(all_indices - set(in_idx) - set(out_idx)))
        if aux_idx:
            print(f"{'Aux Variables':<20} | {str(aux_idx):<15} | {[round(x_full[i], 4) for i in aux_idx]}")
        
        print(f"\n{'CONSTRAINT CHECK':<80}")
        print(f"{'-'*80}")
        print(f"{'Row':<5} | {'LHS (Σ A_ij * x_j)':<20} | {'RHS (b)':<12} | {'Violation':<12}")
        print(f"{'-'*80}")

        for i in range(len(A)):
            lhs_val = np.dot(A[i], x_full)
            violation = lhs_val - b[i]
            status = "[FAILED]" if violation > 1e-5 else "[OK]"
            
            print(f"{i:<5} | {lhs_val:<20.6f} | {b[i]:<12.6f} | {violation:<12.6f} {status}")
            
            if violation > 1e-5:
                # Contribution analysis
                contributions = [(j, A[i][j] * x_full[j]) for j in range(len(x_full)) if abs(A[i][j] * x_full[j]) > 1e-4]
                contrib_str = " + ".join([f"({val:.2f} [x{j}])" for j, val in contributions])
                print(f"      └─ Calculation: {contrib_str} = {lhs_val:.4f}")

        print(f"{'='*80}\n")