import torch
import numpy as np
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from verify.models.lp_augmented_model import LPAugmentedModel
from verify.engines.crown.spec_bounds import pgd_minimize

class CrownRunner:
    def __init__(self, loader):
        self.loader = loader
        self.config = loader.config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = loader.model.to(self.device).eval()


    #def __call__(self, loader):
    def verify_lp_feasibility(self, spec, input_bounds=None):
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
        
        # 4. Certified bound: no input in the box violates any row by more than max_violation
        max_violation, worst_row_idx_tensor = torch.max(ub_viol.flatten(), dim=0)
        worst_row_idx = int(worst_row_idx_tensor.item())

        # 5. Counter-example candidate. The box corner maximizing the row's linear *upper bound* (uA has shape
        #    (batch, n_rows, n_in)) is only a heuristic start: refine it with PGD on the true network and report the
        #    violation it actually attains, which is a real lower bound of the worst case.
        slopes = A_dict[output_name][input_node_name]['uA'][0, worst_row_idx].flatten()
        in_lb = (input_center - input_radius).flatten()
        in_ub = (input_center + input_radius).flatten()
        corner = torch.where(slopes > 0, in_ub, in_lb).unsqueeze(0)
        worst_inputs = pgd_minimize(augmented_model, lambda r: -r.max(dim=1).values,
                                    in_lb.unsqueeze(0), in_ub.unsqueeze(0), corner, 50, 0.1, None).flatten()

        # 6. Run specific forward pass for the "Worst-Case" Counter-Example
        with torch.no_grad():
            attained_violation = float(augmented_model(worst_inputs.unsqueeze(0)).max().item())
            nn_worst_outputs = self.model(worst_inputs.unsqueeze(0)).flatten()
            nn_nominal_outputs = self.model(input_center).flatten()

        # 7. Reconstruct Full Vectors for Reporting
        full_x_nominal = self._reconstruct_full_vector(spec, nn_nominal_outputs, A_phys.shape[1], input_center.flatten())
        full_x_worst = self._reconstruct_full_vector(spec, nn_worst_outputs, A_phys.shape[1], worst_inputs)

        # Keep them as numpy arrays for the internal reporter
        result = {
            "status": "Success",
            "max_violation": float(max_violation.item()),  # certified upper bound of the worst violation
            "attained_violation": attained_violation,       # violation reached at at_input_val (a real point)
            "worst_row_idx": worst_row_idx,
            "violation_per_row": ub_viol.flatten().detach().cpu().numpy(), # Array
            "nn_vals": nn_nominal_outputs.detach().cpu().numpy(),           # Array
            "full_x_vector": full_x_worst.detach().cpu().numpy(),         # Array
            "at_input_val": worst_inputs.detach().cpu().numpy(),          # Array
            "engine": "CROWN"
        }
        
        #self._print_feasibility_report(spec, result)
        return result

    def _reconstruct_full_vector(self, spec, nn_out, total_dim, inputs):
        """Helper to assemble the 6D vector from specific inputs and outputs."""
        vec = torch.zeros(total_dim, device=self.device)
        vec[spec.input_indices] = inputs
        vec[spec.output_indices] = nn_out
        return vec
    
