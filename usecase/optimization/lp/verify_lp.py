import numpy as np
from verify import MILPVerifier, VerifierConfig

def run_lp_verification(loader):
    
    # These are now structured objects/lists
    nn_params = loader.get_layer_params()
    spec = loader.get_spec() # This is a VerificationSpec object
    
    model_name = loader.meta.get('name', 'Unknown Model')
    check_type = loader.meta.get('check', 'constraint')

    print(f"\n--- Verifying LP Surrogate: {model_name} ---")
    print(f"Check Type: {check_type.upper()}")

    # 2. Initialize Verifier with Config
    config = VerifierConfig(use_obbt=True, verbose=False)
    verifier = MILPVerifier(nn_params, config=config)
    
    # 3. Execution Logic
    if check_type == "constraint":
        
        # We pass the full spec so the verifier knows the mapping
        result = verifier.verify_lp_feasibility(
            spec=spec, 
            input_bounds=spec.input_bounds
        )

        if result['status'] == "Success":
            _print_feasibility_report(spec, result)
        else:
            print(f"Solver Error: {result['status']}")
            
        return result

    elif check_type == "distance":
        
        result = verifier.verify_lp_optimality_gap(
            spec=spec, 
            input_bounds=spec.input_bounds
        )

        if result['status'] == "Success":
            _print_optimality_report(spec, result)
        else:
            print(f"Solver Error: {result['status']}")
            
        return result

# --- Helper functions to keep the main logic clean ---

def _print_feasibility_report(spec, result):
    x_full = np.array(result['full_x_vector'])
    A = spec.constraints_A
    b = spec.b_static
    in_idx = spec.input_indices
    out_idx = spec.output_indices
    
    print(f"\n[!] VERIFICATION FAILURE: Max Violation found: {result['max_violation']:.6f}")
    print(f"{'='*80}")
    print(f"{'COMPONENT ANALYSIS':<20} | {'INDICES':<15} | {'VALUES'}")
    print(f"{'-'*80}")
    print(f"{'NN Inputs (Fixed)':<20} | {str(in_idx):<15} | {[round(x_full[i], 4) for i in in_idx]}")
    print(f"{'NN Outputs (Pred)':<20} | {str(out_idx):<15} | {[round(x_full[i], 4) for i in out_idx]}")
    
    # Identify auxiliary variables (those in x_full but not in in/out idxs)
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
        
        # Color the output if violation is high (optional, depends on terminal)
        print(f"{i:<5} | {lhs_val:<20.6f} | {b[i]:<12.6f} | {violation:<12.6f} {status}")
        
        # Explain the calculation for the failing row
        if violation > 1e-5:
            # Show which terms in the dot product contribute most
            contributions = [(j, A[i][j] * x_full[j]) for j in range(len(x_full)) if abs(A[i][j] * x_full[j]) > 1e-4]
            contrib_str = " + ".join([f"({val:.2f} [x{j}])" for j, val in contributions])
            print(f"      └─ Calculation: {contrib_str} = {lhs_val:.4f}")

    print(f"{'='*80}")

def _print_optimality_report(spec, result):
    # Extract data
    x_nn_full = np.array(result['full_x_vector'])
    x_star = np.array(result['x_star'])
    out_idx = spec.output_indices
    c = spec.objective_c
    
    nn_cost = np.dot(c, x_nn_full)
    true_cost = np.dot(c, x_star)
    gap = result['optimality_gap']
    
    print(f"\n{'='*70}")
    print(f"{'SYSTEM OPTIMALITY REGRET REPORT':^70}")
    print(f"{'='*70}")
    
    # 1. High-Level Metrics
    print(f"{'Metric':<30} | {'NN System':<15} | {'True Optimal':<15}")
    print(f"{'-'*70}")
    print(f"{'Total Objective Value':<30} | {nn_cost:<15.6f} | {true_cost:<15.6f}")
    print(f"{'NN Output Costs (Pred)':<30} | {nn_cost:<15.6f} | {true_cost:<15.6f}")
    print(f"{'-'*70}")
    print(f"{'PROVEN MAX REGRET (GAP)':<30} | {gap:<30.6f}")
    
    # 2. Variable Comparison (The "Why")
    print(f"\n{'OUTPUT VARIABLE COMPARISON':^70}")
    print(f"{'-'*70}")
    print(f"{'Index':<10} | {'Cost Coeff':<12} | {'NN Value':<12} | {'True Opt':<12} | {'Diff'}")
    print(f"{'-'*70}")
    
    for i in out_idx:
        coeff = c[i]
        val_nn = x_nn_full[i]
        val_star = x_star[i]
        diff = val_nn - val_star
        print(f"{i:<10} | {coeff:<12.4f} | {val_nn:<12.4f} | {val_star:<12.4f} | {diff:+.4f}")
    
    # 3. Final Status
    status = "[!] FAILED" if gap > 1e-4 else "[+] PASSED"
    print(f"{'-'*70}")
    print(f"STATUS: {status}")
    print(f"{'='*70}")