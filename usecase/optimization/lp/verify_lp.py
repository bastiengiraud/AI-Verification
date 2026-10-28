import time
import numpy as np
from verify import MILPVerifier, VerifierConfig
from verify.engines.crown.wrapper import CrownRunner


def run_lp_verification(loader):
    
    # These are now structured objects/lists
    nn_params = loader.get_layer_params()
    spec = loader.get_spec() # This is a VerificationSpec object
    
    model_name = loader.meta.get('name', 'Unknown Model')
    check_type = loader.meta.get('check', 'constraint')
    solver_type = loader.meta.get('solver', 'gurobi')
    engine_type = loader.meta.get('engine', 'milp')
    A = spec.constraints_A
    cons_names = loader.spec_raw.get('constraints', {}).get('names') or [f"Row {i}" for i in range(len(A))]

    print(f"\n--- Verifying LP Surrogate: {model_name} ---")
    print(f"Check Type: {check_type.upper()}")
    print(f"Engine Type: {engine_type.upper()}")
    if engine_type == "milp":
        print(f"Solver: {solver_type.upper()}")

    # 2. Initialize Verifier with Config
    if engine_type == "milp":
        config = VerifierConfig(solver_name = solver_type, use_obbt=True, verbose=False)
        verifier = MILPVerifier(nn_params, config=config)
    elif engine_type == "crown":
        verifier = CrownRunner(loader)

    # --- START TIMING ---
    start_time = time.perf_counter()
    
    # 3. Execution Logic
    if check_type == "constraint":
        
        # We pass the full spec so the verifier knows the mapping
        result = verifier.verify_lp_feasibility(
            spec=spec, 
            input_bounds=spec.input_bounds
        )
        end_time = time.perf_counter()
        result['runtime_sec'] = end_time - start_time

        if result['status'] == "Success":
            _print_feasibility_report(spec, result, cons_names)
        else:
            print(f"Solver Error: {result['status']}")


    elif check_type == "distance":
        
        if engine_type == "crown":
            raise NotImplementedError("CROWN engine does not currently support 'distance' (optimality gap) checks.")

        result = verifier.verify_lp_optimality_gap(
            spec=spec, 
            input_bounds=spec.input_bounds
        )
        end_time = time.perf_counter()
        result['runtime_sec'] = end_time - start_time

        if result['status'] == "Success":
            _print_optimality_report(spec, result)
        else:
            print(f"Solver Error: {result['status']}")
            
    print(f"\n[Verification Completed in {result.get('runtime_sec', 0):.4f} seconds]")
    return result

# --- Helper functions to keep the main logic clean ---

def _print_feasibility_report(spec, result, cons_names):
    x_full = np.array(result['full_x_vector'])
    A = spec.constraints_A
    b = spec.b_static
    in_idx = spec.input_indices
    out_idx = spec.output_indices
    
    engine = result.get('engine', 'MILP').upper()
    max_viol = result.get('max_violation', 0.0)
    runtime = result.get('runtime_sec', 0.0)
    worst_idx = result.get('worst_row_idx')
    
    # Define a safety threshold (handling floating point noise)
    is_safe = max_viol <= 1e-7 

    # --- HEADER ---
    if is_safe:
        print(f"\n[+] VERIFICATION SUCCESS: System is SAFE")
        print(f"Max Violation Found: {max_viol:.6f} (Satisfied)")
    else:
        print(f"\n[!] VERIFICATION FAILURE: Violation Found")
        print(f"Max Violation Found: {max_viol:.6f} (Violated)")
        
    print(f"Solve Time: {runtime:.4f}s")
    print(f"{'='*80}")

    # For CROWN, we still want to keep the warning if unsafe
    if engine == "CROWN":
        if is_safe:
            print(f"[*] CROWN has formally proven that no violation > {max_viol:.6f} exists.")
            print("    Note that CROWN gives is based on relaxations, and the result is conservative.")
        else:
            attained = result.get('attained_violation', float('nan'))
            print(f"[*] CROWN bound: no violation > {max_viol:.6f}. PGD counter-example attains {attained:.6f} "
                  f"at input {np.round(result['at_input_val'], 4).tolist()}.")
            if attained > 1e-7:
                print("    The attained violation is real: the system is NOT safe.")
            else:
                print("    No real violation found yet: the bound may be loose (CROWN is conservative); use MILP.")
        print(f"{'='*80}\n")
        return
    
    
    # --- STANDARD MILP TABLES (Only reached if engine != CROWN) ---
    x_full = np.array(result['full_x_vector'])
    in_idx, out_idx = spec.input_indices, spec.output_indices
    names = cons_names
    
    print(f"{'COMPONENT ANALYSIS':<20} | {'INDICES':<15} | {'VALUES'}")
    print(f"{'-'*80}")
    print(f"{'NN Inputs':<20} | {str(in_idx):<15} | {[round(x_full[i], 4).item() for i in in_idx]}")
    print(f"{'NN Outputs':<20} | {str(out_idx):<15} | {[round(x_full[i], 4).item() for i in out_idx]}")
    
    # Identify auxiliary variables (those in x_full but not in in/out idxs)
    print(f"{'='*80}\n")
    all_indices = set(range(len(x_full)))
    aux_idx = sorted(list(all_indices - set(in_idx) - set(out_idx)))
    if aux_idx:
        print(f"{'Aux Variables':<20} | {str(aux_idx):<15} | {[round(x_full[i], 4) for i in aux_idx]}")
    
    print(f"\n{'CONSTRAINT CHECK':<80}")
    print(f"{'-'*80}")
    print(f"{'Row':<5} | {'Constraint Name':<25} | {'LHS':<15} | {'RHS (b)':<12} | {'Violation':<12}")
    print(f"{'-'*80}")

    for i in range(len(A)):
        lhs_val = np.dot(A[i], x_full)
        violation = lhs_val - b[i]
        # Logic for the tag
        if violation > 1e-5:
            # Check if this specific row is the absolute worst one
            status = "[MAX VIOLATOR]" if i == worst_idx else "[FAILED]"
        else:
            status = "[OK]"
        c_name = names[i] if i < len(names) else f"Row {i}"
        
        # Color the output if violation is high (optional, depends on terminal)
        print(f"{i:<5} | {c_name:<25} | {lhs_val:<15.6f} | {b[i]:<12.6f} | {violation:<12.8f} {status}")
        
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
    runtime = result.get('runtime_sec', 0.0)
    
    print(f"\n{'='*70}")
    print(f"{'SYSTEM OPTIMALITY REGRET REPORT':^70}")
    print(f"Solve Time: {runtime:.4f}s")
    print(f"{'='*70}")
    
    # 1. High-Level Metrics
    print(f"{'Metric':<30} | {'NN System':<15} | {'True Optimal':<15}")
    print(f"{'-'*70}")
    print(f"{'Total Objective Value':<30} | {nn_cost:<15.6f} | {true_cost:<15.6f}")
    print(f"{'-'*70}")
    print(f"{'PROVEN MAX SUB OPTIMALITY':<30} | {gap:<30.6f}")
    
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
    
    # 3. Final Status Logic
    if gap > 1e-4:
        status = "[!] RISK: SUBOPTIMAL"
        color_tag = "(!) WARNING"
    else:
        status = "[+] PASSED: OPTIMAL"
        color_tag = "[OK]"

    print(f"{'-'*70}")
    print(f"STATUS: {status}")
    print(f"Detail: Found a solution {gap:.4f} units away from optimal.")
    print(f"{'='*70}")