import yaml
import sys
import argparse
from pathlib import Path

# Assuming your project structure follows the naming we discussed
from verify.utils.loader import NNLoader
from usecase import USECASE_ROUTER
from output.utils.report_gen import generate_report

"""
python main.py usecase/optimization/lp/config.yaml

to do:
- implement crown for lp as well
- implement 'minimization' for wc minimization with crown
- implement a dcopf, qp
- implement that output is jupyter notebook under 'output'

"""


def main():
    # 1. Improved Argument Parsing
    parser = argparse.ArgumentParser(description="Neural Network Verification Toolbox")
    parser.add_argument(
        "config", 
        nargs="?", 
        default="config.yaml", 
        help="Path to the YAML configuration file"
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()

    # 2. Safety Check
    if not config_path.exists():
        print(f"[-] ERROR: Config file not found at {config_path}")
        sys.exit(1)

    print(f"[*] Loading Configuration: {config_path}")

    # 3. Load and Parse YAML
    try:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
    except Exception as e:
        print(f"[-] ERROR: Failed to parse YAML: {e}")
        sys.exit(1)

    try:
        loader = NNLoader(config_data)
        engine_type = config_data['model_meta'].get('engine', 'milp').lower()
        
        # Extract routing info from metadata
        p_class = loader.meta.get('pclass')
        p_type = loader.meta.get('ptype')
        model_name = loader.meta.get('name', 'Unnamed_Model')

        if not p_class or not p_type:
            print("[-] ERROR: 'model_meta' must contain 'pclass' and 'ptype'.")
            return

        # 5. Routing Logic with Engine Selection
        engine_type = config_data['model_meta'].get('engine', 'milp').lower()
        
        if engine_type == "crown":
            # CROWN is a unified engine, so we use a specific CrownRunner
            # We import here to keep the startup light if not using CROWN
            from verify.engines.crown.wrapper import CrownRunner
            runner = CrownRunner(loader)
            print(f"[*] Engine selected: CROWN (Formal Bound Propagation)")
        else:
            # Default to MILP Router logic
            class_router = USECASE_ROUTER.get(p_class)
            if not class_router:
                print(f"[-] ERROR: Unknown problem class '{p_class}'.")
                return

            runner = class_router.get(p_type)
            if not runner:
                print(f"[-] ERROR: No MILP runner found for type '{p_type}'.")
                return
            print(f"[*] Engine selected: MILP (Exact Solver)")

        # 6. Execution
        try:
            print(f"[*] Executing {p_class}/{p_type} for model: {model_name}")
            print(f"{'-'*60}")
            
            # Both CrownRunner and MILP Runners now follow the same interface
            results = runner(loader)

            # 7. Conditional Report Generation
            report_setting = config_data['model_meta'].get('report', True)

            # Logic to handle both string "yes"/"no" and boolean True/False
            if isinstance(report_setting, str):
                do_report = report_setting.lower() in ['yes', 'true']
            else:
                # If it's already a bool (from YAML auto-parsing 'no' to False)
                do_report = bool(report_setting)

            if results and do_report:
                print(f"[*] Generating report for {model_name}...")
                
                output_dir = Path("output")
                output_dir.mkdir(exist_ok=True)
                
                report_path = generate_report(
                    results=results, 
                    config=config_data, 
                    output_path=str(output_dir)
                )
                print(f"[+] Success! Verification report saved to: {report_path}")
            else:
                print("[*] Report generation skipped (report: no or runner failed).")

        except Exception as e:
            print(f"[-] UNEXPECTED ERROR: {e}")

    except KeyError as e:
        print(f"[-] CONFIGURATION ERROR: Missing expected key {e}")
    except Exception as e:
        print(f"[-] UNEXPECTED ERROR: {e}")
        # Uncomment for debugging:
        # import traceback; traceback.print_exc()

if __name__ == "__main__":
    main()

