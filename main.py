import yaml
import sys
import argparse
from pathlib import Path

# Assuming your project structure follows the naming we discussed
from verify.utils.loader import NNLoader
from usecase import USECASE_ROUTER

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

    # 4. Agnostic Initialization
    try:
        # NNLoader handles the architecture (MLP vs CNN) internally
        loader = NNLoader(config_data)
        
        # Extract routing info from metadata
        p_class = loader.meta.get('pclass')
        p_type = loader.meta.get('ptype')
        model_name = loader.meta.get('name', 'Unnamed_Model')

        if not p_class or not p_type:
            print("[-] ERROR: 'model_meta' must contain 'pclass' and 'ptype'.")
            return

        # 5. Routing Logic with Safety Nets
        class_router = USECASE_ROUTER.get(p_class)
        if not class_router:
            print(f"[-] ERROR: Unknown problem class '{p_class}'.")
            print(f"    Available classes: {list(USECASE_ROUTER.keys())}")
            return

        runner = class_router.get(p_type)
        if not runner:
            print(f"[-] ERROR: No runner found for type '{p_type}' within class '{p_class}'.")
            return

        # 6. Execution
        print(f"[*] Executing {p_class}/{p_type} for model: {model_name}")
        print(f"{'-'*60}")
        
        # We pass the LOADER object. 
        # This is the key to agnosticism: the runner decides what it needs from the loader.
        runner(loader)

    except KeyError as e:
        print(f"[-] CONFIGURATION ERROR: Missing expected key {e}")
    except Exception as e:
        print(f"[-] UNEXPECTED ERROR: {e}")
        # Uncomment for debugging:
        # import traceback; traceback.print_exc()

if __name__ == "__main__":
    main()

