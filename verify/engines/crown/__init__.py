import sys
from pathlib import Path

# Dynamically add the submodule to the path
lib_path = Path(__file__).parent / "auto_LiRPA"
if lib_path.exists():
    sys.path.insert(0, str(lib_path))