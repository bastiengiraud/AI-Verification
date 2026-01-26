# Change from .lp_verifier to .verify_lp
from .lp.verify_lp import run_lp_verification

OPTIMIZATION_ROUTER = {
    "lp": run_lp_verification
}