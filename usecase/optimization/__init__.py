# Change from .lp_verifier to .verify_lp
from .lp.verify_lp import run_lp_verification


def run_acopf_verification(loader, *args, **kwargs):
    # Imported lazily: the ACOPF model needs pytorch_lightning, which LP-only setups should not require
    from .acopf.verify_acopf import run_acopf_verification as _run
    return _run(loader, *args, **kwargs)


OPTIMIZATION_ROUTER = {
    "lp": run_lp_verification,
    "acopf": run_acopf_verification
}
