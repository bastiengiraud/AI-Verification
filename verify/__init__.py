def __getattr__(name):
    # Lazy: importing verify.engines.crown must not require pyomo, which only the MILP engine needs
    if name in ("MILPVerifier", "VerifierConfig"):
        from .engines.mip import milp_verifier
        return getattr(milp_verifier, name)
    raise AttributeError(name)
