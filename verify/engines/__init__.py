# nn_verify/engines/__init__.py
__all__ = ["MILPVerifier"]


def __getattr__(name):
    # Lazy, so the CROWN engine can be imported without pyomo installed
    if name == "MILPVerifier":
        from .mip import MILPVerifier
        return MILPVerifier
    raise AttributeError(name)
