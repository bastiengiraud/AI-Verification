# usecase/__init__.py
from .optimization import OPTIMIZATION_ROUTER

# Maps "pclass" to the sub-router
USECASE_ROUTER = {
    "optimization": OPTIMIZATION_ROUTER,
    # "tabular": TABULAR_ROUTER
}