# Whenever you add anything here run bash "python tools/generate_api.py"
# and ruff check --fix

# Translates _CREATION operation from numpy into corresponding operations in torch.
# E.g: torch.ones(...) is the same as Tensor(np.ones(...))
CREATION_OPS = (
    "zeros",
    "ones",
    "empty",
    "full",
    "arange",
    "linspace",
    "eye",
    "zeros_like",
    "ones_like",
    "full_like",
)

# Tensor methods re-exported as free functions: x.exp() -> torch.exp(x)
METHOD_OPS = ("exp", "sum", "max", "transpose", "equal")
