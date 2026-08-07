from ..functional import linear 
from ..parameter import Parameter
from .module import Module
from mytorch._backend import xp

class Linear(Module):
    def __init__(self, input_size, out_size, bias=True):
        super().__init__()
        rng = xp.random.default_rng()
        self.bias = bias
        assert input_size > 0
        spread = 1 / xp.sqrt(input_size)
        self.W = Parameter(rng.uniform(low=-spread, high=spread, size=(out_size, input_size)))
        self.b = (
            Parameter(rng.uniform(low=-spread, high=spread, size=(out_size,))) 
            if bias is not None
            else None
            )

    def forward(self, x):
        return linear(x, self.W, self.b)