import pytest
from mytorch.tensor import Tensor
from mytorch._backend import xp

def test_defauls():
    t = Tensor([1, 2, 3])
    assert t.grad is None
    assert t.requires_grad == False
    assert xp.array_equal(
        t.data,
        xp.asarray([1,2,3], dtype=float)
        )
    assert t._children == ()

