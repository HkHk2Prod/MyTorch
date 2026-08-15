import operator

import pytest

import mytorch as torch
from mytorch._backend import xp
from mytorch.tensor import Tensor

# "Pointwise" binary operation. Example +, *, /, etc
# If you want to test one, just add it here.
POINTWISE_BINARY_OP = [
    operator.add,
    operator.mul,
    operator.truediv,
    operator.sub,
]

# Shapes for "pointwise" binary operation. Example +, *, /, etc
POINTWISE_BINARY_OP_SHAPES = [
    ((4, 3), (4, 3)),
    ((2, 3), (2, 1)),
    ((2, 3), (3,)),
    ((1, 1, 1, 3), (3,)),
    ((4, 3, 2), (3, 2)),
]


@pytest.fixture
def rng():
    return xp.random.default_rng(1234)


def test_defauls():
    t = Tensor([1, 2, 3])
    assert t.grad is None
    assert not t.requires_grad
    assert xp.array_equal(t.data, xp.asarray([1, 2, 3], dtype=float))
    assert t._children == ()


def test_pointwise_binary_arithmetics(rng):
    for op in POINTWISE_BINARY_OP:
        for a_shape, b_shape in POINTWISE_BINARY_OP_SHAPES:
            a, b = rng.standard_normal(a_shape), rng.standard_normal(b_shape)
            assert torch.equal(op(Tensor(a), Tensor(b)), Tensor(op(a, b)))


def test_pointwise_binary_result_type(rng):
    for op in POINTWISE_BINARY_OP:
        for a_shape, b_shape in POINTWISE_BINARY_OP_SHAPES:
            a, b = rng.standard_normal(a_shape), rng.standard_normal(b_shape)
            assert isinstance(op(Tensor(a), b), Tensor)
            assert isinstance(op(a, Tensor(b)), Tensor)
