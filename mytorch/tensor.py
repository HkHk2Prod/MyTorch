from mytorch._backend import xp


def sum_to(tensor, shape):
    while tensor.ndim > len(shape):
        tensor = tensor.sum(axis=0)
    for i, d in enumerate(shape):
        if d == 1:
            tensor = tensor.sum(axis=i, keepdims=True)
    return tensor

def _T(tensor):
    return tensor.swapaxes(-1,-2)

def _as_tensor(x):
    return x if isinstance(x, Tensor) else Tensor(x)



class Tensor:
    __slots__ = {'data', 'grad', 'requires_grad', '_backward', '_children'}
    __array_ufunc__ = None # This overrides numpy behavior for __add__ 
                           # in a + b where a is numpy array
                           # it will call __radd__ for Tensor
                           # if b is a Tensor.

    def __init__(self, data, *, requires_grad=False, _children = ()):
        self.data = xp.asarray(data, dtype=float)
        self.requires_grad = requires_grad
        self._backward = lambda g: None
        self.grad = None
        self._children = _children

    def _accumulate(self, g):
        self.grad = g.copy() if self.grad is None else self.grad + g

    def __add__(self, other):
        other = _as_tensor(other)
        out = Tensor(
            data=self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children = (self, other),
        )
        def _backward(g):
            if self.requires_grad:
                self._accumulate(sum_to(g, self.data.shape))
            if other.requires_grad:
                other._accumulate(sum_to(g, other.data.shape))
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = _as_tensor(other)
        out = Tensor(
            data=self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children = (self, other),
        )
        def _backward(g):
            if self.requires_grad:
                self._accumulate(sum_to(g  * other.data, self.data.shape))
            if other.requires_grad:
                other._accumulate(sum_to(g * self.data, other.data.shape))
        out._backward = _backward
        return out    
    
    def __pow__(self, n):
        out = Tensor(self.data ** n,
                     requires_grad=self.requires_grad,
                     _children=(self,))
        def _backward(g):
            if self.requires_grad:
                self._accumulate(g * n * self.data ** (n - 1))
        out._backward = _backward
        return out

    def __truediv__(self, other):
        other = _as_tensor(other)
        out = Tensor(
            data=self.data / other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children = (self, other),
        )
        def _backward(g):
            if self.requires_grad:
                self._accumulate(sum_to(g  / other.data, self.data.shape))
            if other.requires_grad:
                other._accumulate(sum_to(
                    g *(-self.data) / (other.data ** 2),
                    other.data.shape)
                    )
        out._backward = _backward
        return out

    def __neg__(self):
        # not optimal
        return self * (-1)

    def __sub__(self, other):
        # not optimal
        return self + (-_as_tensor(other))

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return _as_tensor(other) + (-self)

    def __rtruediv__(self, other):
        return _as_tensor(other) / self


    def exp(self):
        out = Tensor(xp.exp(self.data),
                     requires_grad=self.requires_grad,
                     _children=(self,))
        def _backward(g):
            if self.requires_grad:
                self._accumulate(out.data * g)
        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims),
                     requires_grad=self.requires_grad,
                     _children=(self,))
        def _backward(g):
            if self.requires_grad:
                if axis is not None and not keepdims:
                    g = xp.expand_dims(g, axis)
                self._accumulate(xp.broadcast_to(g, self.data.shape).copy())
        out._backward = _backward
        return out

    def max(self, axis=None, keepdims=False):
        m = self.data.max(axis=axis, keepdims=True)
        out = Tensor(m if keepdims else self.data.max(axis=axis),
                     requires_grad=self.requires_grad,
                     _children=(self,))
        def _backward(g):
            if self.requires_grad:
                mask = (self.data == m)
                count = mask.sum(axis=axis, keepdims=True)
                if axis is not None and not keepdims:
                    g = xp.expand_dims(g, axis)
                self._accumulate(mask * g / count)
        out._backward = _backward
        return out

    def __matmul__(self, other):
        out = Tensor(self.data @ other.data,
                     requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other))
        def _backward(g):
            if self.requires_grad:
                self._accumulate(sum_to(g @ _T(other.data), self.data.shape))
            if other.requires_grad:
                other._accumulate(sum_to(_T(self.data) @ g, other.data.shape))
        out._backward = _backward
        return out

    def size(self, dim=None):
        return self.data.shape if dim is None else self.data.shape[dim]

    @property
    def shape(self):
        return self.data.shape

    def transpose(self, dim0, dim1):
        out = Tensor(self.data.swapaxes(dim0, dim1),
                     requires_grad=self.requires_grad,
                     _children=(self,))
        def _backward(g):
            if self.requires_grad:
                self._accumulate(g.swapaxes(dim0, dim1))
        out._backward = _backward
        return out

    @property
    def mT(self):
        return self.transpose(-1, -2)

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for c in v._children:
                    build_topo(c)
                topo.append(v)
        self.grad = xp.ones_like(self.data)
        build_topo(self)
        for v in reversed(topo):
            assert v.grad is not None
            v._backward(v.grad)

    @property
    def device(self):
        return self.data.device