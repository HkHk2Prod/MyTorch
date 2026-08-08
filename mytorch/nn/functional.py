import mytorch as torch
from mytorch.tensor import _as_tensor

from .._backend import xp


def linear(x, weight, bias=None):
    output = x @ weight.mT
    return output if bias is None else output + bias


def softmax(x, dim=None):
    x = _as_tensor(x)
    m = xp.max(x.data, axis=dim, keepdims=True)
    out = torch.exp(x - m)
    return out / out.sum(axis=dim, keepdims=True)


def scaled_dot_product_attention(q, k, v, is_causal=False):

    dim = k.size(-1)
    attn = q @ k.transpose(-1, -2) / xp.sqrt(dim)
    if is_causal:
        L, S = q.size(-2), k.size(-2)
        mask = xp.where(xp.triu(xp.ones((L, S)), k=1), -xp.inf, 0.0)
        attn = attn + mask
    attn = softmax(attn, dim=-1)
    return attn @ v
