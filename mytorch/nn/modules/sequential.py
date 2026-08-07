from .module import Module

class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        for i, layer in enumerate(layers):
            setattr(self, f'layer{i}', layer)

    def forward(self, x):
        for layer in self._modules.values():
            x = layer(x)
        return x