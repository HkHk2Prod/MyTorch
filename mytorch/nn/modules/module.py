class Module:
    def __init__(self):
        self._params = {}
        self._modules = {}

    def parameters(self):
        yield from self._params.values()
        for module in self._modules.values():
            yield from module.parameters()

    def __setattr__(self, name, value):
        if isinstance(value, (Parameter, Module)):
            if '_params' not in self.__dict__:
                raise AttributeError(f"cannot assign '{name}' before Module.__init__() call")
        if isinstance(value, Parameter):
            self._params[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        super().__setattr__(name, value)

    def zerograd(self):
        for p in self.parameters():
            p.grad = None

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError(f'{type(self).__name__} has not implemented forward()')
