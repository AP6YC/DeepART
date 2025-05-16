
from collections import defaultdict
from inspect import signature


class MultiDispatcher:
    def __init__(self):
        self._registry = defaultdict(list)

    def register(self, func):
        sig = signature(func)
        types = tuple(param.annotation for param in sig.parameters.values())
        self._registry[types].append(func)
        return func

    def dispatch(self, *args):
        types = tuple(type(arg) for arg in args)
        funcs = self._registry[types]
        if not funcs:
            # Raising a TypeError here provides a clear indication that no matching function was found
            # for the given argument types.
            raise TypeError("No matching function for types {}".format(types))
        return funcs[0](*args)


dispatcher = MultiDispatcher()
