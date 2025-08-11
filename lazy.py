class Lazy:
    def __init__(self, init):
        self._initialiser = init

    def __getattr__(self, attr):
        self._data = self._initialiser()

        def new_getattr(self, attr):
            ga = getattr(self._data, attr)
            return ga if ga is not None else ga.__getattr__(attr)

        self.__getattr__ = new_getattr

        return new_getattr(self, attr)

