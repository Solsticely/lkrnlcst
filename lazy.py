class Lazy:
    def __init__(self, init):
        self._initialiser = init
        
        def first_getattr(self, attr):
            self._data = self._initialiser()

            def new_getattr(self, attr):
                ga = getattr(self._data, attr)
                return ga if ga is not None else ga.__getattr__(attr)

            self._i_getattr = new_getattr

            return new_getattr(self, attr)
        self._i_getattr = first_getattr

    def __getattr__(self, attr):
        return self._i_getattr(self, attr)
