class Galois2:
    def __init__(self, power, conway, value=0, noverify=False):
        self.pwr = power
        self.mask = (1<<self.pwr)-1

        # coefficient of lowest degree is at index zero
        self.mod = conway
        self.value = self._mod(self._norm_usr_in(value))

        if not noverify:
            assert self.mod.bit_length() == self.pwr + 1, (
                "Degree of conway polynomial must match prime power"
            )
            assert self.mod&1, "Conway polynomial isn't prime!"
            # if self.pwr <= 16:
            #     assert self._mod(1<<(1<<self.pwr)) == 2, (
            #         "Conway polynomial doesn't satisfy fermat's little theorem (not irreducible?)"
            #     )

    def new(self, value):
        return Galois2(self.pwr, self.mod, value, noverify=True)

    def zero(self):
        return self.new(0)

    def one(self):
        return self.new(1)

    def alpha(self):
        return self.new(2)

    def _norm_pwr(self, pwr):
        return pwr if pwr is not None else self.pwr

    def digit_count(self):
        return self.value.bit_length()

    def __eq__(self, other):
        return self.value == other.value

    def __neg__(self):
        return self

    def _mult_inv(self, value):
        return self._extended_gcd(value, self.mod)[0]

    def inverse(self):
        return self.new(self._mult_inv(self.value))

    def __truediv__(self, other):
        return self.new(self._mul(self.value, self._mult_inv(other.value)))

    def _extended_gcd(self, a, b):
        r0, r1 = a, b
        s0, s1 = 1, 0
        t0, t1 = 0, 1

        while r1 != 0:
            q = self._euclid_div(r0, r1)[0]
            r0, r1 = r1, r0 ^ self._mul(q, r1, True)
            s0, s1 = s1, s0 ^ self._mul(q, s1, True)
            t0, t1 = t1, t0 ^ self._mul(q, t1, True)

        return (s0, t0, r0)

    def _euclid_div(self, num, den):
        quo = 0
        den_bit_len = den.bit_length()
        while True:
            bit_diff = num.bit_length() - den_bit_len
            if bit_diff < 0:
                break
            num ^= den << bit_diff
            quo ^= 1 << bit_diff

        return (quo, num)

    def _mod(self, value):
        return self._euclid_div(value, self.mod)[1] & self.mask

    def _pow(self, base, exp):
        assert type(exp) is int
        assert exp >= 0

        if exp == 0:
            return 1
        elif exp == 1:
            return base

        # Do exponentiation by squaring
        result = 1
        current_base = base
        
        while True:
            if exp&1 == 1:
                result = self._mul(result, current_base)

            exp >>= 1
            if exp <= 0:
                break

            current_base = self._mul(current_base, current_base)

        return result

    def __pow__(self, exp):
        return self.new(self._pow(self.value, exp))

    def as_arr(self, pwr=None):
        pwr = self._norm_pwr(pwr)
        arr = []
        number = self.value
        while number:
            arr.append(number&1)
            number >>=1
        return arr+[0]*(self.pwr-len(arr))

    def _arr_to_int(self, exp):
        result = 0
        for i in exp[::-1]:
            result = (result<<1) ^ i
        return result

    def _norm_usr_in(self, inp):
        if type(inp) is Galois2:
            return inp.value
        elif type(inp) is int:
            return inp&self.mask
        else:
            return self._arr_to_int(inp)

    def __len__(self):
        return self.pwr

    def __add__(self, other):
        return self.new(self.value ^ other.value)

    __sub__ = __add__

    def _mul(self, a, b, nomod=False):
        # a is the smaller polynomial
        if a > b:
            a, b = b, a

        result = 0
        while a:
            height = a.bit_length() - 1
            result ^= b<<height
            a ^= 1<<height

        return result if nomod else self._mod(result)

    def __mul__(self, other):
        if type(other) is Galois2:
            return self.new(self._mul(self.value, other.value))
        else:
            return self.new(self.value if other&1 else 0)

    def __str__(self):
        return str(self.value)

    def __int__(self):
        return self.value

    def __repr__(self):
        return "Galois2(%d).new(%d)" % (self.pwr, self)

