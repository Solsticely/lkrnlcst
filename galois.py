import math
import numpy as np


class GaloisPN:
    def __init__(self, characteristic, power, conway, value=0, noverify=False):
        self.pwr = power
        self.chr = characteristic
        self.max = self.chr**self.pwr

        # coefficient of lowest degree is at index zero
        self.mod = self.usr_repr_to_arr(conway, self.pwr + 1, try_mod=False)
        self.value = self.usr_repr_to_arr(value)

        if not noverify:
            assert self.pwr < 127, "galois subfields must not be more than 7 bits"
            for i in range(2, min(round(math.sqrt(self.chr) + 0.5) + 1, self.chr)):
                assert self.chr % i != 0, "Characteristic must be prime!"
            assert self._digit_count(self.mod) == self.pwr + 1, (
                "Degree of conway polynomial must match prime power"
            )
            assert self.mod[0] != 0, "Conway polynomial isn't prime!"
            if 300 < math.log2(self.chr) * (self.chr**self.pwr - 1):
                assert (
                    self._int(self.int_to_arr(self.chr ** (self.chr**self.pwr)))
                    == self.chr
                ), (
                    "Conway polynomial doesn't satisfy fermat's little theorem (not irreducible?)"
                )

    def new(self, value):
        return GaloisPN(self.chr, self.pwr, self.mod, value, noverify=True)

    def _zero(self, pwr=None):
        return np.zeros(self._norm_pwr(pwr), dtype=np.uint8)

    def _one(self, pwr=None):
        return np.array([1] + [0] * (self._norm_pwr(pwr) - 1), dtype=np.uint8)

    def _alpha(self, pwr=None):
        return np.array([0, 1] + [0] * (self._norm_pwr(pwr) - 2), dtype=np.uint8)

    def zero(self):
        return self.new(self._zero())

    def one(self):
        return self.new(self._one())

    def alpha(self):
        return self.new(self._alpha)

    def _norm_pwr(self, pwr):
        return pwr if pwr is not None else self.pwr

    def _digit_count(self, value):
        for inx in range(len(value))[::-1]:
            if value[inx] >= 1:
                return inx + 1
        return 0

    def digit_count(self):
        return self._digit_count(self.value)

    def _inv(self, value):
        return (np.full(len(value), self.chr, dtype=np.uint8) - value) % self.chr

    def inv(self):
        return self.new(self._inv(self.value))

    def _pad(self, value, pwr=None):
        return np.append(value, self._zero(max(0, self._norm_pwr(pwr) - len(value))))

    def _pad_and_trunc(self, value, pwr=None, clean=True):
        pwr = self._norm_pwr(pwr)
        if clean:
            for i in value[pwr:]:
                assert i == 0, "Tried to truncate a longer polynomial"
        return self._pad(value[:pwr], pwr)

    def _mod(self, value):
        poly = value
        mod = self._pad(self.mod, len(poly))
        while True:
            poly_digit_count = self._digit_count(poly)
            bit_diff = poly_digit_count - self.pwr - 1
            if bit_diff < 0:
                break
            addition = self._inv(
                self._mul_c(np.roll(mod, bit_diff), poly[poly_digit_count - 1])
            )
            poly = self._add(poly, addition)
        return self._pad_and_trunc(poly)

    def usr_repr_to_arr(self, number, pwr=None, try_mod=True):
        pwr = self._norm_pwr(pwr)
        if type(number) is int:
            return self.int_to_arr(number, pwr, try_mod=try_mod)
        else:
            return self._pad_and_trunc(np.array(number, dtype=np.uint8), pwr)

    def int_to_arr(self, number, pwr=None, try_mod=False):
        pwr = self._norm_pwr(pwr)
        if not (hasattr(self, "mod") and pwr == self.pwr) and try_mod:
            assert False, "Tried to modulate a number before modulation is set up"
        assert 0 <= number < self.chr**pwr or try_mod, (
            "Number needs to be positive and able to fit into galois field"
        )

        output = []
        while number > 0:
            number, new_digit = divmod(number, self.chr)
            output.append(new_digit)

        output = self._pad(np.array(output, dtype=np.uint8), pwr)
        return self._mod(output) if try_mod else output

    def __len__(self):
        return self.pwr

    def _add(self, a, b):
        return (a + b) % self.chr

    def __add__(self, other):
        return self.new(self._add(self.value, other.value))

    def _mul_c(self, a, c):
        return (a * c) % self.chr

    def _mul(self, a, b):
        a_digit_c = self._digit_count(a)
        b_digit_c = self._digit_count(b)
        # a is the smaller polynomial
        if a_digit_c > b_digit_c:
            a_digit_c, a, b_digit_c, b = b_digit_c, b, a_digit_c, a
        total_digit_c = a_digit_c + b_digit_c
        result = self._zero(total_digit_c)
        a = self._pad_and_trunc(a, total_digit_c)
        b = self._pad_and_trunc(b, total_digit_c)
        result = self._zero(total_digit_c)
        for inx, val in enumerate(a):
            result += np.roll(self._mul_c(b, val), inx)
        return result

    def __mul__(self, other):
        if type(other) is GaloisPN:
            return self.new(self._mul(self.value, other.value))
        else:
            return self.new(self._mul_c(self.value, other))

    def __str__(self):
        # return ",".join(str(i) for i in self.value[:self.digit_count()][::-1])
        return str(int(self))

    def _int(self, arr):
        result = 0
        for i in arr[::-1]:
            result = result * self.chr + int(i)
        return result

    def __int__(self):
        return self._int(self.value)

    def __repr__(self):
        return "GaloisPN(%d,%d).new(%d)" % (self.chr, self.pwr, self)
