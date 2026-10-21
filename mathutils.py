class DynPoly:
    def __init__(self, add_identity, mul_identity, coeffs=None):
        # coefficients are constant term first, most significant term last
        self.v0 = add_identity
        self.v1 = mul_identity
        self.value = coeffs if coeffs is not None else [self.v0]
        self.itype = type(self.v0)

        assert type(self.v1) is self.itype, "Additive and multiplicative identity have different types"
        for i in self.value:
            assert type(i) is self.itype, "Tried to make a polynomial with unexpected types for coefficients"
        assert len(self.value) > 0, "Constant term needs to exist"

    def _pad_to_eq(self, *args):
        args = [i for i in args]
        max_len = max(len(i) for i in args)
        return tuple(i + [self.v0] * (max_len - len(i)) for i in args)

    def new(self, value):
        return DynPoly(self.v0, self.v1, value)

    def zero(self):
        return self.new([self.v0])

    def one(self):
        return self.new([self.v1])

    def _eq(self, a, b):
        for i, j in zip(*self._pad_to_eq(a, b)):
            if i != j:
                return False
        return True

    def _add(self, a, b):
        return [i + j for i, j in zip(*self._pad_to_eq(a, b))]

    def _sub(self, a, b):
        return self._add(a, self._neg(b))

    def _neg(self, poly):
        return [-i for i in poly]

    def __add__(self, other):
        return self.new(self._add(self.value, other.value))

    def _mul(self, a, b):
        lena, lenb = len(a), len(b)
        if lena > lenb:
            a, b, lena, lenb = b, a, lenb, lena

        if lena == 0:
            return []

        result = [self.v0] * (lena + lenb - 1)

        for off1, mult1 in enumerate(a):
            for off2, mult2 in enumerate(b):
                result[off1 + off2] += mult2 * mult1

        return result

    def _mul_c(self, a, c):
        if c == self.v0: return []
        return [i*c for i in a]

    def __mul__(self, other):
        inner = None
        if isinstance(other, self.itype):
            inner = self._mul_c(self.value, other)
        else:
            inner = self._mul(self.value, other.value)
        return self.new(inner)

    def _slf_or_val(self, value):
        return self.value if value is None else value

    def degree(self, value=None):
        value = self._slf_or_val(value)
        count = len(value)

        for i in value[:0:-1]:
            if i == self.v0:
                count -= 1
            else:
                break
        return count - 1

    def _trunc(self, value=None):
        value = self._slf_or_val(value)
        return value[:self.degree(value)+1]

    def evaluate_at(self, x, value=None):
        value = self._slf_or_val(value)
        result = value[0]
        power = x
        pointer = 1
        halt_at = len(value) - 1
        while True:
            result += power * value[pointer]

            # Done this way to avoid extra multiplication
            if pointer == halt_at:
                break

            pointer += 1
            power *= x

        return result

    def _divmod(self, num, den):
        den_degree = self.degree(den)
        one_den = self._mul_c(den, -self.v1 / den[den_degree])
        assert den_degree == self.degree(one_den), "Degree of polynomial changed after scalar multiplication. Is denominator zero?"
        quot = []
        
        while True:
            num_deg = self.degree(num)
            bit_diff = num_deg - den_degree
            if bit_diff < 0 or (bit_diff <= 0 and num[0] == self.v0):
                return quot, self._trunc(num)

            mult = num[num_deg]
            assert mult != self.v0, "Polynomial long division tried to multiply denominator by zero"
            pad = [self.v0]*bit_diff
            num = self._add(num, pad+self._mul_c(one_den, mult))
            quot = self._add(quot, pad+[mult])

    def divmod(self, other):
        return tuple(self.new(i) for i in self._divmod(self.value, other.value))

    def __truediv__(self, other):
        return self.new(self._divmod(self.value, other.value)[0])

    def _gcd(self, a, b, stop_at_degree=-1):
        # o, n stand for old, new
        ro, rn = a, b
        so, sn = [self.v1], [self.v0]
        to, tn = [self.v0], [self.v1]
        print("re-gcd", a, b)

        while self.degree(ro) >= stop_at_degree and not self._eq(rn, [self.v0]):
            quot = self._divmod(ro, rn)[0]
            print(ro, rn, self._sub(ro, self._mul(quot, rn)))
            assert False
            ro, rn = rn, self._sub(ro, self._mul(quot, rn))
            so, sn = sn, self._sub(so, self._mul(quot, sn))
            to, tn = tn, self._sub(to, self._mul(quot, tn))

        # Return gcd, bezout coeffs
        return tuple(self._trunc(i) for i in (ro, so, to))

    def gcd(self, other, stop_at_degree=-1):
        return tuple(self.new(i) for i in self._gcd(self.value, other.value, stop_at_degree))
    
    def lagrange(self, points: list):
        from tqdm import tqdm
        final = []
        for xj, yj in tqdm(points):
            basis = [self.v1]
            for xm, ym in points:
                if xm == xj:
                    continue
                poly2 = [-xm, self.v1]
                gcd = self._gcd(poly2, basis)[0]
                basis, residue = self._divmod(basis, gcd)
                assert self.degree(residue) == 0 and residue[0] == self.v0, "Calculating lcm(a,b) by (a/gcd(a,b))*b resulted in a division with remainder"
                basis = self._trunc(self._mul(basis, poly2))
            
            mult_const = yj / self.evaluate_at(xj, basis)
            basis = self._mul_c(basis, mult_const)
            print(self.evaluate_at(xj, basis), yj)
            assert self.evaluate_at(xj, basis) == yj, "Lagrange polynomial is not one"
            final = self._add(final, basis)
        print(points)
        print(final)
        for x, y in points:
            print(x, y, self.evaluate_at(x, final))
            assert self.evaluate_at(x, final) == y, 'Lagrange failed'
    
        return self.new(self._trunc(final))
