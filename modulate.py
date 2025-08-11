import config as C
import util as U
import math
import numpy as np


class Phaser:
    def __init__(self, hzmin, hzmax):
        self.hzadd = hzmin
        self.hzmul = hzmax-hzmin
        self.ph = 0
        pass

    def emit(self, samples, value):  # value between zero and one
        value = 2**value - 1  # remap to reduce density in higher frequencies
        inc = (self.hzmul * value + self.hzadd) / C.HZ
        out = (np.array(range(samples), dtype=C.TY) + 1) * inc + self.ph
        out = np.sin(np.pi * 2 * out)
        self.ph += inc * samples
        self.ph -= math.floor(self.ph)

        return out


def read_aud_in_as_chunks():
    assert math.log2(C.P.FM_BASE)-math.floor(math.log2(C.P.FM_BASE)+.0001) < .0001, "FM data modulation base is not a power of two"
    chunk_bits = round(math.log2(C.P.FM_BASE))

    all = None
    with open(C.DAT_IN_PATH, "rb").detach() as file:
        all = file.readall()

    as_bits = [int(j) for i in all for j in bin(i)[2:].zfill(8)]
    as_groups = U.chunk(as_bits, chunk_bits, 0)
    # as_chunks = [sum([i[j] * 2**j for j in range(chunk_bits)]) for i in as_groups]
    as_chunks = [int("".join(str(j) for j in i), 2) for i in as_groups]
    as_floats = [i/(C.P.FM_BASE-1) for i in as_chunks]

    return as_floats


def modulate():
    chunks = read_aud_in_as_chunks()
    chunks = U.ChunkEater(chunks, 0)
    duration = round(C.P.MIN_FREQTIME * C.HZ)
    ph1 = Phaser(C.P.MIN_AUDIOFREQ+100, C.P.MIN_AUDIOFREQ+1000)
    ph2 = Phaser(C.P.MIN_AUDIOFREQ+2000, C.P.MIN_AUDIOFREQ+4000)
    tbc = Phaser(C.P.TBC_FREQ, 0)

    while not chunks.is_eof:
        sum = ph1.emit(duration, chunks.next()) * 1/3
        sum += ph2.emit(duration, chunks.next()) * 1/3
        sum += tbc.emit(duration, 0) * 1/3

        yield sum


def main():
    wr = U.WaveWriter(C.AUD_OUT_PATH, C.AUD_SAMPWIDTH, C.HZ, C.AUD_NPTYPE)
    with wr as file:
        for chunk in modulate():
            file.write(chunk)


if __name__ == "__main__":
    main()

