import config as C
import util as U
import math
import wave


class Phaser:
    def __init__(self, hzmin, hzmax):
        self.hzadd = hzmin
        self.hzmul = hzmax-hzmin
        self.ph = 0
        pass

    def emit(self, samples, value):  # value between zero and one
        value = (math.exp(value)-1)/(math.e-1)
        out = []
        for i in range(samples):
            out.append(math.sin(math.pi * 2 * self.ph))
            self.ph += (self.hzmul * value + self.hzadd) / C.HZ
        self.ph -= math.floor(self.ph)

        return out


def read_aud_in_as_chunks():
    # assert C.FM_
    assert math.log2(C.FM_BASE)-math.floor(math.log2(C.FM_BASE)+.0001) < .0001, "FM data modulation base is not a power of two"
    chunk_bits = round(math.log2(C.FM_BASE))

    all = None
    with open(C.DAT_IN_PATH, "rb").detach() as file:
        all = file.readall()

    as_bits = [int(j) for i in all for j in bin(i)[2:].zfill(8)]
    as_groups = U.chunk(as_bits, chunk_bits, 0)
    # as_chunks = [sum([i[j] * 2**j for j in range(chunk_bits)]) for i in as_groups]
    as_chunks = [int("".join(str(j) for j in i), 2) for i in as_groups]
    as_floats = [i/(C.FM_BASE-1) for i in as_chunks]

    return as_floats


def modulate():
    chunks = read_aud_in_as_chunks()
    minfreq = 1500
    maxfreq = 1500
    duration = 1/minfreq
    duration = max(0.010, duration)  # 10 ms minimum
    duration = round(duration * C.HZ)
    ph = Phaser(minfreq, maxfreq)
    for i in chunks:
        yield ph.emit(duration, i)


def generate_samples():
    for chunk in modulate():
        factor = C.MAX * 0.7
        clamped = [min(C.MAX, max(C.MIN, round(factor * i))) for i in chunk]
        as_bytes = [int.to_bytes(i, C.AUD_SAMPWIDTH, "little", signed=True) for i in clamped]
        yield b"".join(as_bytes)


def main():
    with wave.open(C.AUD_OUT_PATH, "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(C.AUD_SAMPWIDTH)
        file.setframerate(C.HZ)
        for chunk in generate_samples():
            file.writeframes(chunk)


if __name__ == "__main__":
    main()

