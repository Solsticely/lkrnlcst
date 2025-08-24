import config as C
import util as U
import math
import numpy as np
import time
import os


class Phaser:
    def __init__(self, hzmin, hzmax=None):
        self.hzadd = hzmin
        self.hzmul = (hzmax or hzmin)-hzmin
        self.ph = C.rng.randint(0, 1024) / 1024
        pass

    def emit(self, samples, freq_value = 0, amp_value = 1):
        """
            freq_value is between zero and one, and blends between hzmin and
            hzmax provided during initialisation.

            amp_value is between whatever you want it to be and whatever else
            you want it to be. it is 10^(10*dB_value)
        """
        inc = (self.hzmul * freq_value + self.hzadd) / C.HZ
        out = (np.array(range(samples), dtype=C.TY) + 1) * inc + self.ph
        out = np.sin(np.pi * 2 * out)
        self.ph += inc * samples
        self.ph -= math.floor(self.ph)

        # TODO: implement amp smoothing or ending at zero
        return out * amp_value


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


def sweep():
    # Generates a sine sweep that increases exponentially
    DURATION_SECS = 0.5
    START_HZ = C.P.TOTAL_MINHZ
    END_HZ = C.P.TOTAL_MAXHZ

    length = round(DURATION_SECS * C.HZ)
    t = np.linspace(0, 1, length, dtype=C.TY)

    start_hz = START_HZ * np.pi * 2 * DURATION_SECS
    end_hz = END_HZ * np.pi * 2 * DURATION_SECS

    a = np.log(end_hz) - np.log(start_hz)
    b = start_hz / a

    return np.sin(b * np.exp(a * t) - b)


def modulate():
    chunks = read_aud_in_as_chunks()
    chunks = U.ChunkEater(chunks, 0)
    duration = C.P.MIN_FREQTIME * C.HZ
    duration_accumulator = duration

    yield sweep()

    tbc = Phaser(C.P.TBC_FREQ)
    clock = False

    while not chunks.is_eof:
        c_amp = 1 if clock else 0
        clock = not clock

        samples = round(duration_accumulator)
        duration_accumulator += duration - samples
        data = [chunks.next() for i in range(len(C.P.bin_spacings)-2)] + [c_amp, 1-c_amp]
        dft = np.zeros(C.P.dft_size)

        for inx, bit in enumerate(data):
            dft[C.P.bin_spacings[inx]] = bit * C.P.DFT_AMP_MUL

        signal = U.ttf(dft, samples)  # + tbc.emit(samples)
        yield signal * C.P.DFT_AMP_MUL


def main():
    wr = U.WaveWriter(C.AUD_OUT_PATH, C.AUD_SAMPWIDTH, C.HZ)
    total_time = 0
    before = time.time()
    with wr as file:
        for chunk in modulate():
            total_time += len(chunk) / C.HZ
            file.write(chunk)

    elapsed = time.time()-before
    filesize = os.path.getsize(C.DAT_IN_PATH)
    lin_filesize = os.path.getsize("./linux.dat")
    bytes_per_second = filesize/total_time

    print("Generated %.1f seconds of audio in %.1f seconds (%.2f×)" % (total_time, elapsed, total_time/elapsed))
    print("Approx min/MiB: %.1f" % (1024**2/60/bytes_per_second))
    print("Approx baud: %.1f bit/s, approx storage efficiency: %.1fKiB/s" % (bytes_per_second*8, bytes_per_second/1024))
    print("Approx final C-number w/ linux: C-%.0f (mins) \u00b1 wow&flutter & speed misconfiguration & ends" % (lin_filesize/bytes_per_second/60))


if __name__ == "__main__":
    main()

