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


def modulate():
    chunks = read_aud_in_as_chunks()
    chunks = U.ChunkEater(chunks, 0)
    duration = C.P.MIN_FREQTIME * C.HZ
    duration_accumulator = duration

    tbc = Phaser(C.P.TBC_FREQ)
    clock = False

    # bins = np.arange(starting_bin, ending_bin+1, C.P.DFT_BIN_DELTA, dtype=np.int16)
    max_vol_dft = np.zeros(C.P.dft_size)
    max_vol_sc = math.ceil(duration)
    for i in C.P.bin_spacings:
        max_vol_dft[i] = C.P.DFT_BIN_MULT
    max_vol = U.ttf(max_vol_dft, max_vol_sc) + Phaser(C.P.TBC_FREQ).emit(max_vol_sc)
    amp_mul = 0.8 / np.max(np.abs(max_vol))

    while not chunks.is_eof:
        c_amp = 1 if clock else 0
        clock = not clock

        samples = round(duration_accumulator)
        duration_accumulator += duration - samples
        data = [c_amp, 1-c_amp] + [chunks.next() for i in range(len(C.P.bin_spacings)-2)]
        dft = np.zeros(C.P.dft_size)

        for inx, bit in enumerate(data):
            dft[C.P.bin_spacings[inx]] = bit * C.P.DFT_BIN_MULT

        signal = U.ttf(dft, samples) + tbc.emit(samples)
        yield signal * amp_mul


# def modulate():
    # chunks = read_aud_in_as_chunks()
    # chunks = U.ChunkEater(chunks, 0)
    # duration = C.P.MIN_FREQTIME * C.HZ
    # duration, duration_residue_inc = round(duration), duration-round(duration)
    # duration_accum = 0

    # def bin_to_freq(bin):
        # these are bins for the actual DFT later on in demodulation, not bins
        # for modulation. therefore we don't use duration as sample count, we
        # have to use demodulation sample sizes.
        # dft_size = U.datasize2dftsize(C.WIN_SIZE)
        # return U.index2freq(bin, dft_size, C.HZ)

    # bin_count = len(C.P.bin_spacings)
    # phx = [Phaser(bin_to_freq(i), bin_to_freq(i)) for i in C.P.bin_spacings[2:]]
    # tbc = Phaser(C.P.TBC_FREQ, C.P.TBC_FREQ)
    # timer = Phaser(bin_to_freq(C.P.bin_spacings[0]), bin_to_freq(C.P.bin_spacings[1]))
    # clock = False

    # while not chunks.is_eof:
        # clock = not clock
        # c_amp = 1 if clock else 0
        # accum_whole = round(duration_accum)
        # duration_accum += duration_residue_inc - accum_whole
        # chunk_duration = accum_whole + duration

        # amps = [chunks.next() for i in range(bin_count)]
        # amp_mul = 1/(sum(amps) + 1 + 2)
        # all_chunks = np.zeros(chunk_duration, dtype=C.TY)
        # for ph, amp in zip(phx, amps):
            # all_chunks += ph.emit(chunk_duration, 0, amp * amp_mul)
        # all_chunks += tbc.emit(chunk_duration, 0, 1 * amp_mul)
        # all_chunks += timer.emit(chunk_duration, c_amp, 1 * amp_mul)

        # yield all_chunks


def main():
    wr = U.WaveWriter(C.AUD_OUT_PATH, C.AUD_SAMPWIDTH, C.HZ, C.AUD_NPTYPE)
    total_time = 0
    before = time.time()
    with wr as file:
        for chunk in modulate():
            total_time += len(chunk) / C.HZ
            file.write(chunk)

    elapsed = time.time()-before
    filesize = os.path.getsize(C.DAT_IN_PATH)
    lin_filesize = os.path.getsize("./linux.dat")
    print("Generated %.1f seconds of audio in %.1f seconds (%.2f×)" % (total_time, elapsed, total_time/elapsed))
    print("Approx min/MiB: %.1f" % (total_time/60/filesize*1024*1024,))
    print("Approx final C-number w/ linux: C-%.0f (mins) \u00b1 wow&flutter & speed misconfiguration & ends" % (total_time/60/filesize*lin_filesize))


if __name__ == "__main__":
    main()

