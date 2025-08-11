import numpy as np
import random as rng
from lazy import Lazy
import math


TY = np.float32
DAT_IN_PATH = "in.dat"
DAT_OUT_PATH = "out.dat"
AUD_IN_PATH = "in.wav"
AUD_OUT_PATH = "out.wav"
HZ = 44100  # samples per second
WIN_SIZE = HZ//50  # Window size, samples
WIN_ROFF = (4*WIN_SIZE)//9  # Window rolloff size, samples
KHZ = HZ // 1000
AUD_SAMPWIDTH = 3  # 24 bits. will be signed
# Should be the smallest size of int greater than or equal to AUD_SAMPWIDTH bytes
AUD_NPTYPE = np.int32
MIN = -2**(8*AUD_SAMPWIDTH-1)
MAX = 2**(8*AUD_SAMPWIDTH-1)-1

assert WIN_ROFF * 2 < WIN_SIZE, "window rolloff is bigger than window size"
WIN_MASK = np.concat((np.linspace(0, 1, WIN_ROFF, False, dtype=TY), np.ones(WIN_SIZE-2*WIN_ROFF, dtype=TY), np.linspace(1, 0, WIN_ROFF, False, dtype=TY)))
assert len(WIN_MASK) == WIN_SIZE

seed = 0x1507681e975e8d2ffa0a
rng = rng.Random(seed)


# Profile
class Profile:
    def __init__(self):
        # frequency modulation will have base FM_BASE, and can encode from 0 to FM_BASE
        self.FM_BASE = 2
        self.TOTAL_MINHZ = 200
        self.TOTAL_MAXHZ = 11000
        self.TBC_ERR_AMT = .5  # allow 50% error by wow&flutter
        self.TBC_HIGH = self.TOTAL_MAXHZ
        self.TBC_FREQ = self.TOTAL_MAXHZ / (1+self.TBC_ERR_AMT)
        self.TBC_LOW = self.TBC_FREQ * (1-self.TBC_ERR_AMT)

        # The frequency boundaries for valid data frequencies
        self.MIN_AUDIOFREQ = self.TOTAL_MINHZ
        self.MAX_AUDIOFREQ = self.TBC_LOW  # TODO: minus some boundary!

        # represents how long a certain frequency should last. is modified more.
        # 1 full cycle at the minimum frequency
        self.MIN_FREQTIME = 1/self.TOTAL_MINHZ
        # 75 milliseconds minimum +- wow & flutter error
        self.MIN_FREQTIME = max((1+self.TBC_ERR_AMT)*.075, self.MIN_FREQTIME)
        # rounded up to a whole # of windows, for convenience's sake (and for
        # ability to test what happens when wow & flutter makes windows fall
        # between chunks)
        self.MIN_FREQTIME = math.ceil(self.MIN_FREQTIME / (WIN_SIZE/HZ))*(WIN_SIZE/HZ)

        # minimum DFT bin spacing when encoding
        self.DFT_BIN_DELTA = 3.5
        self.DFT_BIN_DELTA_MULT = 0.05
        # actual spacing is calculated as:
        # last_freq * DELTA_MULT + index2freq(freq2index(last_freq) + DELTA)
        # and rounded to nearest bin during actual encoding & decoding
        # bin spacings are actually bin indexes for a window size of WIN_SIZE
        self.bin_spacings = self.get_spacings()

    def get_spacings(self):
        from util import index2freq, a_freq2index, a_datasize2dftsize
        dft_size = a_datasize2dftsize(WIN_SIZE)

        def increment_freq(freq):
            return freq * self.DFT_BIN_DELTA_MULT + index2freq(a_freq2index(freq, WIN_SIZE, HZ)+self.DFT_BIN_DELTA, dft_size, HZ)

        def increment_index(index):
            return a_freq2index(increment_freq(index2freq(index, dft_size, HZ)), WIN_SIZE, HZ)

        spacings = []
        last_spacing = a_freq2index(self.MIN_AUDIOFREQ, WIN_SIZE, HZ)
        max_bin = a_freq2index(self.MAX_AUDIOFREQ, WIN_SIZE, HZ)

        while True:
            next_spacing = increment_index(last_spacing)
            next_spacing_rounded = round(next_spacing)
            if next_spacing_rounded >= max_bin:
                break
            spacings.append(next_spacing_rounded)
            last_spacing = next_spacing

        return spacings


P = Lazy(Profile)


class Distortion:
    # allow 10% random change in distortion config every run
    DIST_ALTR = .1

    def a(self, x): return ((2*rng.random()-1) * self.DIST_ALTR + 1) * x

    def __init__(self):
        # noise in distortion, amplitude
        self.DIST_NOISE_AMP = self.a(.1)
        # periodic brownian noise variance, can change by 10%
        self.DIST_NOISE_VIB = self.a(.1)
        # brownian noise in time base distortion, 5%
        self.DIST_TBC_BROWN_AMT = self.a(.05)
        # time base distortion maximum variance, includes brown noise and vibrato (18%)
        self.DIST_TBC_MAX = .18


D = Distortion()
