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
WIN_ROLL = WIN_SIZE - WIN_ROFF
KHZ = HZ // 1000
AUD_SAMPWIDTH = 4  # 32 bits. will be signed
# Should be the smallest size of int greater than or equal to AUD_SAMPWIDTH bytes
AUD_NPTYPE = np.int32
MIN = -2**(8*AUD_SAMPWIDTH-1)
MAX = 2**(8*AUD_SAMPWIDTH-1)-1

assert WIN_ROFF * 2 < WIN_SIZE, "window rolloff is bigger than window size"

seed = 0x1507681e975e8d2ffa0a
rng = rng.Random(seed)


# Profile
class Profile:
    def __init__(self):
        # frequency modulation will have base FM_BASE, and can encode from 0 to FM_BASE
        self.FM_BASE = 2
        self.TOTAL_MINHZ = 200
        self.TOTAL_MAXHZ = 11000
        # allow 10% error by wow&flutter and battery voltage.
        self.TBC_ERR_AMT = .09
        
        # Since writing is the operation that limits min and max frequencies,
        # we can forego adding a boundary to the top of the TBC frequency, as
        # reading from the magnetic tape will allow higher frequencies.
        self.TBC_FREQ = self.TOTAL_MAXHZ
        self.TBC_HIGH = self.TBC_FREQ * (1+self.TBC_ERR_AMT)
        self.TBC_LOW = self.TBC_FREQ * (1-self.TBC_ERR_AMT)

        # The frequency boundaries for valid data frequencies
        self.MIN_AUDIOFREQ = self.TOTAL_MINHZ
        self.MAX_AUDIOFREQ = self.TBC_LOW / (1+self.TBC_ERR_AMT)

        # represents how long a certain frequency should last. is modified more.
        # 1 full cycle at the minimum frequency
        self.MIN_FREQTIME = 1/self.TOTAL_MINHZ

        # 37.5 milliseconds minimum +- wow & flutter error
        # self.MIN_FREQTIME = max((1+self.TBC_ERR_AMT)*.0375, self.MIN_FREQTIME)

        # rounded up to a whole # of windows + a boundary of 1ms, for
        # convenience's sake (and for ability to test what happens when wow &
        # flutter makes windows fall between chunks)
        self.MIN_FREQTIME = math.ceil(self.MIN_FREQTIME / (WIN_SIZE/HZ))*(WIN_SIZE/HZ) + 0.002

        self.WIN_SIZE = round(self.MIN_FREQTIME * HZ)

        # minimum DFT bin spacing when encoding
        self.DFT_BIN_DELTA = 3
        self.DFT_BIN_DELTA_MULT = 0  # 0.05
        # actual spacing is calculated as:
        # last_freq * DELTA_MULT + bin2hz(hz2bin(last_freq) + DELTA)
        # and rounded to nearest bin during actual encoding & decoding
        # bin spacings are actually bin indexes for a window size of WIN_SIZE
        from util import dat2dftsz, bin2hz, a_hz2bin
        self.dft_size = round(dat2dftsz(self.MIN_FREQTIME * HZ))
        self.bin_spacings = self.get_spacings()
        self.bin_bounds = [round(a_hz2bin(self.MIN_AUDIOFREQ, self.MIN_FREQTIME*HZ, HZ))-2]+list((self.bin_spacings+np.roll(self.bin_spacings, 1))/2)[1:]+[round(a_hz2bin(self.TBC_LOW, self.MIN_FREQTIME*HZ, HZ))]
        self.freq_spacings = np.array([bin2hz(i, self.dft_size, HZ) for i in self.bin_spacings], dtype=TY)
        self.freq_bounds = np.array([bin2hz(i, self.dft_size, HZ) for i in self.bin_bounds], dtype=TY)
        self.bin_bounds = [int(i) for i in self.bin_bounds]

        # What to multiply DFT amplitude by
        self.DFT_BIN_MULT = .5
        self.DFT_AMP_MUL = self.get_dft_amp_mul()

    def get_dft_amp_mul(self):
        from util import ttf
        from modulate import Phaser
        # bins = np.arange(starting_bin, ending_bin+1, C.P.DFT_BIN_DELTA, dtype=np.int16)
        max_vol_dft = np.zeros(self.dft_size)
        max_vol_sc = math.ceil(self.WIN_SIZE)
        for i in self.bin_spacings:
            max_vol_dft[i] = self.DFT_BIN_MULT
        max_vol = ttf(max_vol_dft, max_vol_sc) + Phaser(self.TBC_FREQ).emit(max_vol_sc)
        amp_mul = 0.8 / np.max(np.abs(max_vol))

        return amp_mul

    def get_spacings(self):
        from util import a_hz2bin
        win_size = round(self.MIN_FREQTIME * HZ / 2)
        # dft_size = self.dft_size

        # def increment_freq(freq):
        #     return freq * self.DFT_BIN_DELTA_MULT + bin2hz(a_hz2bin(freq, win_size, HZ)+self.DFT_BIN_DELTA, dft_size, HZ)

        # def increment_index(index):
        #     return a_hz2bin(increment_freq(bin2hz(index, dft_size, HZ)), win_size, HZ)

        # spacings = []
        # last_spacing = a_hz2bin(self.MIN_AUDIOFREQ, win_size, HZ)
        # max_bin = a_hz2bin(self.MAX_AUDIOFREQ, win_size, HZ)

        # while True:
        #     next_spacing = increment_index(last_spacing)
        #     next_spacing_rounded = round(next_spacing)
        #     if next_spacing_rounded >= max_bin:
        #         break
        #     spacings.append(next_spacing_rounded)
        #     last_spacing = next_spacing
        spacings = np.arange(
            math.ceil(a_hz2bin(self.MIN_AUDIOFREQ, win_size, HZ)),
            math.floor(a_hz2bin(self.MAX_AUDIOFREQ, win_size, HZ)),
            round(self.DFT_BIN_DELTA)
        ) * 2

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
