import numpy as np
import random as rng


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


# Profile
class Profile:
    def __init__(self):
        # frequency modulation will have base FM_BASE, and can encode from 0 to FM_BASE
        self.FM_BASE = 16
        self.TOTAL_MINHZ = 200
        self.TOTAL_MAXHZ = 11000
        self.TBC_ERR_AMT = .2  # allow 20% error by wow&flutter
        self.TBC_FREQ = self.TOTAL_MINHZ / (1-self.TBC_ERR_AMT)
        # represents how long a certain frequency should last. is modified more.
        # 1 full cycle at the minimum frequency
        self.MIN_FREQTIME = 1/self.TOTAL_MINHZ
        # 5 milliseconds minimum +- wow & flutter error
        self.MIN_FREQTIME = max((1+self.TBC_ERR_AMT)*.005, self.MIN_FREQTIME)
        self.MIN_AUDIOFREQ = self.TBC_FREQ*(1+self.TBC_ERR_AMT)  # TODO: plus some boundary!


P = Profile()


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
