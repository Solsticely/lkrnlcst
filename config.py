DAT_IN_PATH = "in.dat"
DAT_OUT_PATH = "out.dat"
AUD_IN_PATH = "in.wav"
AUD_OUT_PATH = "out.wav"
HZ = 44100  # samples per second
KHZ = HZ // 1000
AUD_SAMPWIDTH = 2  # 24 bits. will be signed
MIN = -2**(8*AUD_SAMPWIDTH-1)
MAX = 2**(8*AUD_SAMPWIDTH-1)-1
FM_BASE = 16  # frequency modulation will have base FM_BASE, and can encode from 0 to FM_BASE

