import wave
import math
import numpy as np
import config as C


def chunk(list, size, default=0):
    out = [list[i:i+size] for i in range(0, len(list), size)]
    out[-1] += (size-len(out[-1])) * [default]
    return out


class ChunkEater:
    def __init__(self, iterator, default):
        self.iter = iterator.__iter__()
        self.default = default
        self.counter = 0
        self.is_eof = False

    def next(self):
        try:
            next = self.iter.__next__()
            self.counter += 1
            return next
        except StopIteration:
            self.is_eof = True
            return self.default


class WaveReader:
    def __init__(self, path, size=1, size_type="duration"):
        assert size_type in ["duration", "samples"]
        self.inpt = wave.open(path, "rb")

        assert self.inpt.getnchannels() == 1, "I only support mono audio"
        self.hz = self.inpt.getframerate()
        self.width = self.inpt.getsampwidth()
        self.max = 2**(8*self.width-1)-1
        self.block_size = size if size_type == "samples" else size * self.hz
        self.block_size, self.block_size_residue = divmod(self.block_size, 1)
        self.block_size = round(self.block_size)
        self.remainder = 0

    def __next__(self):
        size = self.block_size + math.floor(self.remainder)
        samples = self.inpt.readframes(size)
        self.remainder += self.block_size_residue - math.floor(self.remainder)
        if len(samples) == 0:
            self.close()
            raise StopIteration()

        # turn into numbes
        as_bytes = [samples[i:i+self.width] for i in range(0, len(samples), self.width)]
        as_samples = [int.from_bytes(i, "little", signed=True) for i in as_bytes]
        as_np = np.array(as_samples, dtype=C.TY) / self.max
        return np.concat((as_np, [0]*(size - len(as_np))))

    def __iter__(self): return self
    def __enter__(self): return (self.hz, self)
    def __exit__(self, exc_type, exc_val, exc_tb): self.close()
    def close(self): self.inpt.close()


class WaveWriter:
    def __init__(self, path, width, hz, nptype):
        self.width = width
        self.max = 2**(8*self.width-1)-1
        self.min = -2**(8*self.width-1)
        self.factor = self.max * 0.7
        self.hz = hz
        self.nptype = nptype

        self.file = wave.open(path, "wb")
        self.file.setnchannels(1)
        self.file.setsampwidth(self.width)
        self.file.setframerate(self.hz)

    def write(self, frames):
        clamped = np.clip((frames * self.factor).astype(self.nptype), a_max=self.max, a_min=self.min)
        # TODO: this is slow
        as_bytes = (int.to_bytes(int(i), self.width, "little", signed=True) for i in clamped)
        self.file.writeframes(b"".join(as_bytes))

    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): self.close()
    def close(self): self.file.close()


def get_hz(path):
    with WaveReader(path) as (hz, _wv):
        return hz


lerp = (lambda t, a, b: (b-a)*t+a)


# fft.size = data.size//2+1
# data.size = (fft.size-1)*2
# index = freq * data.size / hz
# freq = index * hz / data.size
# freq = index * hz / 2 / (fft.size - 1)
# curse you e731, e305 & e302
def index2freq(index, dft_size, hz): return index * hz / (2*dft_size - 2)
def freq2index(freq, data_size, hz): return round(freq * data_size / hz)
def dftsize2datasize(sz): return 2*sz-2
def datasize2dftsize(sz): return sz//2+1
def a_datasize2dftsize(sz): return sz/2+1


# Analytical FFT
def afft(data): return np.abs(np.fft.rfft(data))


# Phase-correct FFT (imaginary FFT)
def ifft(data): return np.fft.rfft(data)


# Inverse FFT
def ttf(data, n=None): return np.fft.irfft(data, n=(n or len(data))).real


def find_peak_freq(data, hz, default_if_silent=None):
    """
        Warning: you can only call this when there is a singular peak!
        constants found in experiment_find_best_exponent_for_simple_peak_finding
    """
    fdomain = afft(data)
    dft_size = len(fdomain)
    as_weights = (fdomain * 0.0032222776) ** 10.213851
    cumsum = np.sum(as_weights)
    if cumsum == 0:
        if default_if_silent is not None: return default_if_silent
        raise Exception("silent audio passed to find_peak_freq. Set default_if_silent")
    as_weights = as_weights / cumsum

    index = np.sum(np.arange(dft_size) * as_weights)
    return index2freq(index, a_datasize2dftsize(len(data)), hz)


def speed_adjust(data, speed):
    final = []
    index = 0
    residue = 0
    while index+residue+1 < len(data):
        datapoint = lerp(residue, data[index], data[index+1])
        final.append(datapoint)

        residue += speed
        whole, residue = divmod(residue, 1)
        index += round(whole)

    return np.array(final)

