import wave
import math
import numpy as np
from tqdm import tqdm
import bisect


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


def read_wave(path, size=1, size_type="duration"):
    assert size_type in ["duration", "samples"]
    with wave.open(path, "r") as inpt:
        assert inpt.getnchannels() == 1, "I only support mono audio"
        hz = inpt.getframerate()
        width = inpt.getsampwidth()
        max = 2**(8*width-1)-1
        block_size = size if size_type == "samples" else size * hz
        block_size, block_size_residue = divmod(block_size, 1)
        block_size = round(block_size)
        remainder = 0

        while True:
            size = block_size + math.floor(remainder)
            samples = inpt.readframes(size)
            remainder += block_size_residue - math.floor(remainder)
            if len(samples) == 0:
                break

            # turn into numbes
            as_bytes = [samples[i:i+width] for i in range(0, len(samples), width)]
            as_samples = [int.from_bytes(i, "little", signed=True) for i in as_bytes]
            as_np = np.array(as_samples, dtype=np.float32) / max
            yield (hz, np.concat((as_np, [0]*(size - len(as_np)))))


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


lerp = (lambda t, a, b: (b-1)*t+a)


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


def find_peak_freq(data, hz):
    """
        Warning: you can only call this when there is a singular peak!
        constants found in experiment_find_best_exponent_for_simple_peak_finding
    """
    fdomain = afft(data)
    dft_size = len(fdomain)
    as_weights = (fdomain * 0.0032222776) ** 10.213851
    as_weights = as_weights / np.sum(as_weights)

    index = np.sum(np.arange(dft_size) * as_weights)
    return index2freq(index, a_datasize2dftsize(len(data)), hz)

