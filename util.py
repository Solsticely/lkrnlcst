import wave
import math
import numpy as np


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
        max = 256**width
        block_size = size if size_type == "samples" else size * hz
        block_size, block_size_residue = divmod(block_size, 1)
        remainder = 0

        while True:
            samples = inpt.readframes(block_size + math.floor(remainder))
            remainder += block_size_residue - math.floor(remainder)
            if len(samples) == 0:
                break

            # turn into numbes
            as_samples = [sum(samples[i+j]*2**j for j in range(width))/max for i in range(0, len(samples), width)]
            padded = np.pad(np.array(as_samples, dtype=np.float32), constant_values=0)
            yield (hz, padded)


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
    def __exit__(self, exc_type, exc_val, exc_tb): self.file.close()

