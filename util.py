import wave
import math
import numpy as np
import config as C


def chunk(list, size, default=0):
    out = [list[i:i+size] for i in range(0, len(list), size)]
    out[-1] += (size-len(out[-1])) * [default]
    return out


def make_rolling_mask(window_size: int, offset_size: int):
    assert offset_size <= window_size
    ramp_size = min(window_size - offset_size, offset_size)
    plateau_size = window_size - ramp_size * 2
    factor = ramp_size / (window_size - offset_size) if ramp_size != 0 else 1
    return (
        np.concat((
            np.linspace(0, 1, ramp_size, False, dtype=C.TY),
            np.ones(plateau_size, dtype=C.TY),
            np.linspace(1, 0, ramp_size, False, dtype=C.TY)
        )) * factor,
        np.concat((
            np.linspace(1, 0, window_size - offset_size, False, dtype=C.TY),
            np.zeros(offset_size, dtype=C.TY)
        ))
    )


def zip_streams(parent, *children):
    children = list(children)
    children = [StreamEeater(child) for child in children]

    for block in parent:
        block_size = len(block)
        child_blocks = [child.take(block_size) for child in children]

        yield (block, *child_blocks)


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


_W_IO_DTYPES = {1: np.int8, 2: np.int16, 4: np.int32, 8: np.int64}
_W_IO_DTYPES = {w: np.dtype(_W_IO_DTYPES[w]).newbyteorder("little") for w in _W_IO_DTYPES}


class WaveReader:
    def __init__(self, path, size=1, size_type="duration"):
        assert size_type in ["duration", "samples"]
        self.inpt = wave.open(path, "rb")

        assert self.inpt.getnchannels() == 1, "I only support mono audio"
        self.hz = self.inpt.getframerate()
        self.width = self.inpt.getsampwidth()
        assert self.width in _W_IO_DTYPES, "Unsupported width"
        self.dtype = _W_IO_DTYPES[self.width]
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

        # turn into numbers
        as_samples = np.frombuffer(samples, self.dtype)
        as_np = np.array(as_samples, dtype=C.TY) / self.max
        return np.append(as_np, np.zeros(size-len(as_np)))

    def __iter__(self): return self
    def __enter__(self): return (self.hz, self)
    def __exit__(self, exc_type, exc_val, exc_tb): self.close()
    def close(self): self.inpt.close()


class WaveWriter:
    def __init__(self, path, width, hz):
        assert abs(math.log2(width)-round(math.log2(width))) < .00001
        assert type(width) is int, "Invalid width"
        assert width in _W_IO_DTYPES, "Non-power-of-two-width, or unsupported width"
        
        nptype = _W_IO_DTYPES[width]
        assert nptype.alignment == width == nptype.itemsize

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
        as_bytes = clamped.view(self.nptype).tobytes()
        self.file.writeframes(as_bytes)

    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): self.close()
    def close(self): self.file.close()


class SlidingReader:
    def __init__(self, chunks, window_size: int = C.WIN_SIZE, roll_size: int = C.WIN_ROLL, pad: bool = True):
        self.roll_size = roll_size
        self.window_size = window_size
        assert roll_size <= window_size, "Roll"
        self.inner = chunks.__iter__()
        self.samples = np.zeros(self.roll_size, dtype=C.TY)
        self.is_finished = False
        self.basic_mask, self.basic_mask_beginning = make_rolling_mask(self.window_size, self.roll_size)
        self.pad = pad

    def __next__(self):
        if self.is_finished and len(self.samples) < self.window_size:
            raise StopIteration()

        self.samples = self.samples[self.roll_size:]
        while len(self.samples) < self.window_size and not self.is_finished:
            try:
                self.samples = np.concat((self.samples, self.inner.__next__()))
            except StopIteration:
                self.is_finished = True
                pad_size = self.window_size - len(self.samples)
                if self.pad:
                    pad_size += self.window_size
                padding = np.zeros(pad_size, dtype=C.TY)
                self.samples = np.concat((self.samples, padding))

                break

        return self.samples[:self.window_size]

    def __iter__(self): return self


class SlidingWriter:
    def __init__(self, mask: np.ndarray, ends: np.ndarray, offset_size: int, ends_value):
        self.mask = mask
        self.ends = ends
        assert len(mask) == len(ends)
        self.roff = offset_size
        assert self.roff <= len(mask)
        self.buffer = ends * ends_value
        self.ends_value = ends_value

    def write(self, value):
        # steps: add, rotate, take
        self.buffer += self.mask * value
        self.buffer = np.roll(self.buffer, -self.roff)
        take = np.copy(self.buffer[-self.roff:])
        self.buffer[-self.roff:] = 0
        return take

    def clean_up(self, value=None):
        self.buffer += (1 - self.ends) * (value or self.ends_value)
        return self.buffer


class StreamSplitter:
    def __init__(self, stream_to_split):
        self.inner = stream_to_split.__iter__()
        self.pointers = []  # For each split, is the last block that's been yielded
        self.blocks = {}
        self.last = 0  # Stores the last block that's been yielded
        self.next = 0  # Stores the index of the next block to be generated
        self.terminates_at = None  # Stores the last yieldable block

    def split(self, count=1):
        splits =  tuple(self.__get_new_generator() for _i in range(count))
        return splits[0] if count == 1 else splits

    def __get_new_generator(self):
        index = len(self.pointers)
        self.pointers.append(self.next - 1)

        try:
            while True:
                yield self.__get_new_block_for_index(index)
        except StopIteration:
            return

    def __get_new_block_for_index(self, index):
        # Steps:
        # 1. if you're at the end, error and bail
        # 2. clean-up if you're the slowest pointer
        # 3. if there is a block to get, get block
        # 4. if there isn't a block to get, add block
        # 5. do stream termination logic

        prev_yield = self.pointers[index]

        # 1. if at end, bail
        if prev_yield == self.terminates_at:
            raise StopIteration()

        # 2. clean up if you're the slowest pointer
        if prev_yield == self.last:
            is_only_pointer_in_last = True

            for inx, pos in enumerate(self.pointers):
                if inx == index:
                    continue
                if pos == self.last:
                    is_only_pointer_in_last = False
                    break

            if is_only_pointer_in_last:
                self.blocks.pop(prev_yield)

        self.pointers[index] += 1
        next_yield = self.pointers[index]

        # 3. If there's a block already available, return it
        if next_yield != self.next:
            return self.blocks[next_yield]

        # 4. If we're here, we need a new block to return, so get that
        new_block = None

        try:
            new_block = self.inner.__next__()

        except StopIteration:
            # 5. Do termination logic
            self.terminates_at = prev_yield
            raise StopIteration()

        # Back to 4.
        self.blocks[self.next] = new_block
        self.next += 1

        return new_block


class StreamEater:
    def __init__(self, inner):
        self.inner = inner.__iter__()
        self.inner_is_open = True
        self.left = np.array([], dtype=C.TY)

    def take(self, amt):
        if amt is None:
            amt = len(self.left)

        while len(self.left) < amt and self.inner_is_open:
            try:
                data = self.inner.__next__()
                self.left = np.append(self.left, data)
            except StopIteration:
                self.inner_is_open = False

        # Pad with zeros
        output = self.left[:amt]
        if len(output) != amt:
            output = np.append(output, np.zeros(amt - len(output), dtype=C.TY))

        return output

    def take_back(self, data: np.ndarray):
        self.left = np.concat((self.left[:0], data, self.left))

    def __iter__(self):
        return self

    def __next__(self):
        if self.left:
            return self.take(None)
        return self.inner.__next__()


lerp = (lambda t, a, b: (b-a)*t+a)


# fft.size = data.size//2+1
# data.size = (fft.size-1)*2
# index = freq * data.size / hz
# freq = index * hz / data.size
# freq = index * hz / 2 / (fft.size - 1)
# curse you e731, e305 & e302
def bin2hz(index, dft_size, hz): return index * hz / (2*dft_size - 2)
def hz2bin(freq, data_size, hz): return round(freq * data_size / hz)
def a_hz2bin(freq, data_size, hz): return freq * data_size / hz
def dft2datsz(sz): return 2*sz-2
def dat2dftsz(sz): return sz//2+1
def a_dat2dftsz(sz): return sz/2+1


# Analytical FFT
def afft(data): return np.abs(np.fft.rfft(data)) / len(data)


# Phase-correct FFT (imaginary FFT)
def ifft(data): return np.fft.rfft(data) / len(data)


# Inverse FFT
def ttf(data, n=None):
    n = n or dft2datsz(len(data))
    return np.fft.irfft(data, n=n).real * n


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
    return bin2hz(index, a_dat2dftsz(len(data)), hz)


def speed_adjust(data, speed):
    if type(speed) is int:
        speed = np.ones_like(data) * speed

    final = []
    index = 0
    residue = 0
    while index+residue+1 < len(data):
        datapoint = lerp(residue, data[index], data[index+1])
        final.append(datapoint)

        residue += lerp(residue, speed[index], speed[index+1])
        whole, residue = divmod(residue, 1)
        index += round(whole)

    return np.array(final, dtype=data.dtype)


def bandpass(data, hz, low: None, high: None, data_is_dft=False):
    if data_is_dft:
        dat_size = dft2datsz(len(data))
        dft = np.copy(data)
    else:
        dat_size = len(data)
        dft = ifft(data)
    low_bin = 0 if low is None else hz2bin(low, dat_size, hz)
    high_bin = len(dft) if high is None else hz2bin(high, dat_size, hz)
    dft[:low_bin+1] = 0
    dft[high_bin:] = 0
    return dft if data_is_dft else ttf(dft, dat_size)


def guess_freq(block: np.ndarray, hz: float, default_if_error: float):
    # find zero crossings & frequency
    sign = np.sign(block)
    change = (sign - np.roll(sign, 1))[1:]
    zxings = np.arange(len(change), dtype=C.TY)[change > 0.5]

    distance = (
        (zxings - np.roll(zxings, 1))[1:] if len(zxings) > 0 else hz / default_if_error
    )
    distance = np.average(distance)  # samples per cycle
    distance /= hz  # seconds per cycle
    freq = 1 / distance  # hz

    return freq


def remove_dft_noise(
    dft: np.ndarray, target_bin_low: int, target_bin_high: int, strength: float,
    strength_is_magnitude: bool = False
):
    # separate sign & magnitude. to be joined later
    dft_sgn, dft_abs = np.sign(dft), np.abs(dft)

    # remove noise
    if strength_is_magnitude:
        dft_abs -= strength
    else:
        dft_abs -= np.median(dft_abs[target_bin_low:target_bin_high]) * strength

    dft_abs = np.maximum(dft_abs, 0)

    # rejoin sign & magnitude
    dft = dft_abs * dft_sgn

    return dft


