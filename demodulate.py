import config as C
import util as U
import numpy as np
import math
import experiments as E
import time

# TODO: do noise reduction


def low_pass_filter(data, freq, hz):
    F = U.ifft(data)
    F[U.hz2bin(freq, len(data), hz) + 1:] = 0
    return U.ttf(F, data.size)


def speed_warp(samples, speed):
    warp_samples = []
    sample_offset = 0
    while math.floor(sample_offset)+1 < len(samples):
        floored = round(math.floor(sample_offset))
        sample = U.lerp(sample_offset-floored, samples[floored], samples[floored+1])
        warp_samples.append(sample)
        sample_offset += U.lerp(sample_offset-floored, speed[floored], speed[floored+1])
    return np.array(warp_samples, dtype=C.TY)


def read_noise_level(hz: int, stream: U.StreamEater):
    THRESHOLD = 1.05  # 110% of baseline noise level
    ADDED_THRESHOLD = 10**-10  # -100 dbfs
    NOISE_TAKE_SIZE = round(C.P.WIN_SIZE / C.HZ * hz)
    # Take noise level
    noise_data = stream.take(NOISE_TAKE_SIZE)
    stream.take_back(noise_data)
    noise_level = (ADDED_THRESHOLD + np.max(np.abs(noise_data))) * THRESHOLD

    print(
        "Approximating noise level: Using %.2f seconds, %.2f dBFS (%.2f)"
        % (NOISE_TAKE_SIZE / hz, 10 * math.log10(noise_level + 2**-1044), noise_level)
    )

    return noise_data, noise_level


# TODO: implement dc-bias removal
def skip_to_amplitude(
    hz: int, stream: U.StreamEater, noise_level: float, invert=False,
    do_print: bool = False, save: bool = False
):
    TAKE_DURATION = 1 / C.P.TOTAL_MINHZ
    TAKE_SIZE = round(hz * TAKE_DURATION)
    total_duration = 0
    stream_data = np.array([], dtype=C.TY)

    # Skip to when noise is over
    while True:
        if stream.is_finished:
            print("Warning: Stream finished while skipping to amplitude!")
            break

        total_duration += TAKE_DURATION
        data = stream.take(TAKE_SIZE)
        level = np.max(np.abs(data))

        if save:
            stream_data = np.append(stream_data, data)

        ready_to_break = level > noise_level
        if invert:
            ready_to_break = not ready_to_break
        if ready_to_break:
            stream.take_back(data)
            break

    if do_print:
        print("Guessing that data starts at %.5fs in the audio sample" % total_duration)

    return stream_data if save else None


def read_header(hz: int, stream: U.StreamEater):
    noise_data, noise_level = read_noise_level(hz, stream)

    # Skip to frequency bias test and start of audio
    skip_to_amplitude(hz, stream, noise_level, do_print=True)

    # Save data and run test
    speed_test_data = skip_to_amplitude(hz, stream, noise_level, True, do_print=True, save=True)
    process_speed_test(hz, speed_test_data, stream)

    # Skip to sweep test
    skip_to_amplitude(hz, stream, noise_level, False, do_print=True)
    # Skip over sweep test
    skip_to_amplitude(hz, stream, noise_level, True, do_print=True)

    # Skip to white noise test
    skip_to_amplitude(hz, stream, noise_level, False, do_print=True)
    # Skip over white noise test
    skip_to_amplitude(hz, stream, noise_level, True, do_print=True)

    # Skip to training profile test
    skip_to_amplitude(hz, stream, noise_level, False, do_print=True)
    # Skip over training profile test
    skip_to_amplitude(hz, stream, noise_level, True, do_print=True)

    # Skip to magic
    skip_to_amplitude(hz, stream, noise_level, False, do_print=True)
    # Skip over magic
    skip_to_amplitude(hz, stream, noise_level, True, do_print=True)

    # Skip to actual data
    skip_to_amplitude(hz, stream, noise_level, False, do_print=True)


def process_speed_test(hz: int, data: np.ndarray, stream: U.StreamEater):
    def adjust_speed(stream, speed):
        for i in stream:
            yield U.speed_adjust_const(i, speed)
    freq = U.guess_freq(data, hz, C.P.TBC_FREQ)
    speed = C.P.TBC_FREQ / freq
    stream.inner = adjust_speed(stream.inner, speed)
    pass


# TODO: add an intermediate step to normalise volume
def read_data(hz, stream):
    CHUNK_SAMPSIZE = round(hz * C.P.MIN_FREQTIME)

    stream = U.remove_dc_bias(hz, stream, C.P.TOTAL_MINHZ)
    stream = U.StreamEater(stream)

    header = read_header(hz, stream)
    if [header, 0][1] + 1 is None:
        print(0)

    # for i in range(30):
    stream = time_base_correct(hz, stream)
    stream = U.SlidingReader(stream, CHUNK_SAMPSIZE, CHUNK_SAMPSIZE)

    bin_bounds = [U.hz2bin(i, CHUNK_SAMPSIZE, hz) for i in C.P.freq_bounds]
    value_thres = C.P.DFT_BIN_MULT * C.P.DFT_AMP_MUL / 2

    bin_c = ((len(C.P.bin_spacings) >> 1) << 1) - 2
    bits = 0
    bit_c = 0
    final = b""
    t_bit_c = 0

    for chunk in stream:
        try:
            dft = U.afft(chunk)
        except ValueError:
            break

        bins = []

        for i in range(bin_c):
            bin_value = np.sum(dft[bin_bounds[i] : bin_bounds[i + 1]+1])
            bins.append(bin_value)

        for a, b in [tuple(bins[i : i + 2]) for i in range(bin_c)[::2]]:
            is_high = 1 if a < b else 0
            bits = (bits << 1) | is_high
            bit_c += 1
            if bit_c == 8:
                final += int.to_bytes(bits)
                if b'\n'[0] <= bits < 128:
                    print(end="\033[36;1m%s\033[0m"%int.to_bytes(bits).decode("ascii"), flush=True)
                bit_c, bits = 0, 0

    final += int.to_bytes(bits)


def time_base_correct(hz, file):
    expected_freq = C.P.TBC_FREQ
    WIN_SIZE = round(hz/C.P.TBC_FREQ * 3000)
    # print(WIN_SIZE, C.P.WIN_SIZE, C.WIN_SIZE)
    ROLL_SIZE = WIN_SIZE//18
    WIN_SIZE *= 2

    file = U.SlidingReader(file, WIN_SIZE, ROLL_SIZE, pad=True)
    hz_swrt = U.SlidingWriter(file.basic_mask, file.basic_mask_beginning, ROLL_SIZE, expected_freq)
    low_bin = U.hz2bin(C.P.TBC_LOW, WIN_SIZE, hz)
    high_bin = U.hz2bin(C.P.TBC_HIGH, WIN_SIZE, hz)
    speeds = []

    for window in file:
        tbc_dft = U.ifft(window)

        # bandpass
        tbc_dft[:low_bin+1] = 0
        tbc_dft[high_bin:] = 0

        # remove noise
        tbc_dft = U.remove_dft_noise(tbc_dft, low_bin, high_bin, 1.5)

        # turn into time-domain again
        tbc = U.ttf(tbc_dft, len(window))

        # find wow&flutter-altered time base frequency
        freq = U.guess_freq(tbc, hz, expected_freq)

        # find speed change required to restore frequency
        # speed = np.clip(expected_freq/hz_swrt.write(freq), 1-C.P.TBC_ERR_AMT, 1+C.P.TBC_ERR_AMT)
        speed = expected_freq/hz_swrt.write(freq)
        speeds.append(expected_freq/freq * 100 - 100)

        # TODO: reconsider approximation, maybe debug it
        yield U.speed_adjust(window[:ROLL_SIZE], speed, approx=1==1)

    print(end="\nSpeed deviance: (in %age; 2*\u03c3) ")
    E.gauss(speeds)

    # No need to yield remaining data in hz_swrt, since we used padding in
    # sliding reader


if __name__ == "__main__":
    total_time = 0
    before = time.time()

    file = U.WaveReader(C.AUD_IN_PATH)
    hz = file.hz

    read_data(file.hz, file)

    elapsed = time.time() - before
    print("Processed %.1f seconds of audio in %.1f seconds (%.2f×)" % (total_time, elapsed, total_time/elapsed))
