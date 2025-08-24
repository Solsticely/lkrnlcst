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


def get_time_difference():
    pass


def speed_warp(samples, speed):
    warp_samples = []
    sample_offset = 0
    while math.floor(sample_offset)+1 < len(samples):
        floored = round(math.floor(sample_offset))
        sample = U.lerp(sample_offset-floored, samples[floored], samples[floored+1])
        warp_samples.append(sample)
        sample_offset += U.lerp(sample_offset-floored, speed[floored], speed[floored+1])
    return np.array(warp_samples, dtype=C.TY)


def time_base_correct(hz, file):
    # TODO: make sure TBC also does amplitude normalisation!!!
    expected_freq = C.P.TBC_FREQ
    WIN_SIZE = C.WIN_SIZE
    # NOTE: keep in mind that the standard window roll is C.WIN_SIZE-C.WIN_ROFF
    # this is a nonstandard roll
    ROLL_SIZE = C.WIN_ROFF
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

        yield U.speed_adjust(window[:ROLL_SIZE], speed)

    print(end="Speed deviance: (in %age; 2*\u03c3) ")
    E.gauss(speeds)

    # No need to yield remaining data in hz_swrt, since we used padding in
    # sliding reader


if __name__ == "__main__":
    file = U.WaveReader(C.AUD_IN_PATH)
    hz = file.hz

    total_time = 0
    before = time.time()
    with U.WaveWriter(C.AUD_OUT_PATH, C.AUD_SAMPWIDTH, hz) as outfile:
        for i in time_base_correct(hz, file):
            total_time += len(i)/hz
            outfile.write(i)

    elapsed = time.time() - before
    print("Processed %.1f seconds of audio in %.1f seconds (%.2f×)" % (total_time, elapsed, total_time/elapsed))
