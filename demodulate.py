import config as C
import util as U
import numpy as np
import math
import experiments as E

# TODO: do noise reduction


def low_pass_filter(data, freq, hz):
    F = U.ifft(data)
    F[U.freq2index(freq, len(data), hz) + 1:] = 0
    return U.ttf(F, data.size)


def get_time_difference():
    pass


def time_base_correct():
    def iter_speed_adjust(chunks, speed):
        for chunk in chunks:
            yield U.speed_adjust(chunk, speed)

    roll_size = C.WIN_SIZE - C.WIN_ROFF
    rolling_speed = np.copy(C.WIN_MASK)
    file = U.WaveReader(C.AUD_IN_PATH, C.WIN_SIZE, "samples")
    hz = file.hz
    tbc_samples = []
    speeds = []

    # NOTE 1
    # TODO: dirty fix: input with a different rate than C.HZ is misconfigured
    # for find_peak_freq, returning the wrong frequencies. until we fix
    # find_peak_freq to use a different method (likely will be zero xing) we
    # will pretend the input is the correct frequency. If note is in effect,
    # All future lines with this hack applied will have a   # see NOTE 1
    # comment next to them.
    hzmul = C.HZ / hz  # see NOTE 1

    file = iter_speed_adjust(file, hzmul)
    file = U.SlidingReader(file)

    for samples in file:
        rolling_speed[:roll_size] = 0
        rolling_speed = np.roll(rolling_speed, -roll_size)

        # see NOTE 1
        tbc_wave = U.bandpass(samples, C.HZ, C.P.TBC_LOW, C.P.TBC_HIGH, False)
        freq = U.find_peak_freq(tbc_wave, C.HZ, C.P.TBC_FREQ)

        speed = max(min(1+C.P.TBC_ERR_AMT, C.P.TBC_FREQ/freq), 1-C.P.TBC_ERR_AMT)
        speeds.append(speed*100-100)
        rolling_speed += C.WIN_MASK * speed

        sample_offset = 0
        while sample_offset < roll_size:
            # floored = round(sample_offset)
            # sample = samples[floored]
            floored = round(math.floor(sample_offset))
            # sample = samples[floored]
            sample = U.lerp(sample_offset-floored, samples[floored], samples[floored+1])
            tbc_samples.append(sample)
            sample_offset += U.lerp(sample_offset-floored, rolling_speed[floored], rolling_speed[floored+1])
        # yield rolling_speed[:roll_size]-1

        while len(tbc_samples) > C.WIN_SIZE:
            print(end=".", flush=True)
            yield np.array(tbc_samples[:C.WIN_SIZE])
            tbc_samples = tbc_samples[C.WIN_SIZE:]

    print()
    print(end="Speed deviance: (in %age; 95th %ile / 2*\u03c3) ")
    E.gauss(speeds)
    tbc_samples += [0]*(C.WIN_SIZE - len(tbc_samples) + 1)
    yield np.array(tbc_samples[:C.WIN_SIZE])
    file.inner.close()


# def write_wave():
# oupt = wave.open(C.AUD_IN_PATH, "w")

if __name__ == "__main__":
    # hz = U.get_hz(C.AUD_IN_PATH)
    hz = C.HZ  # see NOTE 1 in time_base_correct
    with U.WaveWriter(C.AUD_OUT_PATH, C.AUD_SAMPWIDTH, hz, C.AUD_NPTYPE) as file:
        for i in time_base_correct():
            file.write(i)

