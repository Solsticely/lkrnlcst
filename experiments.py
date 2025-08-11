import numpy as np
import math
from tqdm import tqdm
# from scipy import signal
import util as U
import config as C


def p_ile(data, pctile=95, p=True, r=False):
    if type(pctile) not in [float, int]:
        for i in pctile:
            p_ile(data, i, p, r)
        return

    r = (lambda x: 1-x) if r else (lambda x: x)
    thres = np.sort(data)[round(len(data)*r(pctile/100))]
    if p:
        print(str(pctile)+"th %ile:", thres)
    return thres


def gauss(data, n=2, p=True):
    dev = np.std(data)*n
    mean = np.mean(data)
    if p:
        print(mean, "\u00b1", dev, str(n)+"\u03c3")
    return (mean, dev)


def experiment_find_best_exponent_for_simple_peak_finding():
    def find_peak_freq(dft, hz, expn):
        """
            Warning: you can only call this when there is a singular peak!
        """
        as_weights = dft / np.sum(dft) * 4
        as_weights = as_weights ** expn
        as_weights = as_weights / np.sum(as_weights)

        index = np.sum(np.arange(len(dft)) * as_weights)
        return U.index2freq(index, len(dft), hz)

    hz = 44100
    wsize = hz // 50
    targets = np.arange(C.P.TOTAL_MINHZ, C.P.TOTAL_MAXHZ, 4, dtype=C.TY)
    waveforms = [
        U.afft(np.sin(np.arange(wsize, dtype=C.TY)*target/hz*np.pi*2))
        for target in tqdm(targets)
    ]
    expnts = np.log2(np.linspace(2**1, 2**18, 400, dtype=C.TY))

    best_expnts = []
    divnumbers = []

    for target, data in tqdm(list(zip(targets, waveforms))):
        best = (hz, expnts[0])
        for expnt in expnts:
            peak = find_peak_freq(data, hz, expnt)
            if abs(peak-target) < best[0]:
                best = (abs(peak-target), expnt)

        divnumbers.append(4/np.sum(data))
        best_expnts.append(best[1])
    expnt = np.mean(best_expnts)

    errors = []

    for target, data in zip(targets, waveforms):
        peak = find_peak_freq(data, hz, expnt)
        errors.append(peak-target)

    print("Best exponent:")
    gauss(best_expnts)
    print("Best multiplication constant for data:")
    p_ile(divnumbers, 99.5, r=True)
    gauss(divnumbers)
    print("Frequency error for best exponent:")
    p_ile(np.abs(errors), [50, 95, 99.5])
    gauss(errors)


def experiment_locate_low_frequency():
    file = U.WaveReader(C.AUD_OUT_PATH, C.WIN_SIZE * 32, "samples")
    hz = file.hz
    tbc_file = U.WaveWriter("./tbc_isolate.wav", 2, hz, np.int16)
    approxi_file = U.WaveWriter("./approxi.wav", 2, hz, np.int16)

    for chunk in file:
        tbc_isolate = U.bandpass(chunk, hz, C.P.TBC_LOW, C.P.TBC_HIGH)
        tbc_file.write(tbc_isolate)
        freq = U.find_peak_freq(tbc_isolate, hz, C.P.TBC_FREQ)
        approxi_file.write(np.sin(np.arange(C.WIN_SIZE * 32)*(freq/hz*np.pi*2)))

    file.close()
    tbc_file.close()
    approxi_file.close()


def experiment_locate_tbc_with_zxing():

    def find_zxings(signal):
        sign = np.sign(signal)
        change = sign - np.roll(sign, 1)
        change = change[1:]
        zxings = np.arange(len(change), dtype=C.TY)[change > 0.5]
        return zxings

    expected_freq = C.P.TBC_FREQ
    WIN_SIZE = C.WIN_SIZE
    ROLL_SIZE = C.WIN_ROFF
    WIN_SIZE *= 2

    file = U.WaveReader(C.AUD_IN_PATH, WIN_SIZE, "samples")
    hz = file.hz
    file = U.SlidingReader(file, WIN_SIZE, ROLL_SIZE, pad=True)
    hz_file = U.WaveWriter("./speed_variation.wav", 4, hz, np.int32)
    hz_swrt = U.SlidingWriter(file.basic_mask, file.basic_mask_beginning, ROLL_SIZE, 0)
    tbc_file = U.WaveWriter("./tbc_isolate.wav", 2, hz, np.int16)
    import modulate
    approxi_phaser = modulate.Phaser(0, 1)
    approxi_file = U.WaveWriter("./approxi.wav", 2, hz, np.int16)
    medians = []
    low_bin = U.freq2index(C.P.TBC_LOW, WIN_SIZE, hz) + 8  # TODO: remove me
    high_bin = U.freq2index(C.P.TBC_HIGH, WIN_SIZE, hz)

    for window in file:
        tbc_dft = U.ifft(window)

        # bandpass
        tbc_dft[:low_bin+1] = 0
        tbc_dft[high_bin:] = 0

        # separate sign & magnitude. to be joined later
        dft_sgn, tbc_dft = np.sign(tbc_dft), np.abs(tbc_dft)

        # remove noise
        tbc_dft -= np.median(tbc_dft[low_bin:high_bin]) * 1.5
        tbc_dft = np.maximum(tbc_dft, 0)

        # rejoin sign & magnitude
        tbc_dft = tbc_dft * dft_sgn

        # turn into time-domain again
        tbc = U.ttf(tbc_dft, len(window))

        # find zero crossings & frequency
        zxings = find_zxings(tbc)
        freq = expected_freq
        if len(zxings) > 2:
            distance = (zxings - np.roll(zxings, 1))[1:]
            distance = np.average(distance)  # samples per cycle
            distance /= hz  # now seconds per cycle
            freq = 1 / distance  # hz
            medians.append(freq)

        hz_file.write(hz_swrt.write(freq/expected_freq-1))
        tbc_file.write(tbc[:ROLL_SIZE])
        approxi_file.write(approxi_phaser.emit(ROLL_SIZE, freq, 0.8))

    hz_file.write(hz_swrt.clean_up())
    gauss(medians)


def experiment_find_control_points():
    ROLL_SIZE = C.WIN_SIZE // 18
    file = U.WaveReader(C.AUD_OUT_PATH, C.WIN_SIZE, "samples")
    hz = file.hz
    file = U.SlidingReader(file, C.WIN_SIZE, ROLL_SIZE, False)
    amp_file = U.WaveWriter("./amplitude.wav", 2, hz, np.int16)

    lp_freq = U.index2freq((C.P.bin_spacings[0]+C.P.bin_spacings[1])/2, U.a_datasize2dftsize(C.WIN_SIZE), hz)
    hp_freq = C.P.MIN_AUDIOFREQ
    for cl in file:
        max_freq_base = U.freq2index(C.P.TOTAL_MAXHZ, len(cl), hz)
        min_freq_base = U.freq2index(C.P.TOTAL_MINHZ, len(cl), hz)
        max_freq_signal = U.freq2index(lp_freq, len(cl), hz)
        min_freq_signal = U.freq2index(hp_freq, len(cl), hz)
        fft = U.afft(cl)
        amplitude = np.sum(fft[min_freq_signal:max_freq_signal])
        amplitude /= max(np.sum(fft[min_freq_base:max_freq_base]),0.0001)
        # amplitude = np.abs(signal.hilbert(chunk))
        amp_file.write(amplitude * np.ones(ROLL_SIZE))

    file.inner.close()
    amp_file.close()


if __name__ == "__main__":
    # experiment_find_best_exponent_for_simple_peak_finding()
    # experiment_locate_low_frequency()
    experiment_find_control_points()
    # experiment_locate_tbc_with_zxing()
    pass



