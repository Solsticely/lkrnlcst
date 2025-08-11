import numpy as np
import math
from tqdm import tqdm
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
    def low_pass_filter(data, freq, hz):
        F = U.ifft(data)
        F[U.freq2index(freq, len(data), hz) + 1:] = 0
        return U.ttf(F, data.size)

    file = U.WaveReader(C.AUD_IN_PATH, C.WIN_SIZE, "samples")
    hz = file.hz
    lowpass_file = U.WaveWriter("./lowpass.wav", 2, hz, np.int16)
    approxi_file = U.WaveWriter("./approxi.wav", 2, hz, np.int16)

    for chunk in file:
        lowpass = low_pass_filter(chunk, C.P.TBC_FREQ*(1+C.P.TBC_ERR_AMT), hz)
        lowpass_file.write(lowpass)
        freq = U.find_peak_freq(lowpass, hz, C.P.TBC_FREQ)
        approxi_file.write(np.sin(np.arange(C.WIN_SIZE)*(freq/hz*np.pi*2)))

    file.close()
    lowpass_file.close()
    approxi_file.close()


if __name__ == "__main__":
    # experiment_find_best_exponent_for_simple_peak_finding()
    experiment_locate_low_frequency()
    pass



