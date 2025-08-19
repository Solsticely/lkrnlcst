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

    r = (lambda x: 1 - x) if r else (lambda x: x)
    thres = np.sort(data)[round(len(data) * r(pctile / 100))]
    if p:
        print(str(pctile) + "th %ile:", thres)
    return thres


def gauss(data, n=2, p=True):
    dev = np.std(data) * n
    mean = np.mean(data)
    if p:
        print(mean, "\u00b1", dev, str(n) + "\u03c3")
    return (mean, dev)


# NOTE: SEE jupyter notebook at experiments.ipynb!


if __name__ == "__main__":
    pass
