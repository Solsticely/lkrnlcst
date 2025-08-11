def chunk(list, size, default=0):
    out = [list[i:i+size] for i in range(0, len(list), size)]
    out[-1] += (size-len(out[-1])) * [default]
    return out

