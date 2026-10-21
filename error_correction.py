import config as C
import util as U
from tqdm import tqdm
import numpy as np
# import math
# import mathutils as M

# Tuning considerations:
# Type I cassette tape SNR source: https://en.wikipedia.org/wiki/Compact_Cassette_tape_types_and_formulations
# ~57 to 60 dBA. We'll go for 54 dBA

# ECC design:
# The resulting data is modulated in *pages*. Each page is an n×m matrix of bits (trits?)
# Each row of the page is a message encoded using reed-solomon-coding.
#
# The spectrogram looks like
#
# SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS··· <- Sync signal
#
# AAaAaAaAaA=aAaaaAAAAA=aAaAaAAAaA=AaAaAaaaAAA=AaaAaAaAAAa=AaAaAaAaAa=aaAAAAAaa··· <- Data lines
# BBBBbbBbBb=BBBbBbBbBb=bBbbbBBBBB=bBBBbBbBbBb=bbbBBBBBbbB=BbBBBbBbBb=BbBbBbbbB··· <-
# cCcCCCcCcC=cCcCcCcccC=CCCccCcCcC=CcCcCcCcCcc=CCCCCccCCCc=cCcCcCcCcc=cCCCCCccc··· <-
# DDdDdDdDdD=dDdddDDDDD=ddDdDdDDDd=dDdDdDdDdDD=DDddDdDDDDd=dDdDdDdDdd=dDDDDDddD··· <-
# ··········=··········=··········=···········=···········=··········=············ <-
# \--------/ \--------/ ···
#   page 1     page 2

# Blip:
#   --SSSSS--
#      Bias/Sync frequency
#   =-AAaAa-= =-BbBBb-= =-CcCCc-= =-dDdDd-= etc
#      Reed-solomon messages
#   ----=----
#      Page boundary

# Each reed solomon message being a *row* as opposed to a column helps make recovery
# possible if one whole column gets corrupted. # TODO: maybe we can interleave
# pages too, to make pagegroups/books like CD/DVD CIRC coding, to help with burst
# whole-column errors.

# Since data is stored column-by-column, we need to further stripe the data (stripe
# as in how RAID storage does it) to pass to the modem. So the final data would be
# passed like --====ABcDAbCdAbCdAbCDaBcD====ABcD-- as opposed to --=AaAaaA=BbBbbB=--

# def encode(file) -> np.ndarray:
#     yield np.array() # TODO: Implement this

# def encode_page(data: np.ndarray) -> np.ndarray:
#     return data # TODO: Implement this

# def decode_page(columns: np.ndrray) -> np.ndarray:
#     return
# TODO: re-design the ecc functions

def encode(instream: U.ChunkEater):
    assert C.E.GF_PRIME == 2, "Only binary error correction supported for now"
    while not instream.is_eof:
        # TODO: implement interleaving encoded chunks
        for galois_element in encode_reed(instream):
            for digit in galois_element.as_arr():
                yield digit

def encode_chunk(data: U.ChunkEater) -> list:
    # This function returns a list of prime coefficients as a numpy list
    # list<galois_symbol<GF(p)>>

    # TODO: this is very unoptimised
    # chunks = [for sublist in zip(*[encode_reed(data) for i in range(C.E.rs_msg_sz)]) for i in sublist]
    return np.array([j for i in chunks for j in i], dtype=np.uint8)

def encode_reed(data: U.ChunkEater) -> list:
    # This function returns a list of gf symbols as numpy lists
    # list<galois_symbol<GF(p**k)>>
    
    input = [] # This is a polynomial.
    for i in range(C.E.rs_msg_sz):
        symbol = [data.next() for i in range(C.E.GF_SIZE)]
        input.append(C.E.gf.new(symbol))

    codeword = []
    for i in range(C.E.rs_blk_sz):
        token = C.E.gf.zero()
        for j in range(C.E.rs_msg_sz):
            token += C.E.rs_seq_pow_pows[i][j] * input[j]
        codeword.append(token)
    return codeword

def decode(instream: U.ChunkEater):
    while not instream.is_eof:
        yield decode_chunk(instream)

def decode_chunk(instream: U.ChunkEater):
    elements = []
    for i in range(C.E.rs_blk_sz):
        digits = [data.next() for i in range(C.E.GF_SIZE)]
        elements.append(C.E.gf.new(digits))

    # TODO: implement chunk deinterlacing! Should be simple with a basic codeword = chunk[n::int_count]
    codeword = elements
    return decode_reed(codeword)

def decode_reed(code: list) -> list:
    # https://web.archive.org/web/20120712150304/http://www.math.clemson.edu/~sgao/papers/RS.pdf
    lagrange = C.E.poly.lagrange(list(zip(C.E.rs_seq_pows, (C.E.gf.new(i) for i in code))))
    stop_deg = (C.E.rs_blk_sz+C.E.rs_msg_sz)/2
    gcd, _gao_u, gao_v = C.E.rs_gao_poly.gcd(lagrange, stop_deg)
    quot, rem = gcd.divmod(gao_v)
    if rem != C.E.poly.zero() and quot.degree() < C.E.rs_msg_sz:
        raise "Too many errors occured in block!"

    return quot + [0]*(C.E.rs_msg_len-len(quot))

if __name__ == "__main__":
    bytes = b"i get mad whenever i see the self driving car from sanfrancisco" * 1
    data = []
    for i in bytes:
        data += [int(i) for i in bin(i)[2:]]
    print("Encoding %d symbols of data" % (len(data)*40))

    count = 0
    for i in tqdm(range(40)):
        for i in encode(U.ChunkEater(data, 0)):
            count += 1

    count /= len(C.P.bin_spacings) - 2
    duration = count * C.P.WIN_SIZE/C.HZ
    print("Encoded %d pages, equating to %.2fs of audio" % (count, duration))

    newdata = decode(U.ChunkEater(data, 0))
    newbytes = b''
    for i in range(len(data)//8):
        bits = int(''.join(str(i) for i in newdata[8*i:8*i+8]),2)

    print("Decode result:")
    print(repr(newbytes))
    print("Expected:")
    print(repr(bytes))
        
    # print()
