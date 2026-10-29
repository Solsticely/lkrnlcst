import config as C
import util as U
from tqdm import tqdm
import numpy as np
# import math
# import mathutils as M

# Tuning considerations:
# --*---
# Type I cassette tape SNR source: https://en.wikipedia.org/wiki/Compact_Cassette_tape_types_and_formulations
# ~57 to 60 dBA. We'll go for 54 dBA
# Recording onto a cassette induces its own SNR. This site https://www.jaredrester.com/choosing-a-cassette-deck/
# recommends that a good cassette deck has a SNR of >55 dBA. But I'm just using a
# cheap shitty walkman to record, so maybe a SNR of ~45 to 52 dBA is a good idea?
# Maybe that's too low....
# We're not A-weighing the SNR as this is not for human audibility but for data
# storage.
# --*--
# Cassette audible freq range? I'm going for 500-11000 Hz, but this may be too
# limited.

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

# Or alternatively, when shuffling rows
# SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS··· <- Sync signal
#
# ADcBaDcbaD=BAdCBADcBA=CBadcBADcB=dcBadcbadcb=DCbADcBADCB=dCBadadCBa=cbADCbAdc··· <- Data lines
# BAdCbAdCBA=cBadcbADcb=dCbADCBADC=ADCBADCBADc=ADCBadcBADc=ADcBAbADcb=dCBADCbaD··· <-
# cBaDCbaDcb=dCBadCBADC=adCbadcBad=badcbadcbAD=badCBADCbAd=BadCBCBadc=aDCbADcba··· <-
# DCBAdCBAdC=aDcbaDcbAD=bADcbADCBA=CBADCBaDCBA=CbadCBaDCba=cbADcDcbAd=BaDCBAdcB··· <-
# ··········=··········=··········=···········=···········=··········=············ <-
# \--------/ \--------/ ···
#   page 1     page 2

# Blip:
#   --SSSSS--
#      Bias/Sync frequency
#   =-AAaAa-= =-BbBBb-= =-CcCCc-= =-dDdDd-= etc
#      Reed-solomon blocks
#   ----=----
#      Page boundary

# Each reed solomon block being a *row* as opposed to a column helps make recovery
# possible if one whole column gets corrupted. # TODO: maybe we can interleave
# pages too, to make pagegroups/books like CD/DVD CIRC coding, to help with burst
# whole-column errors.
# TODO: maybe also rotate the data lines, so in case of a persistent single
# frequency band / bin corruption we can correct 

# Since data is stored column-by-column, we need to further stripe the data (stripe
# as in how RAID storage does it) to pass to the modem. So the final data would be
# passed like --====ABcDAbCdAbCdAbCDaBcD====ABcD-- as opposed to --=AaAaaA=BbBbbB=--

def encode(instream: U.ChunkEater) -> np.ndarray:
    while not instream.is_eof:
        for column in encode_interleave_pages(instream):
            yield column

def decode(columns: U.ChunkEater) -> np.ndarray:
    while not columns.is_eof:
        for message in decode_interleave_pages(columns):
            yield message

# Encodes a book, which is a series of pages whose columns are interleaved
def encode_interleave_pages(instream: U.ChunkEater) -> np.ndarray:
    # TODO: Interleave pages, see other TODO comment under blip
    for column in encode_page(instream):
        yield column

def decode_interleave_pages(columns: U.ChunkEater) -> np.ndarray:
    for message in decode_page(columns):
        yield message

def encode_page(instream: U.ChunkEater) -> np.ndarray:
    # TODO: OPTIMISE: SLOW: use a 2D numpy table and just use column views instead
    # TODO: WET: Merge with decode_page somehow?
    rows: list[np.ndarray] = []

    # One RS block per row
    for blk_inx in range(C.B.page_row_count):
        rows.append(encode_message(instream))

    for row in rows:
        assert len(row) == C.B.page_col_count

    for col_inx in range(C.B.page_col_count):
        column = np.ndarray(C.B.page_row_count, dtype=np.uint16) # TODO: review dtypes
        for row_inx, row in enumerate(rows):
            column[row_inx] = row[col_inx]

        if col_inx == 0: breakpoint()
        yield column

def decode_page(columns_: U.ChunkEater) -> np.ndarray:
    # TODO: OPTIMISE: SLOW: use a 2D numpy table and just use column views instead
    # TODO: WET: Merge with encode_page somehow?
    column_count: int = C.B.rs_blk_size
    columns: list[np.ndarray] = []

    for col_inx in range(column_count):
        columns.append(columns_.next())

    for column in columns:
        assert len(column) == C.B.page_row_count

    for row_inx in range(C.B.page_row_count):
        row = np.ndarray(C.B.page_col_count, dtype=np.uint16)  # TODO: review dtypes
        for col_inx, col in enumerate(columns):
            row[col_inx] = column[row_inx]
        if row_inx == 0: breakpoint()
        yield decode_message(row)
    

def encode_message(instream: U.ChunkEater) -> np.ndarray:
    # TODO: do reed-solomon encoding
    data = np.zeros(C.B.rs_blk_size, dtype=np.uint8)  # TODO: review dtypes
    for i in range(C.B.rs_msg_size):
        data[i] = instream.next()

    return data # TODO: fix

def decode_message(message: np.ndarray) -> np.ndarray:
    # TODO: do reed-solomon error correction
    assert len(message) == C.B.rs_blk_size
    return message[:C.B.rs_msg_size]

# def encode_page(data: np.ndarray) -> np.ndarray:
#     return data # TODO: Implement this

# def decode_page(columns: np.ndrray) -> np.ndarray:
#     return
# TODO: re-design the ecc functions

# def encode(instream: U.ChunkEater):
#     assert C.E.GF_PRIME == 2, "Only binary error correction supported for now"
#     while not instream.is_eof:
#         # TODO: implement interleaving encoded chunks
#         for galois_element in encode_reed(instream):
#             for digit in galois_element.as_arr():
#                 yield digit

# def encode_chunk(data: U.ChunkEater) -> list:
#     # This function returns a list of prime coefficients as a numpy list
#     # list<galois_symbol<GF(p)>>

#     # TODO: this is very unoptimised
#     # chunks = [for sublist in zip(*[encode_reed(data) for i in range(C.E.rs_msg_sz)]) for i in sublist]
#     return np.array([j for i in chunks for j in i], dtype=np.uint8)

# def encode_reed(data: U.ChunkEater) -> list:
#     # This function returns a list of gf symbols as numpy lists
#     # list<galois_symbol<GF(p**k)>>
    
#     input = [] # This is a polynomial.
#     for i in range(C.E.rs_msg_sz):
#         symbol = [data.next() for i in range(C.E.GF_SIZE)]
#         input.append(C.E.gf.new(symbol))

#     codeword = []
#     for i in range(C.E.rs_blk_sz):
#         token = C.E.gf.zero()
#         for j in range(C.E.rs_msg_sz):
#             token += C.E.rs_seq_pow_pows[i][j] * input[j]
#         codeword.append(token)
#     return codeword

# def decode(instream: U.ChunkEater):
#     while not instream.is_eof:
#         yield decode_chunk(instream)

# def decode_chunk(instream: U.ChunkEater):
#     elements = []
#     for i in range(C.E.rs_blk_sz):
#         digits = [data.next() for i in range(C.E.GF_SIZE)]
#         elements.append(C.E.gf.new(digits))

#     # TODO: implement chunk deinterlacing! Should be simple with a basic codeword = chunk[n::int_count]
#     codeword = elements
#     return decode_reed(codeword)

# def decode_reed(code: list) -> list:
#     # https://web.archive.org/web/20120712150304/http://www.math.clemson.edu/~sgao/papers/RS.pdf
#     lagrange = C.E.poly.lagrange(list(zip(C.E.rs_seq_pows, (C.E.gf.new(i) for i in code))))
#     stop_deg = (C.E.rs_blk_sz+C.E.rs_msg_sz)/2
#     gcd, _gao_u, gao_v = C.E.rs_gao_poly.gcd(lagrange, stop_deg)
#     quot, rem = gcd.divmod(gao_v)
#     if rem != C.E.poly.zero() and quot.degree() < C.E.rs_msg_sz:
#         raise "Too many errors occured in block!"

#     return quot + [0]*(C.E.rs_msg_len-len(quot))

if __name__ == "__main__":
    
    bytes = b"i get mad whenever i see the self driving car from sanfrancisco"
    data = []
    columns = []
    for i in bytes:
        data += [int(i) for i in bin(i)[2:]]
    data *= 3
    print("Encoding %d bits of data" % (len(data)))

    encoded_page = encode_page(U.ChunkEater(data, 0))
    decoded_page = decode_page(U.ChunkEater(encoded_page, C.B.empty_page_col))

    print([*decoded_page])
    # print(decode_message(encode_message(U.ChunkEater(data, 0))))

    count = 0
    for i in tqdm(encode(U.ChunkEater(data, 0))):
        columns.append(i)
        count += 1

    duration = count * C.P.WIN_SIZE/C.HZ
    print("Encoded %d pages, equating to %.2fs of audio" % (count, duration))
    print(f"Target density: {'%.2f'%(len(data)/1024/duration)}Kib/s (kibibit)")

    newdata = []
    for i in decode(U.ChunkEater(columns, C.B.empty_page_col)):
        newdata.extend((int(j) for j in i))

    newbytes = b''
    for i in range(len(data)//8):
        bits = int(''.join(str(i) for i in newdata[8*i:8*i+8]),2)

    print("Decode result:")
    print(repr(newbytes))
    print("Expected:")
    print(repr(bytes))
        
    # print()
