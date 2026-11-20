import config as C
import util as U
from tqdm import tqdm
import numpy as np
import galois

_unused = [galois]

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
    # TODO: Interleave pages, see other TODO comment under blip
    for message in decode_page(columns):
        yield message

def encode_page(rows: U.ChunkEater) -> np.ndarray:
    for column in page_codec(rows, True):
        yield column

def decode_page(columns: U.ChunkEater) -> np.ndarray:
    for row in page_codec(columns, False):
        yield row

def page_codec(data: U.ChunkEater, is_encoding: bool) -> np.ndarray:
    """
        Merged implementation of page encoding and coding.

        Since both page encoding and decoding offload actual reed-solomon processing
        to encode_message or decode_message respectively, the operation taking place
        in this function is a very simple table transposition.
    """
    # Variable naming convention: Role of variable in decoding is first
    shape = [C.B.page_col_count, C.B.page_row_count]
    if is_encoding:
        shape = shape[::-1]
    cols_or_rows: np.ndarray = np.ndarray(shape, dtype=np.uint16) # TODO: review dtypes

    for col_or_row_inx in range(shape[0]):
        if not is_encoding:
            cols_or_rows[col_or_row_inx] = data.next()
        else:
            cols_or_rows[col_or_row_inx] = encode_message(data)

    for col_or_row in cols_or_rows:
        assert len(col_or_row) == shape[1]

    # TODO: do column rolling/unrolling

    for row_or_col_inx, row_or_col in enumerate(np.transpose(cols_or_rows)):
        yield decode_message(row_or_col) if not is_encoding else row_or_col

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

def bytes_to_bits(data: bytes) -> list[int]:
    out = []
    for i in data:
        out.extend(int(i) for i in "%08d"%int(bin(i)[2:]))
    return out

def bits_to_bytes(data: list[int]) -> bytes:
    # MSB first
    out = b""
    residue = ""
    for i in data:
        assert 0<=i<=1
        assert isinstance(i,int)
        residue += str(i)
        while len(residue) >= 8:
            out += int(residue[:8],2).to_bytes()
            residue = residue[8:]

    if len(residue) != 0:
        print(f"WARNING: Extra residue bits when converting bits to bytes: {residue}")

    return out

if __name__ == "__main__":
    sourcebytes = b"i get mad whenever i see the self driving car from sanfrancisco! "
    assert bits_to_bytes(bytes_to_bits(sourcebytes)) == sourcebytes

    data = bytes_to_bits(sourcebytes)
    columns = []
    data *= 3
    print("Encoding %d bits of data" % (len(data)))

    # encoded_page = encode_page(U.ChunkEater(data, 0))
    # decoded_page = decode_page(U.ChunkEater(encoded_page, C.B.empty_page_col))

    # print(decode_message(encode_message(U.ChunkEater(data, 0))))

    count = 0
    for i in tqdm(encode(U.ChunkEater(data, 0))):
        columns.append(i)
        count += 1

    duration = count * C.P.WIN_SIZE/C.HZ
    print("Encoded %d pages, equating to %.2fs of audio" % (count, duration))
    print(f"Target density: {'%.2f'%(len(data)/1024/duration)}Kib/s (kibibit)")
    # print(columns)
    newdata = []
    for i in decode(U.ChunkEater(columns, C.B.empty_page_col)):
        newdata.extend((int(j) for j in i))

    newbytes = bits_to_bytes(newdata).rstrip(b"\0")

    print("Decode result:")
    print(repr(newbytes))
    print("Expected:")
    print(repr(sourcebytes))

    src_data = U.ChunkEater(sourcebytes,0)
    encoded_data = [*encode_interleave_pages(src_data)]
    # print(encoded_data, columns)
    # assert encoded_data == columns
    encoded_data = U.ChunkEater(encoded_data, C.B.empty_page_col)
    decoded_data = decode_interleave_pages(encoded_data)
    decoded_data = bytes([byte for col in list(decoded_data) for byte in col])
    print(decoded_data.rstrip(b"\0"))
