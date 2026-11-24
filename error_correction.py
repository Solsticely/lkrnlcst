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

# def reed_solomon_encoder(instream: U.ChunkEater) -> U.ChunkEater:
#     def generator(instream: U.ChunkEater):
#         while not instream.is_eof:

#     return U.ChunkEater(generator(instream), C.B.empty_page_row)

def encode(instream: U.ChunkEater) -> np.ndarray:
    # instream = reed_solomon_encoder(instream)
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
    # TODO: make a shared interleave function that interleaves inputs with stride n
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
    msg_len = C.B.rs_msg_size * C.B.rs_alphabet_bitcount
    msg = np.ndarray(msg_len, dtype=np.uint8)  # TODO: review dtypes
    for i in range(len(msg)):
        msg[i] = instream.next()
    msg = bits_to_bytes(msg, C.B.rs_alphabet_bitcount, False)
    # TODO: write with numpy
    row = list(C.B.rs.encode(msg))
    assert len(row) == C.B.rs_blk_size
    row = np.array(bytes_to_bits(row, C.B.rs_alphabet_bitcount), dtype=np.uint8) # TODO: review dtypes
    assert len(row) == C.B.page_col_count

    return row

def decode_message(block: np.ndarray) -> np.ndarray:
    msg_len = C.B.rs_msg_size * C.B.rs_alphabet_bitcount
    if C.B.disable_ecc:
        return block[:msg_len]
    assert len(block) == C.B.page_col_count
    block = bits_to_bytes(block, C.B.rs_alphabet_bitcount, False)
    msg = C.B.rs.decode(block)
    assert len(msg) == C.B.rs_msg_size
    msg = bytes_to_bits(msg, C.B.rs_alphabet_bitcount)
    assert len(msg) == msg_len
    return msg

def bytes_to_bits(data: bytes | list[int], bits_per_symbol: int=8) -> list[int]:
    # MSB first
    assert bits_per_symbol > 0
    format_str = "%0"+str(bits_per_symbol)+"d"
    out = []
    for i in data:
        out.extend(int(i) for i in format_str%int(bin(i)[2:]))
    return out

def bits_to_bytes(data: list[int], bits_per_symbol: int=8, as_bytes: bool = True) -> list[int]:
    # MSB first
    assert bits_per_symbol > 0
    if as_bytes:
        assert bits_per_symbol <= 8
    out = []
    residue = ""
    for i in data:
        assert 0<=i<=1
        # assert isinstance(i,int)
        residue += str(i)
        while len(residue) >= bits_per_symbol:
            out.append(int(residue[:8],2))
            residue = residue[8:]

    if len(residue) != 0:
        print(f"WARNING: Extra residue bits when grouping bits: {residue}")

    return bytes(out) if as_bytes else out

def interleave(data: list, stride: int) -> list:
    # TODO: write with numpy (reshape -> transpose -> reshape)
    if stride == 1: return data
    assert stride > 1
    assert len(data)%stride == 0, "Interleaving operation will result in ungrouped excess"
    interleave_count = len(data)//stride
    out = [None]*len(data)
    for offset in range(stride):
        for index in range(interleave_count):
            out[index + offset * interleave_count] = data[index * stride + offset]

    assert None not in out
    return out
    

if __name__ == "__main__":
    # Set to 2, 3, 4 for fast decoding, set to -1 for fast iteration
    opt_level = -1
    C.warm_up(opt_level)

    sourcebytes = b"i get mad whenever i see the self driving car from sanfrancisco! "

    data = bytes_to_bits(sourcebytes)
    columns = []
    data *= 1 # Increase to 1024-8192 if you want to benchmark

    # Encode
    print("Encoding %d bits of data" % (len(data)))
    count = 0
    for i in tqdm(encode(U.ChunkEater(data, 0))):
        columns.append(i)
        count += 1

    duration = count * C.P.WIN_SIZE/C.HZ
    print("Encoded %d pages, equating to %.2fs of audio" % (count, duration))
    print(f"Target density: {'%.2f'%(len(data)/1024/duration)}Kib/s (kibibit)")

    # Corrupt
    # chance to corrupt this column
    CORRUPT_COL_CHANCE = 0.01
    # chance to corrupt this row
    CORRUPT_ROW_CHANCE = 0.30
    print(end="\nCorrupting codepoints")
    for col_inx, col in enumerate(columns):
        if C.rng.random()>CORRUPT_COL_CHANCE:
            continue
        for row_inx, row in enumerate(col):
            if C.rng.random()>CORRUPT_ROW_CHANCE:
                continue
            # print(end=f" {col_inx},{row_inx}")
            columns[col_inx][row_inx] ^= 1
    print('\n')

    # Decode
    newdata = []
    print("Decoding")
    for i in tqdm(decode(U.ChunkEater(columns, C.B.empty_page_col))):
        newdata.extend((int(j) for j in i))

    newbytes = bits_to_bytes(newdata)
    
    print("Decode result:")
    print(repr(newbytes[:len(sourcebytes)]))
    print("Expected:")
    print(repr(sourcebytes))

    # src_data = U.ChunkEater(bytes_to_bits(sourcebytes),0)
    # encoded_data = [*encode_interleave_pages(src_data)]
    # print(encoded_data, columns)
    # assert encoded_data == columns
    # encoded_data = U.ChunkEater(encoded_data, C.B.empty_page_col)
    # decoded_data = decode_interleave_pages(encoded_data)
    # decoded_data = bytes([byte for col in list(decoded_data) for byte in col])
    # decoded_data = bits_to_bytes(decoded_data)
    # print(decoded_data.rstrip(b"\0"))
    # assert bits_to_bytes(bytes_to_bits(sourcebytes, 7), 7) == sourcebytes
