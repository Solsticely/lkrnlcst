from error_correction import encode_reed, decode_reed
import pytest
import util as U

@pytest.fixture
def test_rs_input_data():
    bytes = b"i get mad whenever i see the self driving car from sanfrancisco" * 1
    data = []
    for i in bytes:
        data += [int(i) for i in bin(i)[2:]]
    print(''.join(str(i) for i in data))
    return data


def test_reed_solomon(test_rs_input_data):
    decode_reed(encode_reed(U.ChunkEater(test_rs_input_data, 0)))


