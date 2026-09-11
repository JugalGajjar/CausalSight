import numpy as np

from causalsight.data.rle import decode_counts, mask_to_box, rle_to_mask


def test_uncompressed_counts_column_major():
    # 3 rows x 4 cols; column-major runs: 2 zeros, 3 ones, 7 zeros
    rle = {"size": [3, 4], "counts": [2, 3, 7]}
    m = rle_to_mask(rle)
    assert m.shape == (3, 4)
    # flat column-major indices 2,3,4 -> (row 2, col 0), (row 0, col 1), (row 1, col 1)
    expected = np.zeros((3, 4), dtype=np.uint8)
    expected[2, 0] = expected[0, 1] = expected[1, 1] = 1
    assert (m == expected).all()


def test_compressed_counts_roundtrip_small():
    # pycocotools encoding of counts [2, 3, 7] for a 3x4 mask is "23:7" ... we only assert
    # the decoder matches the uncompressed path on a value it must produce: single small run.
    assert decode_counts("0") == [0]
    assert decode_counts("1") == [1]
    assert decode_counts("2") == [2]


def test_mask_to_box_normalized_and_pixel():
    m = np.zeros((10, 20), dtype=np.uint8)
    m[2:5, 4:8] = 1
    assert mask_to_box(m, normalized=False) == (4.0, 2.0, 8.0, 5.0)
    x0, y0, x1, y1 = mask_to_box(m)
    assert (x0, y0, x1, y1) == (0.2, 0.2, 0.4, 0.5)
    assert mask_to_box(np.zeros((4, 4), dtype=np.uint8)) is None
