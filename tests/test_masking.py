import random

from PIL import Image

from causalsight.eval.masking import (
    mask_frames,
    masked_fraction,
    random_regions,
    sampled_frame_indices,
)


def test_sampled_indices():
    assert sampled_frame_indices(128, 16)[0] == 0 and sampled_frame_indices(128, 16)[-1] == 127
    assert sampled_frame_indices(10, 16) == list(range(10))


def test_mask_frames_only_inside_span():
    frames = [Image.new("RGB", (48, 32), (255, 255, 255)) for _ in range(16)]
    regions = [(0, 10, (0.0, 0.0, 0.5, 0.5))]  # only sampled frames 0 (orig 0) and 1 (orig 8) fall inside
    out = mask_frames(frames, regions, 128)
    assert out[0].getpixel((5, 5)) == (0, 0, 0) and out[0].getpixel((40, 25)) == (255, 255, 255)
    assert out[1].getpixel((5, 5)) == (0, 0, 0)
    assert out[2].getpixel((5, 5)) == (255, 255, 255)
    assert frames[0].getpixel((5, 5)) == (255, 255, 255)  # input untouched


def test_random_regions_preserve_size_and_span():
    regs = [(3, 9, (0.2, 0.3, 0.6, 0.5))]
    r = random_regions(regs, random.Random(1))[0]
    assert (r[0], r[1]) == (3, 9)
    assert abs((r[2][2] - r[2][0]) - 0.4) < 1e-9 and abs((r[2][3] - r[2][1]) - 0.2) < 1e-9
    assert 0 <= r[2][0] and r[2][2] <= 1 and 0 <= r[2][1] and r[2][3] <= 1


def test_masked_fraction():
    assert abs(masked_fraction([(0, 63, (0.0, 0.0, 0.5, 0.5))], 128) - 0.125) < 1e-9
