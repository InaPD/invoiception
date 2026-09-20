"""Page images must reach the API in a format it accepts, without silently degrading them."""

import base64

from PIL import Image

from eval.pages import ImagePart, encode_image


def _save(tmp_path, name, mode="RGB", size=(60, 80), fmt=None):
    path = tmp_path / name
    Image.new(mode, size, color=128).save(path, format=fmt)
    return path


def test_jpeg_passes_through_untouched(tmp_path):
    path = _save(tmp_path, "page.jpg")
    part = encode_image(path)
    assert isinstance(part, ImagePart)
    assert part.media_type == "image/jpeg"
    assert base64.standard_b64decode(part.data) == path.read_bytes()
    assert part.width == 60 and part.height == 80


def test_png_passes_through_untouched(tmp_path):
    path = _save(tmp_path, "page.png")
    part = encode_image(path)
    assert part.media_type == "image/png"
    assert base64.standard_b64decode(part.data) == path.read_bytes()


def test_tiff_is_re_encoded_as_png(tmp_path):
    path = _save(tmp_path, "scan.tif", mode="L")
    part = encode_image(path)
    assert part.media_type == "image/png"
    decoded = base64.standard_b64decode(part.data)
    assert decoded.startswith(b"\x89PNG")
    assert part.width == 60 and part.height == 80


def test_bilevel_tiff_survives_conversion(tmp_path):
    path = _save(tmp_path, "scan.tif", mode="1")
    part = encode_image(path)
    assert part.media_type == "image/png"


def test_oversized_page_is_downscaled_to_max_edge(tmp_path):
    path = _save(tmp_path, "big.png", size=(4000, 2000))
    part = encode_image(path, max_edge=1000)
    assert (part.width, part.height) == (1000, 500)
    assert part.media_type == "image/png"


def test_bytes_input_is_accepted(tmp_path):
    path = _save(tmp_path, "page.jpg")
    part = encode_image(path.read_bytes())
    assert part.media_type == "image/jpeg"
