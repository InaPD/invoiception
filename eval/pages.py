"""Turn a page image into something the model API accepts.

Both datasets are already page images (FATURA JPEG, RVL-CDIP TIFF). PDF rendering is the
serving layer's job (`serve/api.py`); once a PDF is rendered to pages, each page comes
through here, so eval and serving share one encoding path.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

#: Formats the API accepts as-is. Anything else is re-encoded as PNG (lossless).
_PASSTHROUGH = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp", "GIF": "image/gif"}

#: Longest edge above which a page is downscaled. The API resizes anything over 1568px
#: itself, so sending more only costs upload time; both datasets sit well under this.
DEFAULT_MAX_EDGE = 1568


@dataclass(frozen=True, slots=True)
class ImagePart:
    media_type: str
    data: str  # base64, no newlines
    width: int
    height: int


def encode_image(source: Path | bytes, *, max_edge: int = DEFAULT_MAX_EDGE) -> ImagePart:
    """Encode a page image for the API. Accepted formats pass through byte-for-byte."""
    raw = source if isinstance(source, bytes) else Path(source).read_bytes()
    with Image.open(io.BytesIO(raw)) as image:
        image.load()
        fmt = image.format
        width, height = image.size

        if fmt in _PASSTHROUGH and max(width, height) <= max_edge:
            return ImagePart(_PASSTHROUGH[fmt], _b64(raw), width, height)

        scale = min(1.0, max_edge / max(width, height))
        if scale < 1.0:
            image = image.resize((round(width * scale), round(height * scale)), Image.LANCZOS)
        # 1-bit and palette scans do not encode cleanly everywhere; greyscale/RGB PNG does.
        if image.mode not in ("L", "RGB"):
            image = image.convert("L" if image.mode == "1" else "RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        return ImagePart("image/png", _b64(buffer.getvalue()), image.width, image.height)


def _b64(data: bytes) -> str:
    return base64.standard_b64encode(data).decode("ascii")
