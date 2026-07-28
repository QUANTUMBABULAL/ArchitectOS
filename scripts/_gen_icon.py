"""One-off generator for the ArchitectOS desktop app icon set.

Pure-stdlib (no Pillow/ImageMagick available in this environment): builds
RGBA pixel buffers procedurally and encodes them as PNG (via zlib) and as a
multi-resolution Windows ICO (raw BI_RGB bitmap frames). Not meant to be
imported; run once to populate desktop/src-tauri/icons/.
"""

import struct
import zlib
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "desktop" / "src-tauri" / "icons"

# Diagonal gradient, deep blue -> violet, matching the app's dark theme.
COLOR_A = (37, 99, 235)    # #2563EB
COLOR_B = (124, 58, 237)   # #7C3AED


def lerp(a: int, b: int, t: float) -> int:
    return round(a + (b - a) * t)


def make_rgba(size: int) -> list[tuple[int, int, int, int]]:
    """Rounded-square gradient glyph, row-major, top-to-bottom."""
    radius = max(2, round(size * 0.22))
    pixels = []
    for y in range(size):
        for x in range(size):
            # distance outside the rounded-rect corners -> alpha falloff
            cx = min(x, size - 1 - x)
            cy = min(y, size - 1 - y)
            if cx < radius and cy < radius:
                dx = radius - cx
                dy = radius - cy
                inside = (dx * dx + dy * dy) <= radius * radius
            else:
                inside = True

            if not inside:
                pixels.append((0, 0, 0, 0))
                continue

            t = (x + y) / (2 * (size - 1)) if size > 1 else 0.0
            r = lerp(COLOR_A[0], COLOR_B[0], t)
            g = lerp(COLOR_A[1], COLOR_B[1], t)
            b = lerp(COLOR_A[2], COLOR_B[2], t)

            # simple inset "A" mark for larger sizes only
            pixels.append((r, g, b, 255))
    return pixels


def draw_mark(pixels: list[tuple[int, int, int, int]], size: int) -> None:
    """Overlay a simple triangular 'A' mark in white for legibility."""
    if size < 16:
        return
    margin = round(size * 0.28)
    top = margin
    bottom = size - margin
    left = margin
    right = size - margin
    mid_x = size / 2
    bar_y = round(bottom - (bottom - top) * 0.38)
    thickness = max(1, round(size * 0.09))

    def set_px(x: int, y: int) -> None:
        if 0 <= x < size and 0 <= y < size:
            idx = y * size + x
            r, g, b, a = pixels[idx]
            if a:
                pixels[idx] = (255, 255, 255, 255)

    for y in range(top, bottom):
        t = (y - top) / max(1, (bottom - top))
        half_width = (right - left) / 2 * t
        for edge in (-1, 1):
            xf = mid_x + edge * half_width
            for k in range(-thickness // 2, thickness // 2 + 1):
                set_px(round(xf) + k, y)

    for x in range(round(mid_x - (right - left) / 2 * 0.55), round(mid_x + (right - left) / 2 * 0.55)):
        for k in range(-thickness // 2, thickness // 2 + 1):
            set_px(x, bar_y + k)


def png_chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def encode_png(pixels: list[tuple[int, int, int, int]], size: int) -> bytes:
    raw = bytearray()
    for y in range(size):
        raw.append(0)  # filter type 0 (none)
        for x in range(size):
            r, g, b, a = pixels[y * size + x]
            raw += bytes((r, g, b, a))

    ihdr = struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0)
    compressed = zlib.compress(bytes(raw), 9)

    out = b"\x89PNG\r\n\x1a\n"
    out += png_chunk(b"IHDR", ihdr)
    out += png_chunk(b"IDAT", compressed)
    out += png_chunk(b"IEND", b"")
    return out


def encode_ico(sizes_pixels: dict[int, list[tuple[int, int, int, int]]]) -> bytes:
    sizes = sorted(sizes_pixels)
    n = len(sizes)
    header = struct.pack("<HHH", 0, 1, n)

    entries = b""
    image_data = b""
    offset = 6 + n * 16

    for size in sizes:
        pixels = sizes_pixels[size]

        # BITMAPINFOHEADER + XOR (BGRA, bottom-up) + AND mask (1bpp, padded)
        bih = struct.pack(
            "<IiiHHIIiiII",
            40, size, size * 2, 1, 32, 0, size * size * 4, 0, 0, 0, 0,
        )
        xor = bytearray()
        for y in range(size - 1, -1, -1):
            for x in range(size):
                r, g, b, a = pixels[y * size + x]
                xor += bytes((b, g, r, a))

        row_bytes = ((size + 31) // 32) * 4
        and_mask = bytearray(row_bytes * size)
        for y in range(size - 1, -1, -1):
            for x in range(size):
                _, _, _, a = pixels[(size - 1 - y) * size + x]
                if a == 0:
                    byte_index = y * row_bytes + (x // 8)
                    and_mask[byte_index] |= 0x80 >> (x % 8)

        image = bih + bytes(xor) + bytes(and_mask)
        image_data += image

        w_byte = 0 if size >= 256 else size
        h_byte = 0 if size >= 256 else size
        entries += struct.pack(
            "<BBBBHHII", w_byte, h_byte, 0, 0, 1, 32, len(image), offset
        )
        offset += len(image)

    return header + entries + image_data


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    png_sizes = {32: "32x32.png", 128: "128x128.png", 256: "128x128@2x.png"}
    ico_sizes = [16, 32, 48, 128, 256]

    all_pixels: dict[int, list[tuple[int, int, int, int]]] = {}
    for size in sorted(set(png_sizes) | set(ico_sizes)):
        px = make_rgba(size)
        draw_mark(px, size)
        all_pixels[size] = px

    for size, name in png_sizes.items():
        (OUT / name).write_bytes(encode_png(all_pixels[size], size))

    ico_bytes = encode_ico({s: all_pixels[s] for s in ico_sizes})
    (OUT / "icon.ico").write_bytes(ico_bytes)

    print(f"Wrote icons to {OUT}")


if __name__ == "__main__":
    main()
