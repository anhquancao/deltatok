"""Encode a float32 depth .npy file as an 8-bit RGB PNG."""

import sys
import numpy as np
from PIL import Image
from occrae.depth_rgb import depth_to_rgb

# 8-bit quantization introduces a decoding floor (~1/255 per channel)
# that pure-float round-trip tests won't see.

def main() -> None:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <depth.npy> <out.png>")
        sys.exit(1)

    depth = np.load(sys.argv[1]).astype(np.float64)
    rgb_float = depth_to_rgb(depth)
    rgb_uint8 = np.clip(np.round(rgb_float * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(rgb_uint8).save(sys.argv[2])


if __name__ == "__main__":
    main()