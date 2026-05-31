"""Decode an 8-bit RGB PNG back to a float32 depth .npy file."""

import sys
import numpy as np
from PIL import Image
from occrae.depth_rgb import rgb_to_depth


def main() -> None:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <in.png> <out.npy>")
        sys.exit(1)

    img = np.array(Image.open(sys.argv[1])).astype(np.float64) / 255.0
    depth = rgb_to_depth(img).astype(np.float32)
    np.save(sys.argv[2], depth)


if __name__ == "__main__":
    main()