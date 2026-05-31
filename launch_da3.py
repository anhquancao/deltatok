from pathlib import Path

import torch.multiprocessing as mp

from occany.utils.runtime_paths import prepend_vendored_import_paths


mp.set_sharing_strategy("file_descriptor")

prepend_vendored_import_paths(Path(__file__).resolve().parent)

from occany.training_da3 import get_args_parser, train


if __name__ == "__main__":
    train(get_args_parser().parse_args())
