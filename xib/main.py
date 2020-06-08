import logging

import torch

from dev_misc.arglib import g, parse_args, show_args
from dev_misc.devlib import initiate
from dev_misc.devlib.initiate import DirAlreadyExists
from dev_misc.devlib.named_tensor import (patch_named_tensors,
                                          register_tensor_cls)
from dev_misc.trainlib import set_random_seeds
from xib.cfg import reg
from xib.model.log_tensor import LogTensor
from xib.training.manager import ExtractManager


def train():
    manager = ExtractManager()
    manager.run()


if __name__ == "__main__":
    try:
        initiate(reg, logger=True, log_dir=True, log_level=True, gpus=True, random_seed=True, commit_id=True)
    except DirAlreadyExists:
        logging.exception('')
        exit()

    patch_named_tensors()
    register_tensor_cls(LogTensor)

    parse_args()
    show_args()
    set_random_seeds(g.random_seed)
    torch.set_printoptions(sci_mode=False)

    train()
