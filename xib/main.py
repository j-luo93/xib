import torch

from dev_misc.arglib import g, parse_args, show_args
from dev_misc.devlib import initiate
from dev_misc.devlib.named_tensor import patch_named_tensors
from dev_misc.trainlib import set_random_seeds
from xib.cfg import reg
from xib.training.manager import (AdaptCbowManager, AdaptLMManager,
                                  BaseManager, CbowManager, DecipherManager,
                                  ExtractManager, LMManager,
                                  SearchSolverManager)


def train():
    manager: BaseManager
    if g.task == 'lm':
        manager = LMManager()
    elif g.task == 'cbow':
        manager = CbowManager()
    elif g.task == 'adapt_cbow':
        manager = AdaptCbowManager()
    elif g.task == 'adapt':
        manager = AdaptLMManager()
    # elif g.task == 'transfer':
    #     manager = TransferManager()
    elif g.task == 'search':
        manager = SearchSolverManager()
    elif g.task == 'extract':
        manager = ExtractManager()
    else:
        manager = DecipherManager()
    manager.run()


if __name__ == "__main__":
    initiate(reg, logger=True, log_dir=True, log_level=True, gpus=True, random_seed=True, commit_id=True)
    patch_named_tensors()

    # IDEA(j_luo) Set random seed here?
    parse_args()
    show_args()
    set_random_seeds(g.random_seed)
    torch.set_printoptions(sci_mode=False)

    train()
