from arglib import parse_args, show_args, g
from devlib import initiate

from manager import Manager, DecipherManager
from cfg import reg


def train():
    if g.task == 'lm':
        manager = Manager()
    else:
        manager = DecipherManager()
    manager.train()


if __name__ == "__main__":
    initiate(reg, logger=True, log_dir=True, log_level=True, gpus=True)
    parse_args()
    show_args()

    train()
