from arglib import parse_args, show_args
from devlib import initiate

from manager import Manager
from cfg import reg


def train():
    manager = Manager()
    manager.train()


if __name__ == "__main__":
    initiate(reg, logger=True, log_dir=True, log_level=True, gpus=True)
    parse_args()
    show_args()

    train()
