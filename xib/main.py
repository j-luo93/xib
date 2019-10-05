from arglib import g, parse_args
from devlib import initiate

from .manager import Manager


def train():
    manager = Manager()
    manager.train()


if __name__ == "__main__":
    initiate(logger=True, log_dir=True, log_level=True, gpus=True)
    parse_args()

    train()
