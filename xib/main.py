from arglib import parse_args, show_args, g
from devlib import initiate

from xib.manager import Manager, DecipherManager, MetricLearningManager
from cfg import reg


def train():
    if g.task == 'lm':
        manager = Manager()
    elif g.task == 'metric':
        manager = MetricLearningManager()
    else:
        manager = DecipherManager()
    manager.train()


if __name__ == "__main__":
    initiate(reg, logger=True, log_dir=True, log_level=True, gpus=True, random_seed=True)
    parse_args()
    show_args()

    train()
