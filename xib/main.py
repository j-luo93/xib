from arglib import g, parse_args, show_args
from cfg import reg
from devlib import initiate
from devlib.named_tensor import patch_named_tensors
from xib.manager import DecipherManager, Manager, MetricLearningManager, AdaptManager


def train():
    if g.task == 'lm':
        manager = Manager()
    elif g.task == 'adapt':
        manager = AdaptManager()
    elif g.task == 'metric':
        manager = MetricLearningManager()
    else:
        manager = DecipherManager()
    manager.train()


if __name__ == "__main__":
    initiate(reg, logger=True, log_dir=True, log_level=True, gpus=True, random_seed=True, commit_id=True)
    patch_named_tensors()

    parse_args()
    show_args()

    train()
