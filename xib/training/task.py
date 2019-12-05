from dev_misc.trainlib.tracker.tracker import Task, task_class


@task_class
class LMTask(Task):
    name = 'lm'


@task_class
class CbowTask(Task):
    name = 'cbow'


@task_class
class AdaptLMTask(Task):
    name = 'adapt_lm'


@task_class
class AdaptCbowTask(Task):
    name = 'adapt_cbow'


@task_class
class DecipherTask(Task):
    name = 'decipher'
    split: str

    def __str__(self):
        return self.split


@task_class
class TransferTask(DecipherTask):
    name = 'transfer'


@task_class
class MlmTask(DecipherTask):
    """This task is used for DecipherManager."""
    name = 'mlm'


@task_class
class ExtractTask(Task):
    name = 'extract'
