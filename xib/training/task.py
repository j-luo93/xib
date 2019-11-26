from dev_misc.trainlib.tracker.tracker import Task, task_class


@task_class
class LMTask(Task):
    name = 'lm'


@task_class
class DecipherTask(Task):
    name = 'decipher'
    split: str

    def __str__(self):
        return self.split


@task_class
class TransferTask(DecipherTask):
    name = 'transfer'
