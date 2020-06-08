from dev_misc.trainlib.tracker.tracker import Task, task_class


@task_class
class ExtractTask(Task):
    name = 'extract'
    training: bool = False
