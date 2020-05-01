import argparse
import sys
from pathlib import Path

import tensorflow as tf

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder')
    parser.add_argument('--name', '-n', default='prf_exact_content_f1')
    parser.add_argument('--match_step', '-ms', default=-1, type=int)
    args = parser.parse_args()

    for run_folder in sorted(Path(args.folder).iterdir()):
        event_file = list(run_folder.glob('events*'))
        assert len(event_file) == 1
        event_file = str(event_file[0])

        last_value = (0, None)
        try:
            for record in tf.compat.v1.train.summary_iterator(event_file):
                try:
                    step = record.step
                    summary = record.summary
                    summary = summary.value[0]
                    tag = summary.tag
                    value = summary.simple_value
                    if tag == args.name and step > last_value[0]:
                        last_value = (step, value)
                except (IndexError, AttributeError):
                    pass
        except tf.errors.DataLossError:
            pass

        if args.match_step != -1 and args.match_step != last_value[0]:
            print('Mismatched step. Aborting.')
            exit(1)

        if args.match_step != -1:
            print(last_value[1])
        else:
            print(f'{run_folder.name}\t{last_value[0]}\t{last_value[1]}')
