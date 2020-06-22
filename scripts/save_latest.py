from pathlib import Path
import re
import sys

if __name__ == '__main__':
    log_folder = Path(sys.argv[1])
    to_gather = Path(sys.argv[2])
    pat = sys.argv[3]
    num_runs = int(sys.argv[4])

    to_gather.mkdir(exist_ok=True)

    pat = re.compile(fr'{pat}')


    def try_link(link, target):
        try:
            link.symlink_to(target)
        except FileExistsError:
            print(f'--[W] Link {link} already exists.')
        

    for folder in log_folder.iterdir():
        if pat.search(str(folder)):
            print(f'Found {folder}')
            for run_id in range(num_runs):
                run_folder = folder / str(run_id)
                if not run_folder.exists():
                    print(f'--[E] Run {run_id} not found.')
                else:
                    dest = to_gather / folder.name / str(run_id)
                    saves = [save.name for save in run_folder.glob('saved.0*.latest')]
                    if saves:
                        dest.mkdir(parents=True, exist_ok=True)
                        steps = [int(save.split('.')[1].split('_')[1]) for save in saves]
                        max_step = max(steps)
                        print(f'--[I] Max step {max_step} for run id {run_id}.')
                        log_dest = dest / 'log'
                        try_link(log_dest, (run_folder / 'log').resolve())
                        save_name = f'saved.0_{max_step}.latest'
                        save_dest = dest / save_name
                        try_link(save_dest, (run_folder / save_name).resolve())
                    else:
                        print(f'--[E] No save found for run id {run_id}.')


