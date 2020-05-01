from pathlib import Path
import sys
import re

def find(folder, folders):
    # Find all folders that contain a log file.
    fs = list(folder.iterdir())
    if any('log' == f.name for f in fs):
        folders.append(folder)
        return
    for f in fs:
        if f.is_dir():
            find(f, folders)
        
def try_link(target_path):
    link_path = save_root / target_path.relative_to(root)
    link_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        link_path.symlink_to(target_path.resolve())
    except FileExistsError:
        pass


if __name__ == '__main__':
    folders = list()

    root = Path(sys.argv[1])
    save_root = Path('./to_save')
    find(Path(root), folders)
    save_pat = re.compile(r'saved\.0_(\d+)\.latest')

    for folder in folders:
        log_file = folder / 'log'
        saves = list(folder.glob('saved*latest'))
        event_file = list(folder.glob('events*'))
        
        if saves:
            save_matches = [int(save_pat.match(save.name).group(1)) for save in saves]
            latest_save = folder / f'saved.0_{max(save_matches)}.latest'
            try_link(latest_save)
        try_link(log_file)
        for ef in event_file:
            try_link(ef)
