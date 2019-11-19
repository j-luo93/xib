import sys
from pathlib import Path

import torch
from devlib import initiate
from arglib import add_argument, parse_args, g

from xib.ipa.process import (apply_all, clean_data, get_ipa_data,
                                get_pth_content, indexify, merge)

if __name__ == "__main__":
    initiate(logger=True)
    add_argument('in_path', dtype='path')
    add_argument('lang', dtype=str)
    parse_args()

    with g.in_path.open('r', encoding='utf8') as fin:
        cnt, total, df = get_ipa_data(fin, progress=True)
        print(f'Ignore {cnt} / {total} lines.')

    folder: Path = g.in_path.parent

    apply_all(df, progress=True)
    cleaned_df = clean_data(df, progress=True)

    cleaned_df.to_csv(folder / f'phones_{g.lang}.tsv', sep='\t', index=False)

    merged_df = merge(cleaned_df, progress=True)

    # Save intermediate merged results.
    merged_df.to_csv(folder / f'phones_merged_{g.lang}.tsv', sep='\t', index=False)

    indexify(merged_df)

    out = get_pth_content(merged_df)
    torch.save(out, folder / f'phones_{g.lang}_idx.pth')
