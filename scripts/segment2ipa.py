import sys
from pathlib import Path

import torch

from xib.ipa.preprocess import (apply_all, clean_data, get_ipa_data,
                                get_pth_content, indexify, merge)

if __name__ == "__main__":
    in_path = Path(sys.argv[1])
    lang = sys.argv[2]
    out_path = Path(sys.argv[3])
    assert lang, 'Specify lang'

    with in_path.open('r', encoding='utf8') as fin:
        cnt, total, df = get_ipa_data(fin, progress=True)
        print(f'Ignore {cnt} / {total} lines.')

    apply_all(df, progress=True)
    cleaned_df = clean_data(df, progress=True)

    cleaned_df.to_csv(f'phones_{lang}.tsv', sep='\t', index=False)

    merged_df = merge(cleaned_df, progress=True)

    # Save intermediate merged results.
    merged_df.to_csv(f'phones_merged_{lang}.tsv', sep='\t', index=False)

    indexify(merged_df)

    out = get_pth_content(merged_df)
    torch.save(out, out_path)
