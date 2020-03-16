import os
import uuid
from pathlib import Path
from typing import Dict, List

import bokeh
import pandas as pd
import torch
from bokeh.io import output_notebook
from bokeh.layouts import column, layout, row
from bokeh.models import (BasicTicker, ColorBar, ColumnDataSource, CustomJS,
                          FactorRange, LinearColorMapper, PrintfTickFormatter,
                          Rect, Slider)
from bokeh.models.tools import HoverTool, WheelZoomTool
from bokeh.plotting import figure, output_file, save, show
from IPython import embed

from dev_misc import FT, g
from dev_misc.arglib import set_argument
from dev_misc.devlib.named_tensor import NoName, patch_named_tensors
from dev_misc.utils import pbar
from xib.aligned_corpus.char_set import CharSet, CharSetFactory
from xib.aligned_corpus.data_loader import BaseAlignedBatch
from xib.aligned_corpus.vocabulary import Vocabulary
from xib.data_loader import DataLoaderRegistry
from xib.ipa import Category, should_include
from xib.model.extract_model import ExtractModel
from xib.training.task import ExtractTask

torch.set_printoptions(sci_mode=False)
patch_named_tensors()


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def load(prefix: str, suffix: str) -> FT:
    ext = '' if suffix == 'init' else '.almt'
    prefix = prefix if suffix == 'init' else f'{prefix}/almt'
    path = f'{prefix}/saved.{suffix}{ext}'
    saved_dict = torch.load(path)
    # Compute almt.
    almt = saved_dict['alignment'].detach()
    try:
        raw_weight = saved_dict['unit_aligner']['weight'].sigmoid().detach().cpu().numpy()
    except KeyError:
        raw_weight = None
    return almt, raw_weight


def get_heatmap_df(prefix: str, suffix: str, char_sets: Dict[str, CharSet], raw: bool = False):
    almt, raw_weight = load(prefix, suffix)
    lu = list(map(str, char_sets[g.lost_lang]))
    ku = list(map(str, char_sets[g.known_lang]))

    data = list()
    almt = almt.cpu().detach().numpy()
    for i, l in enumerate(lu):
        for j, k in enumerate(ku):
            if raw:
                p = raw_weight[i, j]
            else:
                p = almt[j, i]
            data.append((l, k, p))
    df = pd.DataFrame(data, columns=['lu', 'ku', 'weight' if raw else 'prob'])
    return lu, ku, df


def set_up_heatmap(title, lu, ku, raw=False):
    p = figure(
        x_range=ku, y_range=lu,
        x_axis_location="below", plot_width=650, plot_height=600, title=title)

    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_text_font_size = "5pt"
    p.axis.major_label_standoff = 0

    # if raw:
    #    mapper = LinearColorMapper(palette=bokeh.palettes.Cividis256)
    # else:
    mapper = LinearColorMapper(palette=bokeh.palettes.Cividis256, low=0.0, high=1.0)
    color_bar = ColorBar(color_mapper=mapper, major_label_text_font_size="5pt",
                         label_standoff=6, border_line_color=None, location=(0, 0))
    p.add_layout(color_bar, 'right')
    return p, mapper


def get_heatmap_hover(names, raw=False):
    TOOLTIPS = [
        ("prob", "@prob{0.000f}") if not raw else ('weight', '@weight{0.0000f}'),
        ('lu', '@lu'),
        ('ku', '@ku')
    ]
    hover = HoverTool(tooltips=TOOLTIPS, names=names)
    return hover


def iheatmap(prefix: str, title: str, suffixes: List[str], char_sets: Dict[str, CharSet], step_size=100, max_step=1000, raw=False):
    min_total_step = 1000000
    max_total_step = 0
    all_rect = dict()
    try:
        for i, suffix in pbar(enumerate(suffixes), text_only=True):
            lu, ku, df = get_heatmap_df(prefix, suffix, char_sets, raw=raw)
            # Set up heatmap the first time. Only the first heatmap is visible.
            if i == 0:
                p, mapper = set_up_heatmap(title, lu, ku, raw=raw)
                visible = True
            else:
                visible = False

            # Total step.
            if suffix == 'init':
                total_step = 0
            else:
                round_, step = map(int, suffix.split('.')[0].split('_'))
                total_step = round_ * max_step + step
            min_total_step = min(min_total_step, total_step)
            max_total_step = max(max_total_step, total_step)
            # Add glyph.
            rect = p.rect(x="ku", y="lu", width=1, height=1,
                          source=df,
                          line_color=None,
                          fill_color={'field': 'prob' if not raw else 'weight', 'transform': mapper}, name=str(total_step))
            rect.visible = visible
            all_rect[total_step] = rect
            # Add hover.
            hover = get_heatmap_hover([str(total_step)], raw=raw)
            p.add_tools(hover)
    except FileNotFoundError:
        pass
    # Configure slider.
    total_step_slider = Slider(start=min_total_step, end=max_total_step,
                               value=min_total_step, step=step_size, title="total_step")
    for rect_total_step, rect in all_rect.items():
        total_step_slider.js_on_change('value',
                                       CustomJS(args={'rect': rect, 'rect_total_step': rect_total_step},
                                                code="""
                                                     let visible = this.value === rect_total_step;
                                                     rect.visible = visible;
                                                     """
                                                ))
    return layout([p, total_step_slider])


def draw_iheatmap(prefix: str, title: str, char_sets: Dict[str, CharSet], vocab: Vocabulary, model: ExtractModel, num_rounds=4, step_size=100, max_step=1000, raw=False):
    saves = list()
    for round_ in range(0, num_rounds):
        for step in range(step_size, max_step + 1, step_size):
            saves.append(f'{round_}_{step}')
    img = iheatmap(prefix, title, saves, char_sets, step_size=step_size, max_step=max_step, raw=raw)
    return img


def init_setup(init_path, vocab_path, data_path):
    saved = torch.load(init_path)
    g.load_state_dict(saved['g'])
    set_argument('vocab_path', vocab_path, _force=True)
    set_argument('data_path', data_path, _force=True)

    dl_reg = DataLoaderRegistry()
    task = ExtractTask()
    dl = dl_reg.register_data_loader(task, g.data_path)

    dl = dl_reg[task]
    lost_cs = dl.dataset.corpus.char_sets[g.lost_lang]
    vocab = BaseAlignedBatch.known_vocab
    known_cs = vocab.char_set
    lu_size = len(lost_cs)
    ku_size = len(known_cs)
    char_sets = {g.lost_lang: lost_cs, g.known_lang: known_cs}

    model = ExtractModel(lu_size, ku_size, vocab)
    return char_sets, vocab, model.cuda()


def show_all(prefixes, titles, char_sets, vocab, model, num_rounds=5, step_size=50, max_step=1000, output='test.html'):
    output_file(output)
    to_show = list()
    for prefix, title in zip(prefixes, titles):
        print(title)
        this_row = list()
        # for raw in [True, False]:
        for raw in [False]:
            img = draw_iheatmap(prefix, title, char_sets, vocab, model, num_rounds=num_rounds,
                                step_size=step_size, max_step=max_step, raw=raw)
            this_row.append(img)
        to_show.append(row(this_row))
    to_show = column(to_show)
    show(to_show)


def get_service_function():
    """Turn this script into a service function."""

    init_path = '/scratch2/j_luo/xib/log/grid/matched_cmdl/sanity-old_downsample-longer-longer-batch1/saved.init'
    vocab_path = '/scratch2/j_luo/xib/data/wulfila/processed/germ.small.matched.stems'
    data_path = '/scratch2/j_luo/xib/data/wulfila/processed/corpus.small.got-germ.tsv'
    char_sets, vocab, model = init_setup(init_path, vocab_path, data_path)

    def run(project_root, prefixes, step_size, max_step, num_rounds) -> str:
        output = (project_root / 'plot' / uuid.uuid4().hex).with_suffix('.html')
        output.parent.mkdir(exist_ok=True)
        output = str(output)
        show_all(prefixes, prefixes, char_sets, vocab, model, output=output,
                 step_size=step_size, max_step=max_step, num_rounds=num_rounds)
        return output

    return run


if __name__ == "__main__":
    sf = get_service_function()
    embed()
