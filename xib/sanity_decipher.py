import math
from itertools import chain, product

import torch
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim import Adam
from tqdm import tqdm

from dev_misc.devlib.named_tensor import NoName, patch_named_tensors
from xib.gumbel import gumbel_softmax
from xib.ipa.process import B, I, O


def get_likelihoods(samples):
    samples = samples.align_to('batch', 'sample', 'length')[0]
    samples = samples.cpu().numpy().tolist()
    scores = list()
    unique = set()
    is_unique = list()
    for sample in samples:
        if sample == [B, B, I, I, I]:
            score = 0.5
        elif sample == [B, B, O, O, B]:
            score = 0.2
        elif sample == [B, O, O, O, O]:
            score = 0.1
        else:
            score = 0.2 / 240
        sample = tuple(sample)
        scores.append(score)
        is_unique.append(sample not in unique)
        unique.add(sample)
    return (torch.FloatTensor(scores).rename('sample').align_to('batch', 'sample') + 1e-8).log(), torch.BoolTensor(is_unique).rename('sample').align_to('batch', 'sample')


if __name__ == "__main__":
    patch_named_tensors()

    chars = 'ieste'
    id2char = ['i', 'e', 's', 't']
    char2id = {v: i for i, v in enumerate(id2char)}

    char_ids = [char2id[c] for c in chars]
    char_ids = torch.LongTensor([char_ids]).rename('batch', 'length')

    emb_layer = nn.Embedding(len(char2id), 10)
    emb_layer.refine_names('weight', ['vocab', 'dim'])
    label_predictor = nn.Linear(len(chars) * 10, len(chars) * 3)
    label_predictor.refine_names('weight', ['all_labels', 'dim'])

    params = list(chain(emb_layer.parameters(), label_predictor.parameters()))
    optimizer = Adam(params, lr=0.0002)
    num_samples = 100

    for step in tqdm(range(1, 10001)):
        emb_layer.train()
        label_predictor.train()
        optimizer.zero_grad()

        temperature = math.exp(math.log(10))  # - step * math.log(100) / 10000)
        emb = emb_layer(char_ids)
        emb = emb.flatten(['length', 'dim'], 'dim')
        logits = label_predictor(emb)
        logits = logits.unflatten('all_labels', [('length', len(chars)), ('label', 3)])
        log_probs = logits.log_softmax(dim='label')
        flat_log_probs = log_probs.flatten(['batch', 'length'], 'batch_X_length')
        with NoName(flat_log_probs):
            # samples = torch.multinomial(flat_log_probs.exp(), 100, replacement=True)
            # samples.rename_('batch_X_length', 'sample')
            # samples = samples.unflatten('batch_X_length', [('batch', 1), ('length', len(chars))])
            samples = torch.LongTensor(sum(list(product([B, I, O], repeat=5)), tuple())).view(-1, 5)
            samples = samples.rename('sample', 'length').align_to('batch', 'sample', 'length')
            sample_log_probs = log_probs.gather('label', samples).sum(dim='length')
        likelihoods, is_unique = get_likelihoods(samples)
        print(likelihoods.max())

        modified_sample_log_probs = (sample_log_probs + (~is_unique) * (-99.9)).log_softmax(dim='sample')
        loss = (modified_sample_log_probs.exp() * likelihoods)

        # loss = likelihoods * seq_probs
        loss = -loss.sum()  # / num_samples

        loss.backward()
        print(clip_grad_norm_(params, 5.0))
        optimizer.step()
        print(temperature)
        print(loss)
        print(logits)
