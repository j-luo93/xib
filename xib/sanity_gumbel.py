import math
from itertools import chain

import torch
import torch.nn as nn
from torch.optim import Adam

from xib.gumbel import gumbel_softmax

if __name__ == "__main__":

    x = torch.FloatTensor([[0.0], [1.0], [2.0]])
    enc_layer = nn.Sequential(
        nn.Linear(1, 10),
        nn.LeakyReLU(0.1),
        nn.Linear(10, 2))
    dec_layer = nn.Linear(2, 2)

    optimizer = Adam(chain(enc_layer.parameters(), dec_layer.parameters()), lr=0.002)

    from tqdm import tqdm
    for step in tqdm(range(1, 1001)):
        enc_layer.train()
        optimizer.zero_grad()

        out = enc_layer(x).rename('batch', 'label')
        # print(10 / step, out)
        label_probs, label_probs_hard, samples = gumbel_softmax(out, 1.0, 100)  # 10 / step)

        # _samples = samples.detach().cpu().numpy().reshape(100, -1).tolist()
        _samples = samples.detach().cpu().numpy().reshape(1, -1).tolist()
        # assert len(_samples) == 3
        all_probs = list()
        for _s in _samples:
            all_probs.append(
                [0.01, 0.1, 0.01, 0.01, 0.01, 0.2, 0.65, 0.01]
            )
            # if _s == [0, 0, 1]:
            #     probs = 0.1
            # elif _s == [1, 1, 0]:
            #     probs = 0.65
            # elif _s == [1, 0, 1]:
            #     probs = 0.2
            # else:
            #     probs = 0.01
            # probs = math.log(probs)
            # all_probs.append(probs)

        # li = torch.LongTensor([0, 1, 2]).view(3, -1)
        # si = torch.LongTensor(list(range(1))).view(-1, 1)
        # y = (label_probs_hard.rename(None)[li, samples.rename(None), si] + 1e-8).log()
        from itertools import product
        inds = sum(list(product([0, 1], repeat=3)), tuple())
        li = [0, 1, 2] * 8
        label_probs_hard.rename_(None)
        probs = label_probs_hard[li, inds].view(8, 3, -1)
        probs = (1e-8 + probs).log().sum(dim=1).exp()
        # y = (1e-8 + label_probs_hard[0, samples[0]]).log() + (1e-8 + label_probs_hard[1, samples[1]]).log()
        breakpoint()  # DEBUG(j_luo)
        loss = (probs * (1e-8 + torch.FloatTensor(all_probs).view(8, -1)).log())
        # loss = (loss + 1e-8).log()
        # label_probs.rename_(None)
        # label_probs_hard.rename_(None)

        # tmp = torch.stack([label_probs_hard[0, 1], label_probs_hard[1, 1], label_probs_hard[2, 0]], dim=-1)
        # tmp = (1e-8 + tmp).log()
        # loss = tmp.sum(dim=-1)

        # tmp = torch.stack([label_probs[0, 1], label_probs[1, 1], label_probs[2, 0]], dim=-1)
        # tmp = (1e-8 + tmp).log()
        # loss = tmp.sum(dim=-1)

        # logits = dec_layer(label_probs)
        # log_probs = logits.log_softmax(dim=-1)
        # loss = log_probs[0, 0] + log_probs[1, 1]

        kl = label_probs * ((1e-8 + label_probs).log() - math.log(1.0 / 2))
        kl = kl.sum()
        la = 0.001 * step * .0
        elbo = loss.mean() - la * kl

        (-elbo).backward()

        print('-' * 30)
        print('z:')
        print(out)
        # print('label_probs:')
        # print(label_probs)
        # print('samples:')
        # print(samples)
        # print('loss:')
        # print(loss)
        # print('kl:')
        # print(kl)
        import time; time.sleep(0.05)

        optimizer.step()
