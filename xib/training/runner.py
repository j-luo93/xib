import torch

from dev_misc import g, get_tensor
from dev_misc.devlib import IT, get_length_mask
from dev_misc.devlib.helper import get_tensor
from dev_misc.devlib.named_tensor import NoName
from dev_misc.trainlib import Metric, Metrics
from xib.batch import mask_out_target_weight
from xib.data_loader import ContinuousTextIpaBatch
from xib.extract_words_impl import extract_words_v8 as extract_words  # pylint: disable=no-name-in-module
from xib.ipa import should_include
from xib.model.lm_model import LM


class BaseLMRunner:

    def analyze_scores(self, scores) -> Metrics:
        metrics = Metrics()
        total_loss = 0.0
        total_weight = 0.0
        for name, (losses, weights) in scores.items():
            if should_include(g.feat_groups, name):
                loss = (losses * weights).sum()
                weight = weights.sum()
                total_loss += loss
                total_weight += weight
                loss = Metric(f'loss_{name.snake}', loss, weight)
                metrics += loss
        metrics += Metric('loss', total_loss, total_weight)
        return metrics

    @property
    def track(self):
        return self.tracker.step  # pylint: disable=no-member


class BaseDecipherRunner:

    def get_metrics(self, batch: ContinuousTextIpaBatch) -> Metrics:
        ret = self.model(batch, self.mode)  # pylint: disable=no-member
        metrics = Metrics()
        with NoName(batch.lengths):
            length_mask = get_length_mask(batch.lengths, batch.lengths.max()).refine_names('batch', 'length')
        weight = length_mask.sum()
        if 'supervised' in g.mode:
            local_target_log_probs = ret['label_log_probs'].gather('label', batch.gold_tag_seqs)
            local_losses = (length_mask.float().align_as(local_target_log_probs) * local_target_log_probs)
            local_loss = Metric('local_loss', -local_losses.sum(), weight)
            metrics += local_loss
        total_loss = 0.0
        if self.mode == 'local':  # pylint: disable=no-member
            # total_loss = Metric('total_loss', local_loss.total, weight)
            total_loss += local_loss.total
        elif self.mode == 'global':  # pylint: disable=no-member
            # # DEBUG(j_luo)
            # modified_log_probs = ret['sample_log_probs'] * g.concentration + (~ret['is_unique']).float() * (-999.9)
            # sample_probs = modified_log_probs.log_softmax(dim='sample').exp()
            # score = (sample_probs * ret['seq_scores']).sum()
            # total_loss = Metric('total_loss', -score, batch.batch_size)
            # metrics += total_loss

            global_target_log_probs = ret['seq_log_probs'].align_to('batch', 'sample')[:, 0]
            # FIXME(j_luo) This should be divided by batch size, not weight.
            global_loss = Metric('global_loss', -global_target_log_probs.sum(), weight)
            metrics += global_loss
            # total_loss = Metric('total_loss', local_loss.total + global_loss.total, weight)
            total_loss += global_loss.total
        total_loss = Metric('total_loss', total_loss, weight)
        metrics += total_loss
        return metrics
