from dev_misc import g
from dev_misc.devlib import IT, get_length_mask
from dev_misc.trainlib import Metric, Metrics
from xib.data_loader import ContinuousTextIpaBatch
from xib.extract_words_impl import extract_words_v8 as extract_words  # pylint: disable=no-name-in-module
from xib.ipa import should_include


class BaseLMRunner:

    def analyze_scores(self, scores) -> Metrics:
        metrics = Metrics()
        total_loss = 0.0
        total_weight = 0.0
        for name, (losses, weights) in scores.items():
            if should_include(self.feat_groups, name):
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
        return self.tracker.step


class BaseDecipherRunner:

    def get_metrics(self, batch: ContinuousTextIpaBatch) -> Metrics:
        ret = self.model(batch)
        bs = batch.batch_size
        metrics = Metrics()
        if 'supervised' in g.mode:
            local_target_log_probs = ret['label_log_probs'].gather('label', batch.gold_tag_seqs)
            length_mask = get_length_mask(batch.lengths, batch.lengths.max()).refine_names('batch', 'length')
            local_losses = (length_mask.float().align_as(local_target_log_probs) * local_target_log_probs)
            local_loss = Metric('local_loss', -local_losses.sum(), bs)
            metrics += local_loss
        if g.mode == 'local-supervised':
            total_loss = Metric('total_loss', local_loss.total, bs)
        elif g.mode == 'global-supervised':
            modified_seq_log_probs = ret['seq_scores'] + (~ret['is_unique']).float() * (-999.9)
            seq_log_probs = modified_seq_log_probs.log_softmax(dim='sample')
            global_target_log_probs = seq_log_probs.align_to('batch', 'sample', 'seq_feat')[:, 0]
            global_loss = Metric('global_loss', -global_target_log_probs.sum(), bs)
            metrics += global_loss
            total_loss = Metric('total_loss', local_loss.total + global_loss.total, bs)
        metrics += total_loss
        return metrics
