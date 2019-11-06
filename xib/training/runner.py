from trainlib import Metric, Metrics
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
