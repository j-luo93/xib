from dev_misc import TestCase, test_with_arguments
from xib.ipa.process import Segment
from xib.search.search_solver import SearchSolver


class TestSearchSolver(TestCase):

    def test_basic(self):
        test_with_arguments(min_word_length=2, _force=True)

        vocab = {'aba', 'cd', 'ab'}
        segment = Segment('abacd')
        solver = SearchSolver(vocab, 3)
        best_value, best_state = solver.find_best(segment)
        self.assertEqual(best_value, 5)
        self.assertEqual(len(best_state.spans), 2)
        span1, span2 = best_state.spans
        if span1.start > span2.start:
            span1, span2 = span2, span1
        self.assertEqual(span1.start, 0)
        self.assertEqual(span1.end, 2)
        self.assertEqual(span2.start, 3)
        self.assertEqual(span2.end, 4)
