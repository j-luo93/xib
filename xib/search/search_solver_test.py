from xib.search.search_solver import SearchSolver

from dev_misc import TestCase, test_with_arguments


class TestSearchSolver(TestCase):

    def test_basic(self):
        test_with_arguments(min_word_length=2, _force=True)

        vocab = {'aba', 'cd', 'ab'}
        segment = 'abacd'
        solver = SearchSolver(vocab, 3)
        best_value, best_state = solver.find_best(segment)
        self.assertEqual(best_value, 3)
        self.assertEqual(len(best_state.spans), 2)
        span1, span2 = best_state.spans
        self.assertEqual(span1.start, 0)
        self.assertEqual(span1.end, 2)
        self.assertEqual(span2.start, 3)
        self.assertEqual(span2.end, 4)
