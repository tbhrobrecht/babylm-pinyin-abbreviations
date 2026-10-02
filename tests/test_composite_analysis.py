import unittest

from results_analysis.build_composite_analysis import aggregate, average_ranks, quantile


class CompositeAnalysisTests(unittest.TestCase):
    def test_average_ranks_preserves_ties(self) -> None:
        self.assertEqual(average_ranks([1.0, 2.0, 2.0, 4.0]), [1.0, 2.5, 2.5, 4.0])
        self.assertEqual(average_ranks([1.0, 2.0, 2.0, 4.0], descending=True),
                         [4.0, 2.5, 2.5, 1.0])

    def test_quantile_uses_linear_interpolation(self) -> None:
        values = [0.0, 10.0, 20.0, 30.0]
        self.assertEqual(quantile(values, 0.0), 0.0)
        self.assertEqual(quantile(values, 0.5), 15.0)
        self.assertEqual(quantile(values, 1.0), 30.0)

    def test_equal_suite_prevents_large_suite_domination(self) -> None:
        values = {
            "large": [[0.0, 0.0, 0.0]],
            "small": [[100.0]],
        }
        self.assertEqual(
            aggregate(values, 0, equal_suite=True, included_suites=("large", "small")),
            50.0,
        )
        self.assertEqual(
            aggregate(values, 0, equal_suite=False, included_suites=("large", "small")),
            25.0,
        )


if __name__ == "__main__":
    unittest.main()
