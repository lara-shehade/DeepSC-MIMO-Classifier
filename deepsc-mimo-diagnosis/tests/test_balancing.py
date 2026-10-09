from collections import Counter
import unittest

from analysis.classical_diagnostics import CLASSES, balanced_indices
from analysis.neural_diagnostics import balanced_records


def unbalanced_rows():
    rows = []
    for label in CLASSES:
        severities = ("",)
        repetitions = 4
        if label == "imperfect_csi":
            severities, repetitions = ("0.99", "0.95", "0.90"), 4
        elif label == "spatial_correlation":
            severities, repetitions = ("0.3", "0.6", "0.9"), 4
        for severity in severities:
            for _ in range(repetitions):
                rows.append({"label": label, "severity": severity})
    return rows


class BalancingTests(unittest.TestCase):
    def check_output(self, rows):
        labels = Counter(row["label"] for row in rows)
        self.assertEqual(set(labels.values()), {4})
        for label in ("imperfect_csi", "spatial_correlation"):
            severity = Counter(
                row["severity"] for row in rows if row["label"] == label
            )
            self.assertLessEqual(max(severity.values()) - min(severity.values()), 1)

    def test_classical_fold_balancing(self):
        rows = unbalanced_rows()
        indices = balanced_indices(rows, 11)
        self.check_output([rows[index] for index in indices])

    def test_neural_fold_balancing(self):
        self.check_output(balanced_records(unbalanced_rows(), 11))


if __name__ == "__main__":
    unittest.main()
