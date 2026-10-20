"""Regression tests for metric helpers, without importing the training stack."""

import ast
from pathlib import Path
import unittest

import numpy as np


def load_metric_helpers():
    # The training entry point imports model/CUDA/data dependencies. Compile only
    # its actual metric definitions so these unit tests require NumPy alone.
    source = Path(__file__).resolve().parents[1] / "main_multimodal_seg_single.py"
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    names = {"compute_iou", "compute_metrics", "compute_accuracy", "compute_miou"}
    definitions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if {node.name for node in definitions} != names:
        raise AssertionError("The training entry point is missing a metric helper")
    namespace = {"np": np}
    module = ast.Module(body=definitions, type_ignores=[])
    exec(compile(module, str(source), "exec"), namespace)
    return namespace


METRICS = load_metric_helpers()


class TensorLike:
    """Provide only the tensor operations consumed by the metric helpers."""

    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.int64)

    def cpu(self):
        return self

    def numpy(self):
        return self.values


def metrics(pred, target, num_classes=2):
    return METRICS["compute_metrics"](
        TensorLike(pred), TensorLike(target), num_classes
    )


class EvaluationMetricsTests(unittest.TestCase):
    def test_perfect_prediction(self):
        precision, recall, f1 = metrics([0, 1], [0, 1])
        np.testing.assert_allclose(precision, [1.0, 1.0])
        np.testing.assert_allclose(recall, [1.0, 1.0])
        np.testing.assert_allclose(f1, [1.0, 1.0])

    def test_partial_overlap(self):
        precision, recall, f1 = metrics([1, 1, 0, 0], [1, 0, 1, 0])
        np.testing.assert_allclose(precision, [0.5, 0.5])
        np.testing.assert_allclose(recall, [0.5, 0.5])
        np.testing.assert_allclose(f1, [0.5, 0.5])

    def test_complete_miss_has_zero_f1(self):
        precision, recall, f1 = metrics([0, 0], [1, 1])
        self.assertTrue(np.isnan(precision[1]))
        self.assertEqual(recall[1], 0.0)
        self.assertEqual(f1[1], 0.0)

    def test_false_positives_without_reference_have_zero_f1(self):
        precision, recall, f1 = metrics([1, 1], [0, 0])
        self.assertEqual(precision[1], 0.0)
        self.assertTrue(np.isnan(recall[1]))
        self.assertEqual(f1[1], 0.0)

    def test_disjoint_prediction_has_zero_f1(self):
        _, _, f1 = metrics([1, 0], [0, 1])
        np.testing.assert_allclose(f1, [0.0, 0.0])

    def test_class_absent_from_prediction_and_reference_is_undefined(self):
        precision, recall, f1 = metrics([0, 0], [0, 0])
        self.assertTrue(np.isnan(precision[1]))
        self.assertTrue(np.isnan(recall[1]))
        self.assertTrue(np.isnan(f1[1]))
        self.assertEqual(f1[0], 1.0)

    def test_imagewise_average_includes_zero_and_excludes_undefined(self):
        per_image_f1 = np.stack([
            metrics([1, 0], [1, 0])[2],
            metrics([0, 0], [1, 0])[2],
            metrics([0, 0], [0, 0])[2],
        ])
        # Foreground F1 values are [1, 0, NaN], not [1, NaN, NaN].
        self.assertEqual(np.nanmean(per_image_f1, axis=0)[1], 0.5)

    def test_multiclass_zero_and_absent_class(self):
        _, _, f1 = metrics([0, 1, 1, 2, 2], [0, 1, 2, 2, 3], 5)
        np.testing.assert_allclose(f1[:4], [1.0, 2.0 / 3.0, 0.5, 0.0])
        self.assertTrue(np.isnan(f1[4]))

    def test_f1_matches_per_image_iou_identity(self):
        rng = np.random.RandomState(0)
        for _ in range(25):
            pred = TensorLike(rng.randint(0, 5, size=(3, 4)))
            target = TensorLike(rng.randint(0, 5, size=(3, 4)))
            iou = METRICS["compute_iou"](pred, target, 5)
            _, _, f1 = METRICS["compute_metrics"](pred, target, 5)
            np.testing.assert_allclose(f1, 2 * iou / (1 + iou), equal_nan=True)

    def test_oa_and_imagewise_miou_are_unchanged(self):
        pred = TensorLike([1, 1, 0, 0])
        target = TensorLike([1, 0, 1, 0])
        self.assertEqual(METRICS["compute_accuracy"](pred, target), 0.5)
        self.assertAlmostEqual(METRICS["compute_miou"](pred, target, 2), 1.0 / 3.0)
        absent = TensorLike([0, 0])
        self.assertEqual(METRICS["compute_miou"](absent, absent, 2), 1.0)


if __name__ == "__main__":
    unittest.main()
