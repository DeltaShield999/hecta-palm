from pathlib import Path
import importlib.util
import sys
import tempfile
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from experiment.fhe.config import (  # noqa: E402
    DEFAULT_STAGE3_FHE_CONFIG_PATH,
    PINNED_FHE_BACKEND,
    PINNED_FHE_SCHEME,
    Stage3FheConfig,
)
from experiment.fhe.data import (  # noqa: E402
    compute_plaintext_logits,
    load_and_validate_embedding_artifact,
    load_plaintext_model_parameters,
    load_plaintext_prediction_artifact,
    sigmoid,
    validate_prediction_artifact,
)
from experiment.fhe.metrics import (  # noqa: E402
    LatencySample,
    classification_metrics_from_document,
    compute_prediction_comparison_metrics,
    summarize_latency_samples,
)
from experiment.fhe.openfhe_backend import (  # noqa: E402
    OpenFheBundlePaths,
    OpenFheCkksScorer,
)


class Stage3FheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = Stage3FheConfig.from_toml(DEFAULT_STAGE3_FHE_CONFIG_PATH)
        cls.model_parameters = load_plaintext_model_parameters(
            cls.config.plaintext_artifacts.model_parameters_path
        )
        cls.test_embeddings = load_and_validate_embedding_artifact(
            cls.config.plaintext_artifacts.test_embeddings_path,
            model_parameters=cls.model_parameters,
        )

    def test_config_loader_matches_frozen_stage3_fhe_contract(self) -> None:
        self.assertEqual(self.config.fhe.backend, PINNED_FHE_BACKEND)
        self.assertEqual(self.config.fhe.scheme, PINNED_FHE_SCHEME)
        self.assertEqual(self.config.fhe.security_level, "HEStd_128_classic")
        self.assertEqual(self.config.benchmark.split_name, "test")
        self.assertEqual(self.config.benchmark.example_count, 300)

    def test_plaintext_model_parameters_load_and_validate_saved_predictions(self) -> None:
        self.assertEqual(self.model_parameters.embedding_dimension, 384)
        self.assertEqual(len(self.model_parameters.weights), 384)
        self.assertAlmostEqual(self.model_parameters.threshold, 0.4199950085320943)

        predictions = load_plaintext_prediction_artifact(
            self.config.plaintext_artifacts.test_predictions_path
        )
        validate_prediction_artifact(
            predictions,
            embeddings=self.test_embeddings,
            model_parameters=self.model_parameters,
        )

        logits = compute_plaintext_logits(self.model_parameters, self.test_embeddings.embeddings)
        probabilities = sigmoid(logits)
        self.assertLessEqual(
            float(np.max(np.abs(probabilities - predictions.block_probabilities))),
            1e-12,
        )

    def test_prediction_and_latency_metric_aggregation(self) -> None:
        plaintext_metrics = classification_metrics_from_document(
            {
                "allow_precision": 1.0,
                "allow_recall": 0.5,
                "allow_f1": 2.0 / 3.0,
                "block_precision": 2.0 / 3.0,
                "block_recall": 1.0,
                "block_f1": 0.8,
                "macro_f1": (2.0 / 3.0 + 0.8) / 2.0,
                "accuracy": 0.75,
                "adversarial_block_rate": 1.0,
                "benign_false_positive_rate": 0.5,
                "true_allow_pred_allow": 1,
                "true_allow_pred_block": 1,
                "true_block_pred_allow": 0,
                "true_block_pred_block": 2,
            }
        )
        fhe_metrics = classification_metrics_from_document(
            {
                "allow_precision": 1.0,
                "allow_recall": 0.5,
                "allow_f1": 2.0 / 3.0,
                "block_precision": 2.0 / 3.0,
                "block_recall": 1.0,
                "block_f1": 0.8,
                "macro_f1": (2.0 / 3.0 + 0.8) / 2.0,
                "accuracy": 0.75,
                "adversarial_block_rate": 1.0,
                "benign_false_positive_rate": 0.5,
                "true_allow_pred_allow": 1,
                "true_allow_pred_block": 1,
                "true_block_pred_allow": 0,
                "true_block_pred_block": 2,
            }
        )
        comparison = compute_prediction_comparison_metrics(
            plaintext_probabilities=np.asarray([0.1, 0.9], dtype=np.float64),
            plaintext_predictions=np.asarray([0, 1], dtype=np.int8),
            plaintext_metrics=plaintext_metrics,
            fhe_probabilities=np.asarray([0.1002, 0.8997], dtype=np.float64),
            fhe_predictions=np.asarray([0, 1], dtype=np.int8),
            fhe_metrics=fhe_metrics,
        )
        self.assertAlmostEqual(comparison.prediction_match_rate, 1.0)
        self.assertAlmostEqual(comparison.plaintext_vs_fhe_accuracy_delta, 0.0)
        self.assertAlmostEqual(comparison.mean_abs_probability_delta, 0.00025)
        self.assertAlmostEqual(comparison.max_abs_probability_delta, 0.0003)

        latency_summary = summarize_latency_samples(
            (
                LatencySample("a", 1.0, 2.0, 3.0, 6.0),
                LatencySample("b", 2.0, 4.0, 6.0, 12.0),
                LatencySample("c", 3.0, 6.0, 9.0, 18.0),
            )
        )
        self.assertAlmostEqual(latency_summary["encryption_ms"]["mean"], 2.0)
        self.assertAlmostEqual(latency_summary["scoring_ms"]["p50"], 4.0)
        self.assertAlmostEqual(latency_summary["end_to_end_ms"]["p95"], 17.4)

    def test_openfhe_smoke_matches_plaintext_linear_score(self) -> None:
        if importlib.util.find_spec("openfhe") is None:
            self.skipTest("Optional openfhe dependency is not installed.")

        toy_model_parameters = type(self.model_parameters)(
            encoder_model_name=self.model_parameters.encoder_model_name,
            embedding_dimension=4,
            normalize_embeddings=True,
            classes=self.model_parameters.classes,
            class_mapping=self.model_parameters.class_mapping,
            weights=np.asarray([1.0, 0.5, -0.25, 2.0], dtype=np.float64),
            intercept=0.125,
            threshold=0.5,
            score_definition=self.model_parameters.score_definition,
            decision_rule=self.model_parameters.decision_rule,
            threshold_selection_split=self.model_parameters.threshold_selection_split,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle_paths = OpenFheBundlePaths.for_root(Path(temp_dir) / "compiled")
            scorer = OpenFheCkksScorer.load_or_create(
                settings=self.config.fhe,
                model_parameters=toy_model_parameters,
                bundle_paths=bundle_paths,
            )
            vector = np.asarray([0.1, -0.2, 0.3, 0.4], dtype=np.float64)
            expected_logit = float(vector @ np.asarray([1.0, 0.5, -0.25, 2.0]) + 0.125)
            decrypted_logit, _ = scorer.score_embedding(vector)
            self.assertAlmostEqual(decrypted_logit, expected_logit, places=6)
            self.assertFalse(scorer.reused_existing_bundle)
            self.assertTrue(bundle_paths.is_complete())

            reloaded_scorer = OpenFheCkksScorer.load_or_create(
                settings=self.config.fhe,
                model_parameters=toy_model_parameters,
                bundle_paths=bundle_paths,
            )
            reloaded_logit, _ = reloaded_scorer.score_embedding(vector)
            self.assertAlmostEqual(reloaded_logit, expected_logit, places=6)
            self.assertTrue(reloaded_scorer.reused_existing_bundle)


if __name__ == "__main__":
    unittest.main()
