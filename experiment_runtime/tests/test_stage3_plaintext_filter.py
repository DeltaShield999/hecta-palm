from collections import Counter
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from experiment.filter_train.config import (  # noqa: E402
    DEFAULT_STAGE3_PLAINTEXT_FILTER_CONFIG_PATH,
    PINNED_LOGISTIC_REGRESSION_C_VALUES,
    PINNED_STAGE3_ENCODER_MODEL_NAME,
    Stage3PlaintextFilterConfig,
)
from experiment.filter_train.data import INT_TO_LABEL, load_stage3_datasets  # noqa: E402
from experiment.filter_train.embeddings import (  # noqa: E402
    EmbeddedStage3Split,
    load_embedding_artifact,
    write_embedding_artifact,
)
from experiment.filter_train.metrics import (  # noqa: E402
    ClassificationMetrics,
    ThresholdSweepResult,
    compare_selection_results,
    compute_classification_metrics,
    select_best_threshold,
)
from experiment.schemas.stage3 import (  # noqa: E402
    STAGE3_ALLOW_LABEL,
    STAGE3_BLOCK_LABEL,
    STAGE3_ROWS_BY_SPLIT,
    STAGE3_ROWS_PER_LABEL_BY_SPLIT,
)


class Stage3PlaintextFilterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = Stage3PlaintextFilterConfig.from_toml(DEFAULT_STAGE3_PLAINTEXT_FILTER_CONFIG_PATH)
        cls.datasets = load_stage3_datasets(
            train_path=cls.config.datasets.train,
            val_path=cls.config.datasets.val,
            test_path=cls.config.datasets.test,
        )

    def test_config_loader_matches_frozen_stage3_contract(self) -> None:
        self.assertEqual(self.config.encoder.model_name, PINNED_STAGE3_ENCODER_MODEL_NAME)
        self.assertTrue(self.config.encoder.normalize_embeddings)
        self.assertEqual(self.config.encoder.device, "cpu")
        self.assertEqual(
            self.config.logistic_regression.candidate_c_values,
            PINNED_LOGISTIC_REGRESSION_C_VALUES,
        )
        self.assertEqual(self.config.threshold_selection.selection_split, "val")
        self.assertEqual(self.config.threshold_selection.score_label, "BLOCK")

    def test_dataset_loader_preserves_split_counts_and_order(self) -> None:
        for split_name, dataset in self.datasets.items():
            self.assertEqual(len(dataset.rows), STAGE3_ROWS_BY_SPLIT[split_name])
            self.assertEqual(dataset.rows[0].message_id, f"stage3_allow_transaction_scoring_c{'01' if split_name == 'train' else '15' if split_name == 'val' else '18'}_v01")
            label_counts = Counter(dataset.label_names)
            self.assertEqual(
                label_counts,
                Counter(
                    {
                        STAGE3_ALLOW_LABEL: STAGE3_ROWS_PER_LABEL_BY_SPLIT[split_name],
                        STAGE3_BLOCK_LABEL: STAGE3_ROWS_PER_LABEL_BY_SPLIT[split_name],
                    }
                ),
            )
            self.assertEqual(INT_TO_LABEL[int(dataset.labels[0])], dataset.rows[0].label)

    def test_embedding_artifact_round_trip_preserves_arrays(self) -> None:
        sample_dataset = self.datasets["val"]
        sample_embeddings = np.zeros((len(sample_dataset.rows), 384), dtype=np.float32)
        sample_embeddings[:, 0] = 1.0
        embedded_split = EmbeddedStage3Split(
            dataset=sample_dataset,
            embeddings=sample_embeddings,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_path = Path(temp_dir) / "val_embeddings.npz"
            write_embedding_artifact(
                artifact_path,
                embedded_split,
                encoder_model_name=PINNED_STAGE3_ENCODER_MODEL_NAME,
                normalize_embeddings=True,
            )
            artifact = load_embedding_artifact(artifact_path)

        self.assertEqual(artifact.split_name, "val")
        self.assertEqual(artifact.embeddings.shape, (300, 384))
        self.assertTrue(np.array_equal(artifact.labels, sample_dataset.labels))
        self.assertEqual(artifact.message_ids[0], sample_dataset.rows[0].message_id)
        self.assertEqual(artifact.encoder_model_name, PINNED_STAGE3_ENCODER_MODEL_NAME)
        self.assertTrue(artifact.normalize_embeddings)

    def test_threshold_selection_uses_frozen_tie_breakers(self) -> None:
        candidate_lower_threshold = ThresholdSweepResult(
            threshold=0.40,
            metrics=ClassificationMetrics(
                allow_precision=0.8,
                allow_recall=0.8,
                allow_f1=0.8,
                block_precision=0.8,
                block_recall=0.8,
                block_f1=0.8,
                macro_f1=0.8,
                accuracy=0.8,
                adversarial_block_rate=0.8,
                benign_false_positive_rate=0.2,
                true_allow_pred_allow=8,
                true_allow_pred_block=2,
                true_block_pred_allow=2,
                true_block_pred_block=8,
            ),
        )
        candidate_higher_threshold = ThresholdSweepResult(
            threshold=0.60,
            metrics=ClassificationMetrics(
                allow_precision=0.8,
                allow_recall=0.8,
                allow_f1=0.8,
                block_precision=0.8,
                block_recall=0.8,
                block_f1=0.8,
                macro_f1=0.8,
                accuracy=0.8,
                adversarial_block_rate=0.8,
                benign_false_positive_rate=0.2,
                true_allow_pred_allow=8,
                true_allow_pred_block=2,
                true_block_pred_allow=2,
                true_block_pred_block=8,
            ),
        )
        better_recall = ThresholdSweepResult(
            threshold=0.55,
            metrics=ClassificationMetrics(
                allow_precision=0.7,
                allow_recall=0.7,
                allow_f1=0.7,
                block_precision=0.9,
                block_recall=0.85,
                block_f1=0.875,
                macro_f1=0.7875,
                accuracy=0.775,
                adversarial_block_rate=0.85,
                benign_false_positive_rate=0.3,
                true_allow_pred_allow=7,
                true_allow_pred_block=3,
                true_block_pred_allow=1,
                true_block_pred_block=9,
            ),
        )

        best = select_best_threshold(
            (candidate_higher_threshold, candidate_lower_threshold)
        )
        self.assertEqual(best.threshold, 0.40)
        self.assertEqual(compare_selection_results(better_recall, candidate_lower_threshold), -1)

    def test_metrics_aggregation_matches_security_definitions(self) -> None:
        true_labels = np.asarray([0, 0, 1, 1], dtype=np.int8)
        predicted_labels = np.asarray([0, 1, 1, 1], dtype=np.int8)

        metrics = compute_classification_metrics(true_labels, predicted_labels)

        self.assertAlmostEqual(metrics.allow_precision, 1.0)
        self.assertAlmostEqual(metrics.allow_recall, 0.5)
        self.assertAlmostEqual(metrics.allow_f1, 2.0 / 3.0)
        self.assertAlmostEqual(metrics.block_precision, 2.0 / 3.0)
        self.assertAlmostEqual(metrics.block_recall, 1.0)
        self.assertAlmostEqual(metrics.block_f1, 0.8)
        self.assertAlmostEqual(metrics.macro_f1, (2.0 / 3.0 + 0.8) / 2.0)
        self.assertAlmostEqual(metrics.adversarial_block_rate, 1.0)
        self.assertAlmostEqual(metrics.benign_false_positive_rate, 0.5)
        self.assertEqual(
            metrics.confusion_matrix,
            {
                "true_ALLOW_pred_ALLOW": 1,
                "true_ALLOW_pred_BLOCK": 1,
                "true_BLOCK_pred_ALLOW": 0,
                "true_BLOCK_pred_BLOCK": 2,
            },
        )


if __name__ == "__main__":
    unittest.main()
