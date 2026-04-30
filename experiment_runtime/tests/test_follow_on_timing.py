from math import sqrt
from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from experiment.follow_on.timing import (  # noqa: E402
    FILTER_TIMING_COLUMNS,
    PIPELINE_TIMING_COLUMNS,
    FilterTimingSample,
    PipelineTimingSample,
    SetupTimingEntry,
    summarize_filter_timing_samples,
    summarize_numeric_values,
    summarize_pipeline_timing_samples,
    summarize_timing_rows,
)


class FollowOnTimingTests(unittest.TestCase):
    def test_timing_row_helpers_match_contract_columns(self) -> None:
        filter_sample = FilterTimingSample(
            row_id="adaptive_001",
            eval_dataset="adaptive",
            filter_mode="fhe_filter",
            embedding_ms=1.0,
            encryption_ms=2.0,
            fhe_scoring_ms=3.0,
            decryption_ms=4.0,
            threshold_ms=0.5,
            io_ms=None,
            total_filter_ms=10.5,
        )
        pipeline_sample = PipelineTimingSample(
            row_id="adaptive_001",
            exposure_condition="1x",
            eval_dataset="adaptive",
            condition="fhe_filter",
            filter_mode="fhe_filter",
            source_type="adaptive_adversarial",
            filter_decision="ALLOW",
            response_generated=1,
            filter_total_ms=10.5,
            llm_generation_ms=120.0,
            routing_ms=1.5,
            total_pipeline_ms=132.0,
        )
        setup_entry = SetupTimingEntry(
            component="sentence_encoder_load",
            duration_ms=25.0,
            detail="cpu",
        )

        self.assertEqual(tuple(filter_sample.to_row().keys()), FILTER_TIMING_COLUMNS)
        self.assertEqual(tuple(pipeline_sample.to_row().keys()), PIPELINE_TIMING_COLUMNS)
        self.assertEqual(setup_entry.to_row()["component"], "sentence_encoder_load")
        self.assertEqual(setup_entry.to_row()["duration_ms"], 25.0)

    def test_timing_summary_percentiles_and_population_std(self) -> None:
        summary = summarize_numeric_values((10.0, 20.0, 40.0))

        self.assertEqual(summary["count"], 3)
        self.assertAlmostEqual(summary["mean"], 70.0 / 3.0)
        self.assertAlmostEqual(summary["p50"], 20.0)
        self.assertAlmostEqual(summary["p90"], 36.0)
        self.assertAlmostEqual(summary["p95"], 38.0)
        self.assertAlmostEqual(summary["p99"], 39.6)
        self.assertAlmostEqual(summary["min"], 10.0)
        self.assertAlmostEqual(summary["max"], 40.0)
        self.assertAlmostEqual(
            summary["std"],
            sqrt((((10.0 - 70.0 / 3.0) ** 2) + ((20.0 - 70.0 / 3.0) ** 2) + ((40.0 - 70.0 / 3.0) ** 2)) / 3.0),
        )

    def test_timing_summary_ignores_none_and_empty_values(self) -> None:
        rows = (
            {"embedding_ms": None, "total_filter_ms": ""},
            {"embedding_ms": 2.0, "total_filter_ms": 8.0},
            {"embedding_ms": 4.0, "total_filter_ms": None},
        )

        summary = summarize_timing_rows(
            rows,
            numeric_columns=("embedding_ms", "total_filter_ms", "fhe_scoring_ms"),
        )

        self.assertEqual(summary["embedding_ms"]["count"], 2)
        self.assertAlmostEqual(summary["embedding_ms"]["mean"], 3.0)
        self.assertEqual(summary["total_filter_ms"]["count"], 1)
        self.assertAlmostEqual(summary["total_filter_ms"]["p95"], 8.0)
        self.assertEqual(summary["fhe_scoring_ms"]["count"], 0)
        self.assertIsNone(summary["fhe_scoring_ms"]["mean"])
        self.assertIsNone(summary["fhe_scoring_ms"]["std"])

    def test_filter_and_pipeline_summary_helpers_use_duration_columns(self) -> None:
        filter_summary = summarize_filter_timing_samples(
            (
                FilterTimingSample(
                    row_id="row-1",
                    eval_dataset="mixed_traffic",
                    filter_mode="plaintext_filter",
                    embedding_ms=1.0,
                    threshold_ms=0.2,
                    total_filter_ms=1.2,
                ),
                FilterTimingSample(
                    row_id="row-2",
                    eval_dataset="mixed_traffic",
                    filter_mode="plaintext_filter",
                    embedding_ms=3.0,
                    threshold_ms=0.4,
                    total_filter_ms=3.4,
                ),
            )
        )
        pipeline_summary = summarize_pipeline_timing_samples(
            (
                PipelineTimingSample(
                    row_id="row-1",
                    exposure_condition="1x",
                    eval_dataset="mixed_traffic",
                    condition="plaintext_filter",
                    filter_mode="plaintext_filter",
                    source_type="benign",
                    filter_decision="BLOCK",
                    response_generated=0,
                    filter_total_ms=1.2,
                    llm_generation_ms="",
                    routing_ms=0.3,
                    total_pipeline_ms=1.5,
                ),
                PipelineTimingSample(
                    row_id="row-2",
                    exposure_condition="1x",
                    eval_dataset="mixed_traffic",
                    condition="plaintext_filter",
                    filter_mode="plaintext_filter",
                    source_type="benign",
                    filter_decision="ALLOW",
                    response_generated=1,
                    filter_total_ms=3.4,
                    llm_generation_ms=100.0,
                    routing_ms=0.6,
                    total_pipeline_ms=104.0,
                ),
            )
        )

        self.assertAlmostEqual(filter_summary["embedding_ms"]["mean"], 2.0)
        self.assertEqual(filter_summary["encryption_ms"]["count"], 0)
        self.assertEqual(pipeline_summary["llm_generation_ms"]["count"], 1)
        self.assertAlmostEqual(pipeline_summary["total_pipeline_ms"]["max"], 104.0)


if __name__ == "__main__":
    unittest.main()
