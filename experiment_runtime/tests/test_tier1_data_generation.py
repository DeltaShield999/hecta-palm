from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from experiment.data_gen.config import DEFAULT_CONFIG_PATH, Tier1DataConfig  # noqa: E402
from experiment.data_gen.materialize_tier1 import materialize_tier1_artifacts  # noqa: E402
from experiment.data_gen.tier1 import generate_tier1_dataset  # noqa: E402
from experiment.data_gen.validators import Tier1ValidationError, validate_tier1_dataset  # noqa: E402
from experiment.schemas.tier1 import CanaryRegistryEntry, NON_MEMBER_SPLIT  # noqa: E402


class Tier1DataGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = Tier1DataConfig.from_toml(DEFAULT_CONFIG_PATH)

    def test_generation_is_deterministic(self) -> None:
        first = generate_tier1_dataset(self.config)
        second = generate_tier1_dataset(self.config)

        self.assertEqual(
            [record.to_row() for record in first.records],
            [record.to_row() for record in second.records],
        )
        self.assertEqual(
            [entry.to_row() for entry in first.canary_registry],
            [entry.to_row() for entry in second.canary_registry],
        )

    def test_generated_dataset_satisfies_frozen_contract(self) -> None:
        dataset = generate_tier1_dataset(self.config)
        summary = validate_tier1_dataset(dataset.records, dataset.canary_registry, self.config)

        self.assertEqual(summary.total_records, self.config.total_records)
        self.assertEqual(summary.member_records, self.config.member_records)
        self.assertEqual(summary.non_member_records, self.config.non_member_records)
        self.assertEqual(summary.canary_count, self.config.canary_count)
        self.assertAlmostEqual(summary.fraud_rate, self.config.fraud_base_rate, places=6)

    def test_validator_rejects_registry_rows_for_non_members(self) -> None:
        dataset = generate_tier1_dataset(self.config)
        non_member = next(record for record in dataset.records if record.split == NON_MEMBER_SPLIT)

        invalid_registry = list(dataset.canary_registry)
        invalid_registry[0] = CanaryRegistryEntry(
            canary_id=invalid_registry[0].canary_id,
            record_id=non_member.record_id,
            customer_name=non_member.customer_name,
            date_of_birth=non_member.date_of_birth,
            account_number=non_member.account_number,
        )

        with self.assertRaisesRegex(Tier1ValidationError, "member record"):
            validate_tier1_dataset(dataset.records, invalid_registry, self.config)

    def test_materialization_round_trip_preserves_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = replace(self.config, output_dir=Path(temp_dir))
            result = materialize_tier1_artifacts(config)

            self.assertTrue(result.records_path.exists())
            self.assertTrue(result.registry_path.exists())
            self.assertEqual(result.validation.total_records, config.total_records)
            self.assertEqual(result.validation.canary_count, config.canary_count)


if __name__ == "__main__":
    unittest.main()
