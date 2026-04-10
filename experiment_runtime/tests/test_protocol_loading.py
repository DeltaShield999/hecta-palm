from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from qwen_langgraph_demo.runtime.protocol import load_protocol_bundle  # noqa: E402
from qwen_langgraph_demo.runtime.sample_data import DEFAULT_TRANSACTION_CONTEXT  # noqa: E402


class ProtocolLoadingTests(unittest.TestCase):
    def test_default_protocol_bundle_loads(self) -> None:
        bundle = load_protocol_bundle()

        self.assertEqual(bundle.core.model_family, "Qwen2")
        self.assertEqual(bundle.core.primary_model, "Qwen/Qwen2-1.5B-Instruct")
        self.assertEqual(bundle.core.exposure_conditions, ("1x", "10x", "50x"))
        self.assertEqual(bundle.stage2.decoding.temperature, 0.0)
        self.assertEqual(bundle.stage3.dataset_defaults.allow_count, 1000)

    def test_render_intake_message_uses_frozen_template(self) -> None:
        bundle = load_protocol_bundle()
        rendered = bundle.render_intake_message(
            DEFAULT_TRANSACTION_CONTEXT,
            bundle.stage1.benign_request,
        )

        self.assertIn("Transaction intake summary:", rendered)
        self.assertIn("Account ID: ACC-10427", rendered)
        self.assertIn("Request: Review this transaction for fraud risk.", rendered)


if __name__ == "__main__":
    unittest.main()
