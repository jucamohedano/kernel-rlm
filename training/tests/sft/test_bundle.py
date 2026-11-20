import hashlib
import json
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from rlm_train.sft.bundle import collate_group, load_bundle, masked_nll_sum, write_synthetic_fixture


class BundleTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory(prefix="xs21-sft-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.pin = write_synthetic_fixture(self.root)

    def test_exact_identity_and_padding(self):
        b = load_bundle(self.root, self.pin, allow_synthetic=True)
        self.assertEqual(len(b.groups), 2)
        batch = collate_group(
            b.groups[0],
            pad_id=b.pad_id,
            member_cap=b.member_cap,
            padded_slots_cap=b.padded_slots_cap,
        )
        self.assertEqual(batch["attention_mask"].tolist(), [[True] * 4, [True, True, True, False]])
        self.assertEqual(batch["labels"].tolist(), [[-100, -100, 3, 4], [-100] * 4])
        self.assertEqual(batch["input_ids"].tolist(), [[1, 2, 3, 4], [5, 0, 6, 0]])
        with self.assertRaisesRegex(ValueError, "entire trajectory"):
            collate_group(b.groups[0], pad_id=b.pad_id, member_cap=b.member_cap, padded_slots_cap=7)

    def test_pins_and_production_gate(self):
        with self.assertRaisesRegex(ValueError, "production blocked"):
            load_bundle(self.root, self.pin)
        with self.assertRaisesRegex(ValueError, "pin mismatch"):
            load_bundle(self.root, "0" * 64, allow_synthetic=True)
        p = self.root / "groups.jsonl"
        p.write_bytes(p.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "pin mismatch"):
            load_bundle(self.root, self.pin, allow_synthetic=True)
        p = self.root / "manifest.json"
        m = json.loads(p.read_bytes())
        m["scope"] = "production"
        p.write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError, "production blocked"):
            load_bundle(self.root, hashlib.sha256(p.read_bytes()).hexdigest(), allow_synthetic=True)

    def test_selected_label_causal_shift_and_token_mean(self):
        logits = torch.zeros((2, 4, 8), requires_grad=True)
        with torch.no_grad():
            logits[0, 1, 3] = 2.0
            logits[0, 2, 4] = 1.5
        labels = torch.tensor([[-100, -100, 3, 4], [-100, -100, -100, -100]])
        loss, n = masked_nll_sum(logits, labels)
        expected = sum(math.log(7 + math.exp(x)) - x for x in (2.0, 1.5))
        self.assertEqual(n.item(), 2)
        self.assertAlmostEqual(loss.item(), expected, places=6)
        (loss / n).backward()
        self.assertEqual(
            logits.grad.abs().sum(dim=-1).ne(0).tolist(), [[False, True, True, False], [False] * 4]
        )

    def test_reject_first_position_label(self):
        with self.assertRaisesRegex(ValueError, "position zero"):
            masked_nll_sum(torch.zeros((1, 2, 8)), torch.tensor([[3, -100]]))


if __name__ == "__main__":
    unittest.main()
