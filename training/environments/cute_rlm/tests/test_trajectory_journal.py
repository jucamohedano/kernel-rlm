import json
import tempfile
import unittest
from pathlib import Path

from cute_rlm.trajectory_journal import TrajectoryJournal, read_progress


class TrajectoryJournalTests(unittest.TestCase):
    def test_persists_evidence_progress_and_quota_pause_across_reopen(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            journal_path, progress_path = root / "evidence.jsonl", root / "progress.json"
            journal = TrajectoryJournal(journal_path, progress_path)
            journal.call_intent(task_id="task-a", branch="main", call_id="c1", intent={"prompt": "x"})
            journal.raw_response(call_id="c1", raw_response={"text": "raw"})
            journal.tool_result(call_id="c1", tool_result={"stdout": "done"})
            journal.call_finished(call_id="c1", outcome="completed")
            journal.pause_for_quota(reason="provider quota", details={"retry_after": 60})

            reopened = TrajectoryJournal(journal_path, progress_path)
            self.assertEqual(reopened.progress()["last_sequence"], 5)
            self.assertTrue(reopened.progress()["quota_paused"])
            self.assertEqual(reopened.uncertain_calls(), [])
            snapshot = read_progress(progress_path)
            self.assertIn("not a live Python interpreter", snapshot["resume_contract"])
            rows = [json.loads(line) for line in journal_path.read_text().splitlines()]
            self.assertEqual(rows[1]["raw_response"], {"text": "raw"})
            self.assertEqual(rows[2]["tool_result"], {"stdout": "done"})

    def test_intent_without_response_is_uncertain_and_never_replayed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            journal = TrajectoryJournal(root / "evidence.jsonl", root / "progress.json")
            journal.call_intent(task_id="t", branch="b", call_id="uncertain", intent="request")
            journal.raw_response(call_id="uncertain", raw_response={"partial": True})
            reopened = TrajectoryJournal(root / "evidence.jsonl", root / "progress.json")
            self.assertEqual(reopened.uncertain_calls(), [{
                "call_id": "uncertain", "intent": "request",
                "has_raw_response": True, "has_tool_result": False,
            }])
            with self.assertRaisesRegex(ValueError, "already exists"):
                reopened.call_intent(task_id="t", branch="b", call_id="uncertain", intent="retry")

    def test_invalid_json_value_does_not_append_partial_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            journal = TrajectoryJournal(root / "evidence.jsonl", root / "progress.json")
            with self.assertRaises(ValueError):
                journal.append("evidence", value=float("nan"))
            self.assertFalse(journal.journal_path.exists())

    def test_reserved_fields_cannot_override_record_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            journal = TrajectoryJournal(root / "evidence.jsonl", root / "progress.json")
            for field in ("sequence", "recorded_at_utc_seconds", "event"):
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, "reserved journal field"):
                    journal.append("evidence", **{field: "forged"})
            self.assertEqual(journal.progress()["last_sequence"], 0)
            self.assertFalse(journal.journal_path.exists())

    def test_stale_progress_is_rebuilt_from_journal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            journal_path, progress_path = root / "evidence.jsonl", root / "progress.json"
            journal = TrajectoryJournal(journal_path, progress_path)
            journal.pause_for_quota(reason="rate limited")
            progress_path.write_text('{"stale": true}')
            reopened = TrajectoryJournal(journal_path, progress_path)
            self.assertTrue(read_progress(progress_path)["quota_paused"])
            self.assertEqual(reopened.progress()["last_sequence"], 1)

    def test_non_monotonic_journal_sequence_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "evidence.jsonl"
            path.write_text('{"sequence":2,"event":"one"}\n')
            with self.assertRaisesRegex(ValueError, "non-monotonic journal sequence"):
                TrajectoryJournal(path, root / "progress.json")

    def test_truncated_journal_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "evidence.jsonl"
            path.write_text('{"sequence":1}')
            with self.assertRaisesRegex(ValueError, "incomplete journal"):
                TrajectoryJournal(path, root / "progress.json")


if __name__ == "__main__":
    unittest.main()
