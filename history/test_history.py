import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from history.build import import_source
from history.store import (
    HistoryIndex,
    connect,
    embed_pending,
    initialize,
    refresh_metadata,
)


class FakeModel:
    def encode(self, texts, **kw):
        out = np.zeros((len(texts), 384), dtype="float32")
        for i, t in enumerate(texts):
            out[i, 0 if "russian" in t.lower() else 1] = 1
        return out


class IndexTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.path = self.root / "session.jsonl"
        self.dbpath = self.root / "index.sqlite"
        self.db = connect(self.dbpath)
        initialize(self.db)

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def write(self, rows, mode="w"):
        with self.path.open(mode) as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    def ingest(self):
        with self.db:
            result = import_source(self.db, self.path, "codex")
            refresh_metadata(self.db)
        return result

    def message(self, text):
        return {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text}],
            },
        }

    def test_readable_reasoning_without_encrypted_payloads(self):
        encrypted = "opaque-ciphertext-do-not-render"
        self.write(
            [
                {"type": "session_meta", "payload": {"id": "reasoning"}},
                {
                    "type": "response_item",
                    "payload": {
                        "type": "reasoning",
                        "summary": [{"type": "summary_text", "text": "Readable summary"}],
                        "encrypted_content": encrypted,
                    },
                },
                {
                    "type": "response_item",
                    "payload": {"type": "reasoning", "encrypted_content": encrypted},
                },
            ]
        )
        self.ingest()
        texts = [r[0] for r in self.db.execute("SELECT text FROM chunks WHERE origin='thinking'")]
        self.assertEqual(texts, ["Readable summary"])
        self.assertFalse(
            self.db.execute(
                "SELECT 1 FROM chunks WHERE text LIKE ?", ("%" + encrypted + "%",)
            ).fetchone()
        )
        idx = HistoryIndex(self.dbpath)
        try:
            self.assertIn(encrypted, json.dumps(idx.records("codex:reasoning")))
        finally:
            idx.close()

    def test_append_partial_rewrite_and_fts(self):
        self.write(
            [
                {
                    "type": "session_meta",
                    "payload": {"id": "one", "cwd": "/example-project"},
                },
                self.message("teach Learner Russian"),
            ]
        )
        self.assertTrue(self.ingest())
        self.assertFalse(self.ingest())
        with patch("history.store.model", return_value=FakeModel()):
            self.assertGreater(embed_pending(self.db), 0)
            self.assertEqual(embed_pending(self.db), 0)
        self.write([self.message("ARASAAC pictures")], "a")
        self.assertTrue(self.ingest())
        idx = HistoryIndex(self.dbpath)
        self.assertEqual(len(idx.terms("Learner")), 1)
        self.assertEqual(len(idx.terms("ARASAAC")), 1)
        self.assertEqual(len(idx.terms("overall")), 0)
        with self.path.open("a") as f:
            f.write('{"type":')
        self.ingest()
        count = self.db.execute("select count(*) from records").fetchone()[0]
        self.ingest()
        self.assertEqual(count, self.db.execute("select count(*) from records").fetchone()[0])
        self.write(
            [
                {"type": "session_meta", "payload": {"id": "one"}},
                self.message("replacement"),
            ]
        )
        self.ingest()
        self.assertEqual(idx.terms("ARASAAC"), [])
        with patch("history.store.model", return_value=FakeModel()):
            embed_pending(self.db)
            self.assertTrue(idx.semantic("replacement", project="/example-project"))
        idx.close()

    def test_legacy_and_mirrors(self):
        self.write(
            [
                {"id": "old"},
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "legacy Russian"}],
                },
            ]
        )
        self.ingest()
        self.assertEqual(
            self.db.execute("select count(*) from chunks where origin='user'").fetchone()[0],
            1,
        )
        self.write(
            [
                {"type": "session_meta", "payload": {"id": "one"}},
                self.message("mirror"),
                {
                    "type": "event_msg",
                    "payload": {"type": "user_message", "message": "mirror"},
                },
            ]
        )
        self.ingest()
        self.assertEqual(
            self.db.execute(
                "select count(*) from searchable_chunks where origin='user'"
            ).fetchone()[0],
            1,
        )

    def test_failed_export_rolls_back_and_compressed_ingestion(self):
        import zstandard

        p = self.root / "conversations.json"
        p.write_text(json.dumps([{"uuid": "web", "name": "Original", "chat_messages": []}]))
        with self.db:
            import_source(self.db, p, "export")
        p.write_text('[{"uuid":"web","name":"Should roll back","chat_messages":[]}, BROKEN]')
        with self.assertRaises(Exception):
            with self.db:
                import_source(self.db, p, "export")
        self.assertEqual(
            self.db.execute("select title from threads where id='claude_web:web'").fetchone()[0],
            "Original",
        )
        self.write(
            [
                {"type": "session_meta", "payload": {"id": "zipped"}},
                self.message("compressed Russian"),
            ]
        )
        zp = self.root / "rollout.jsonl.zst"
        zp.write_bytes(zstandard.ZstdCompressor().compress(self.path.read_bytes()))
        with self.db:
            import_source(self.db, zp, "codex")
        self.assertEqual(
            self.db.execute(
                "select count(*) from chunks where thread_id='codex:zipped'"
            ).fetchone()[0],
            1,
        )

    def test_rebuild_indexes_without_reembedding(self):
        self.write(
            [
                {"type": "session_meta", "payload": {"id": "repair"}},
                self.message("repair Russian index"),
            ]
        )
        self.ingest()
        with patch("history.store.model", return_value=FakeModel()):
            embed_pending(self.db)
        self.db.execute("DROP TABLE chunks_fts")
        self.db.execute("DROP TABLE chunk_vectors")
        self.db.commit()
        initialize(self.db)
        with patch("history.store.model", side_effect=AssertionError("Should use cache")):
            self.assertGreater(embed_pending(self.db), 0)
        index = HistoryIndex(self.dbpath)
        self.assertTrue(index.terms("Russian"))
        index.close()

    def test_claude_subagent_and_custom_title(self):
        p = self.root / "agent-one.jsonl"
        p.write_text(
            "".join(
                json.dumps(r) + "\n"
                for r in [
                    {
                        "type": "user",
                        "sessionId": "parent",
                        "isSidechain": True,
                        "message": {"role": "user", "content": "real task"},
                    },
                    {"type": "custom-title", "customTitle": "Human title"},
                    {"type": "ai-title", "aiTitle": "Machine title"},
                ]
            )
        )
        with self.db:
            import_source(self.db, p, "claude_code")
        t = self.db.execute("select * from threads where id='claude_code:agent-one'").fetchone()
        self.assertEqual(t["title"], "Human title")
        self.assertEqual(t["parent_id"], "claude_code:parent")
        self.assertTrue(self.db.execute("select 1 from chunks where origin='user'").fetchone())

    def test_web_branch_attachment_and_incrementality(self):
        p = self.root / "conversations.json"
        c = {
            "id": "g",
            "title": "Russian",
            "current_node": "b",
            "mapping": {
                "a": {
                    "parent": None,
                    "message": {
                        "author": {"role": "assistant"},
                        "content": {"parts": ["oldbranch"]},
                    },
                },
                "b": {
                    "parent": None,
                    "message": {
                        "author": {"role": "assistant"},
                        "content": {"parts": ["activebranch"]},
                    },
                },
            },
        }
        p.write_text(json.dumps([c]))
        with self.db:
            import_source(self.db, p, "export")
            refresh_metadata(self.db)
        idx = HistoryIndex(self.dbpath)
        self.assertEqual(idx.terms("oldbranch"), [])
        self.assertTrue(idx.terms("activebranch"))
        with patch("history.store.model", return_value=FakeModel()):
            embed_pending(self.db)
        before = self.db.execute("select count(*) from embedding_cache").fetchone()[0]
        c["title"] = "Renamed"
        p.write_text(json.dumps([c]))
        with self.db:
            import_source(self.db, p, "export")
            refresh_metadata(self.db)
        with patch("history.store.model", return_value=FakeModel()):
            embed_pending(self.db)
        self.assertLessEqual(
            self.db.execute("select count(*) from embedding_cache").fetchone()[0] - before,
            1,
        )
        idx.close()


if __name__ == "__main__":
    unittest.main()
