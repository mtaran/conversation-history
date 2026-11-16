import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from history.build import discover, import_source
from history.chatgpt import graph, normalize
from history.store import HistoryIndex, connect, initialize


class ChatGPTTests(unittest.TestCase):
    def test_reasoning_graph_assets_and_archive(self):
        with tempfile.TemporaryDirectory() as d:
            r = Path(d)
            p = r / "conversations-000.json"
            (r / "file-picture.dat").write_bytes(b"\x89PNG\r\n\x1a\n")
            (r / "library_files.json").write_text(
                json.dumps(
                    [
                        {
                            "file_id": {"opaque": "value"},
                            "id": "other",
                            "file_name": "other.txt",
                        }
                    ]
                )
            )
            (r / "conversation_asset_file_names.json").write_text(
                json.dumps({"file-picture": "family.png"})
            )

            def node(parent, role, value):
                return {
                    "parent": parent,
                    "message": {
                        "author": {"role": role},
                        "content": value,
                        "metadata": {},
                    },
                }

            c = {
                "id": "g",
                "title": "Russian",
                "create_time": 1700000000,
                "is_archived": True,
                "current_node": "answer",
                "mapping": {
                    "answer": node(
                        "thought",
                        "assistant",
                        {"content_type": "text", "parts": ["finalanswer"]},
                    ),
                    "old": node(
                        "u",
                        "assistant",
                        {"content_type": "text", "parts": ["oldbranch"]},
                    ),
                    "thought": node(
                        "u",
                        "assistant",
                        {
                            "content_type": "thoughts",
                            "thoughts": [{"content": "THINK_SENTINEL", "summary": "summary"}],
                        },
                    ),
                    "u": node(
                        None,
                        "user",
                        {
                            "content_type": "multimodal_text",
                            "parts": [
                                {
                                    "content_type": "image_asset_pointer",
                                    "asset_pointer": "sediment://file-picture",
                                },
                                "QUESTION_SENTINEL",
                            ],
                        },
                    ),
                },
            }
            p.write_text(json.dumps([c]))
            db = connect(r / "db")
            initialize(db)
            with db:
                import_source(db, p, "export")
            x = HistoryIndex(r / "db")
            self.assertEqual(x.terms("oldbranch"), [])
            self.assertTrue(x.terms("finalanswer"))
            self.assertEqual(x.terms("THINK_SENTINEL"), [])
            self.assertTrue(x.terms("THINK_SENTINEL", trace=True))
            active = [
                v[0]
                for v in db.execute(
                    "select native_id from searchable_chunks where origin<>'asset' order by id"
                )
            ]
            self.assertEqual(active, ["u", "thought", "answer"])
            asset = db.execute("select * from assets").fetchone()
            self.assertEqual(asset["filename"], "family.png")
            self.assertEqual(asset["mime_type"], "image/png")
            self.assertTrue(Path(asset["local_path"]).exists())
            t = db.execute("select * from threads").fetchone()
            self.assertEqual(t["archived"], 1)
            self.assertTrue(t["created_at"].startswith("2023-"))
            self.assertFalse(import_source(db, p, "export"))
            x.close()
            db.close()

    def test_codex_web_export_and_unarchive(self):
        with tempfile.TemporaryDirectory() as d:
            r = Path(d)
            p = r / "codex.json"
            db = connect(r / "db")
            initialize(db)
            c = {
                "id": "cloud",
                "title": "Cloud task",
                "archived": True,
                "turns": [
                    {
                        "role": "user",
                        "input_items": [
                            {
                                "type": "message",
                                "role": "user",
                                "content": [{"type": "input_text", "text": "CLOUD_INPUT"}],
                            }
                        ],
                    },
                    {
                        "role": "assistant",
                        "output_items": [
                            {
                                "type": "message",
                                "role": "assistant",
                                "content": [{"type": "output_text", "text": "CLOUD_OUTPUT"}],
                            }
                        ],
                    },
                ],
            }
            p.write_text(json.dumps([c]))
            with db:
                import_source(db, p, "export")
            self.assertEqual(
                db.execute("select count(*) from chunks where thread_id='codex:cloud'").fetchone()[
                    0
                ],
                2,
            )
            c = {
                "id": "g",
                "title": "Web",
                "is_archived": True,
                "mapping": {},
                "current_node": None,
            }
            p = r / "conversations.json"
            p.write_text(json.dumps([c]))
            with db:
                import_source(db, p, "export")
            self.assertEqual(
                db.execute("select archived from threads where id='chatgpt:g'").fetchone()[0],
                1,
            )
            c["is_archived"] = False
            p.write_text(json.dumps([c]))
            with db:
                import_source(db, p, "export")
            self.assertEqual(
                db.execute("select archived from threads where id='chatgpt:g'").fetchone()[0],
                0,
            )
            db.close()

    def test_graph_fallback_and_tool_classification(self):
        order, path, status = graph({"a": {"parent": "b"}, "b": {"parent": "a"}}, "a")
        self.assertEqual(status, "cycle")
        self.assertFalse(path)
        self.assertEqual(set(order), {"a", "b"})
        self.assertEqual(
            list(normalize({"author": {"role": "tool"}, "content": {"parts": ["TOOL"]}})),
            [("tool_output", "TOOL")],
        )
        self.assertEqual(
            list(
                normalize(
                    {
                        "author": {"role": "assistant"},
                        "content": {
                            "content_type": "reasoning_recap",
                            "content": "RECAP",
                        },
                    }
                )
            ),
            [("thinking", "RECAP")],
        )

    def test_sharded_zip_discovery_and_extraction_marker(self):
        with tempfile.TemporaryDirectory() as d:
            r = Path(d)
            p = r / "export.zip"
            with zipfile.ZipFile(p, "w") as z:
                z.writestr("conversations-000.json", "[]")
                z.writestr("conversations-001.json", "[]")
                z.writestr("file-image.dat", b"PNG")
            with patch.dict(
                "os.environ",
                {
                    "CODEX_HOME": str(r / "no-codex"),
                    "CLAUDE_CONFIG_DIR": str(r / "no-claude"),
                },
            ):
                files, _ = discover(r)
                self.assertEqual(len(files), 2)
                self.assertTrue((files[0][0].parent / "file-image.dat").exists())
                again, _ = discover(r)
                self.assertEqual(set(files), set(again))


if __name__ == "__main__":
    unittest.main()
