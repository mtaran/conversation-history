"""MCP protocol and reader tests using only synthetic temporary archives."""

import contextlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from history import mcp_server
from history.store import (
    ROOT,
    HistoryIndex,
    connect,
    initialize,
    put_chunk,
    upsert_thread,
)


class ReaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = connect(Path(self.tmp.name) / "test.sqlite")
        initialize(self.db)
        upsert_thread(self.db, "codex:parent", "codex", "parent", project="/ru")
        upsert_thread(
            self.db,
            "codex:fork",
            "codex",
            "fork",
            project="/ru",
            parent="codex:parent",
            metadata={"forked_from_id": "parent"},
        )
        upsert_thread(
            self.db,
            "codex:agent",
            "codex",
            "agent",
            project="/ru",
            parent="codex:parent",
            metadata={"source": {"subagent": {"thread_spawn": {}}}},
        )
        upsert_thread(
            self.db,
            "claude_code:agent",
            "claude_code",
            "agent",
            project="/ru",
            parent="claude_code:parent",
        )
        for origin, text in [
            ("user", "Russian Learner " * 180),
            ("assistant", "A useful answer"),
            ("attachment", "long attached journal " * 200),
            ("thinking", "Readable reasoning"),
            ("tool_input", "tool arguments"),
            ("tool_output", "tool output " * 250),
            ("context", "system context"),
        ]:
            put_chunk(self.db, "codex:parent", None, origin, text, origin)
        self.db.commit()

        @contextlib.contextmanager
        def fixture():
            x = HistoryIndex(Path(self.tmp.name) / "test.sqlite", readonly=True)
            try:
                yield x
            finally:
                x.close()

        self.mock = patch.object(mcp_server, "index", fixture)
        self.mock.start()

    def tearDown(self):
        self.mock.stop()
        self.db.close()
        self.tmp.cleanup()

    def test_trace_is_opt_in_and_paging_is_bounded(self):
        page = mcp_server.read_thread("codex:parent")
        self.assertLessEqual(len(page["chunks"]), 10)
        self.assertTrue(
            {c["origin"] for c in page["chunks"]} <= {"user", "assistant", "metadata", "attachment"}
        )
        all_chunks = []
        offset = 0
        while offset is not None:
            page = mcp_server.read_thread("codex:parent", offset=offset, include_trace=True)
            self.assertLessEqual(len(page["chunks"]), 10)
            all_chunks.extend(page["chunks"])
            offset = page["next_offset"]
        self.assertEqual(len({c["id"] for c in all_chunks}), len(all_chunks))
        self.assertEqual(
            {c["origin"] for c in all_chunks},
            {
                "user",
                "assistant",
                "attachment",
                "thinking",
                "tool_input",
                "tool_output",
                "context",
            },
        )
        self.assertTrue(all(len(c["text"]) <= 1000 for c in all_chunks))
        resource = json.loads(mcp_server.thread_detail_resource("codex:parent"))
        self.assertTrue(
            {c["origin"] for c in resource["chunks"]}
            <= {"user", "assistant", "metadata", "attachment"}
        )
        self.assertLessEqual(len(resource["chunks"]), 10)

    def test_forks_are_not_subagents(self):
        page = mcp_server.list_history_threads(project="/ru")
        self.assertEqual({r["id"] for r in page["threads"]}, {"codex:parent", "codex:fork"})
        self.assertEqual(
            mcp_server.list_history_threads(project="/ru", include_subagents=True)["total"],
            4,
        )

    def test_thread_scoping_in_every_search_mode(self):
        from history.store import embed_pending
        from history.test_history import FakeModel

        put_chunk(
            self.db,
            "codex:fork",
            None,
            "user",
            "Russian Learner in another conversation",
            "fork-message",
        )
        self.db.commit()
        with patch("history.store.model", return_value=FakeModel()):
            embed_pending(self.db)
            for mode in ("terms", "semantic", "hybrid"):
                with self.subTest(mode=mode):

                    def search(ids, **kwargs):
                        return mcp_server.search_history(
                            "Russian", mode=mode, thread_ids=ids, **kwargs
                        )["results"]

                    self.assertEqual({r["id"] for r in search(["codex:fork"])}, {"codex:fork"})
                    self.assertEqual(
                        {r["id"] for r in search(["codex:parent", "codex:fork"])},
                        {"codex:parent", "codex:fork"},
                    )
                    self.assertEqual(search([]), [])
                    self.assertEqual(search(["codex:missing"]), [])
                    self.assertEqual(search(["codex:fork"], project="/elsewhere"), [])
                    self.assertEqual(search(["codex:fork"], exclude_thread_ids=["codex:fork"]), [])

    def test_search_pagination_and_invalid_fts(self):
        for i in range(5):
            tid = f"codex:more{i}"
            upsert_thread(self.db, tid, "codex", f"more{i}")
            put_chunk(self.db, tid, None, "user", "Russian Learner learning", tid)
        self.db.commit()
        ids = []
        offset = 0
        while offset is not None:
            page = mcp_server.search_history(
                "Russian", mode="terms", limit=2, offset=offset, snippet_chars=0
            )
            ids.extend(r["id"] for r in page["results"])
            self.assertTrue(all(r["snippet"] == "" for r in page["results"]))
            offset = page["next_offset"]
        self.assertEqual(len(ids), 6)
        self.assertEqual(len(set(ids)), 6)
        self.assertEqual(mcp_server.search_history("Russian", mode="terms", limit=50)["count"], 6)
        excluded = mcp_server.search_history(
            "Russian", mode="terms", exclude_thread_ids=["codex:parent"]
        )
        self.assertNotIn("codex:parent", {r["id"] for r in excluded["results"]})
        with self.assertRaisesRegex(ValueError, "Invalid FTS5"):
            mcp_server.search_history('"unterminated', mode="terms")


class MCPTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from history.test_history import FakeModel

        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "history.sqlite3"
        db = connect(self.db_path)
        initialize(db)
        for i in range(60):
            tid = f"codex:example-{i}"
            upsert_thread(
                db,
                tid,
                "codex",
                f"example-{i}",
                title=f"Example thread {i}",
                project="/example/project",
            )
            put_chunk(
                db,
                tid,
                None,
                "user",
                "example learning material " * (1000 if i == 0 else 1),
                tid,
            )
        db.commit()
        with patch("history.store.model", return_value=FakeModel()):
            from history.store import embed_pending

            embed_pending(db)
        db.close()

    async def asyncTearDown(self):
        self.tmp.cleanup()

    async def test_stdio_tools_and_resources(self):
        import os

        from pydantic import AnyUrl

        env = dict(
            os.environ,
            CONVERSATION_HISTORY_ROOT=self.tmp.name,
            CONVERSATION_HISTORY_DB=str(self.db_path),
        )
        params = StdioServerParameters(command=str(ROOT / "history-mcp"), args=[], env=env)
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as client:
                await client.initialize()
                tools = (await client.list_tools()).tools
                self.assertEqual(len(tools), 6)
                self.assertTrue(all(t.annotations.readOnlyHint for t in tools))
                search = next(t for t in tools if t.name == "search_history")
                self.assertEqual(search.inputSchema["properties"]["limit"]["maximum"], 100)
                self.assertIn("thread_ids", search.inputSchema["properties"])
                self.assertIn("next_offset", search.outputSchema["properties"])

                async def call(name, args):
                    result = await client.call_tool(name, args)
                    self.assertFalse(result.isError)
                    return result.structuredContent or json.loads(result.content[0].text)

                async def resource(uri):
                    result = await client.read_resource(AnyUrl(uri))
                    return json.loads(result.contents[0].text)

                self.assertEqual(len((await client.list_resources()).resources), 2)
                self.assertEqual(len((await client.list_resource_templates()).resourceTemplates), 5)
                projects = await resource("history://projects")
                project = await resource(projects["projects"][0]["uri"])
                self.assertEqual(project["total"], 60)
                self.assertTrue((await resource(project["next_uri"]))["threads"])
                catalog = await resource("history://threads")
                self.assertEqual(len(catalog["threads"]), 50)
                self.assertEqual(len((await resource(catalog["next_uri"]))["threads"]), 10)
                detail = await resource("history://threads/codex:example-0")
                self.assertTrue((await resource(detail["next_uri"]))["chunks"])
                self.assertGreater((await call("archive_status", {}))["vectors"], 0)
                hits = await call(
                    "search_history", {"query": "example", "mode": "terms", "limit": 3}
                )
                self.assertEqual(hits["count"], 3)
                self.assertTrue(
                    (await call("read_thread", {"thread_id": "codex:example-0", "limit": 2}))[
                        "next_offset"
                    ]
                )
                self.assertEqual((await call("list_history_threads", {"limit": 2}))["total"], 60)
                self.assertEqual(
                    (await call("project_threads", {"thread_id": "codex:example-0"}))["total"],
                    60,
                )
                self.assertTrue((await call("list_history_projects", {}))["projects"])
                bad = await client.call_tool("search_history", {"query": "example", "limit": 1000})
                self.assertTrue(bad.isError)

    async def test_read_only_connection(self):
        x = HistoryIndex(self.db_path, readonly=True)
        try:
            self.assertEqual(x.db.execute("PRAGMA query_only").fetchone()[0], 1)
            with self.assertRaises(sqlite3.OperationalError):
                x.db.execute("INSERT INTO settings VALUES('write-test','no')")
        finally:
            x.close()


if __name__ == "__main__":
    unittest.main()
