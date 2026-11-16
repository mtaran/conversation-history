"""No-argument, restartable importer for local coding logs and account exports."""

from __future__ import annotations

import contextlib
import fcntl
import io
import json
import os
import sqlite3
import time
import zipfile
import zlib
from pathlib import Path

import ijson
import zstandard

from . import chatgpt
from .store import (
    DB_PATH,
    MODEL,
    ROOT,
    connect,
    digest,
    dumps,
    embed_pending,
    initialize,
    put_chunk,
    refresh_metadata,
    remove_record_chunks,
    remove_records,
    upsert_thread,
)

CONTEXT = (
    "<",
    "# AGENTS.md",
    "The following is the Codex agent history",
    "You are an AI assistant",
)


def text_parts(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(text_parts(v) for v in value)
    if isinstance(value, dict):
        return str(value.get("text") or value.get("thinking") or "")
    return ""


def content(db, tid, rid, value, role, identity, fmt="", native=None, parent=None, active=True):
    if isinstance(value, str):
        origin = "context" if role == "user" and value.lstrip().startswith(CONTEXT) else role
        put_chunk(db, tid, rid, origin, value, identity, fmt, native, parent, active)
        return
    if not isinstance(value, list):
        return
    for i, b in enumerate(value):
        if not isinstance(b, dict):
            continue
        typ = b.get("type", "")
        key = identity + ":" + str(i)
        if typ in ("text", "input_text", "output_text"):
            content(db, tid, rid, b.get("text", ""), role, key, fmt, native, parent, active)
        elif typ in ("thinking", "reasoning"):
            put_chunk(
                db,
                tid,
                rid,
                "thinking",
                text_parts(b),
                key,
                fmt,
                native,
                parent,
                active,
            )
        elif typ == "tool_use":
            put_chunk(
                db,
                tid,
                rid,
                "tool_input",
                dumps({"name": b.get("name"), "input": b.get("input")}),
                key,
                fmt,
                native,
                parent,
                active,
            )
        elif typ == "tool_result":
            put_chunk(
                db,
                tid,
                rid,
                "tool_output",
                text_parts(b.get("content")) or dumps(b.get("content")),
                key,
                fmt,
                native,
                parent,
                active,
            )
        elif typ in ("document", "artifact"):
            put_chunk(
                db,
                tid,
                rid,
                "attachment",
                text_parts(b.get("content")) or text_parts(b),
                key,
                fmt,
                native,
                parent,
                active,
            )


def record(db, sid, ordinal, tid, obj, salt=""):
    raw = dumps(obj)
    h = digest(salt + raw)
    old = db.execute(
        "SELECT id,fingerprint FROM records WHERE source_id=? AND ordinal=?",
        (sid, ordinal),
    ).fetchone()
    if old and old["fingerprint"] == h:
        return old["id"], False
    if old:
        remove_record_chunks(db, old["id"])
        db.execute("DELETE FROM records WHERE id=?", (old["id"],))
    cur = db.execute(
        "INSERT INTO records(source_id,ordinal,thread_id,fingerprint,raw_zlib) VALUES(?,?,?,?,?)",
        (sid, ordinal, tid, h, zlib.compress(raw.encode())),
    )
    return cur.lastrowid, True


def normalize_native(db, kind, path, tid, rid, r, identity):
    payload = r.get("payload", {})
    typ = r.get("type")
    native = r.get("uuid")
    fmt = ""
    if kind == "claude_code":
        if typ in ("custom-title", "ai-title"):
            title = r.get("customTitle") or r.get("aiTitle")
            if title:
                meta = json.loads(
                    db.execute("SELECT metadata FROM threads WHERE id=?", (tid,)).fetchone()[0]
                )
                if typ == "custom-title":
                    meta["custom_title"] = title
                if typ == "custom-title" or not meta.get("custom_title"):
                    db.execute(
                        "UPDATE threads SET title=?,metadata=? WHERE id=?",
                        (title, dumps(meta), tid),
                    )
        parent = r.get("parentUuid")
        m = r.get("message", {})
        if r.get("cwd"):
            db.execute("UPDATE threads SET project=? WHERE id=?", (r["cwd"], tid))
        session = r.get("sessionId")
        if (path.stem.startswith("agent-") or "subagents" in path.parts) and session:
            db.execute(
                "UPDATE threads SET parent_id=? WHERE id=?",
                ("claude_code:" + session, tid),
            )
        if typ in ("user", "assistant") and isinstance(m, dict):
            content(
                db,
                tid,
                rid,
                m.get("content"),
                m.get("role", typ),
                identity,
                native=native,
                parent=parent,
            )
        return
    if typ == "session_meta":
        source = payload.get("source")
        sub = source.get("subagent", {}) if isinstance(source, dict) else {}
        parent = (
            payload.get("parent_thread_id")
            or payload.get("forked_from_id")
            or sub.get("thread_spawn", {}).get("parent_thread_id")
        )
        review = sub.get("other") == "guardian" or payload.get("thread_source") == "guardian_review"
        upsert_thread(
            db,
            tid,
            kind,
            tid.split(":", 1)[1],
            project=payload.get("cwd", ""),
            created=payload.get("timestamp"),
            parent="codex:" + parent if parent else None,
            review=review,
            metadata=payload,
        )
        return
    if typ == "event_msg" and isinstance(payload, dict):
        if payload.get("type") in ("user_message", "agent_message"):
            content(
                db,
                tid,
                rid,
                payload.get("message"),
                "user" if payload["type"] == "user_message" else "assistant",
                identity,
                "event",
            )
        return
    # Legacy Codex writes flat response items; modern logs wrap the same payloads.
    p = payload if typ == "response_item" else r
    fmt = "response" if typ == "response_item" else "legacy"
    if not isinstance(p, dict):
        return
    t = p.get("type")
    native = p.get("id")
    if t == "message" and p.get("role") in ("user", "assistant", "system", "developer"):
        content(
            db,
            tid,
            rid,
            p.get("content"),
            p["role"] if p["role"] in ("user", "assistant") else "context",
            identity,
            fmt,
            native,
        )
    elif t == "reasoning":
        put_chunk(
            db,
            tid,
            rid,
            "thinking",
            text_parts(p.get("summary", p.get("content"))),
            identity,
            fmt,
            native,
        )
    elif t in ("function_call", "custom_tool_call"):
        put_chunk(
            db,
            tid,
            rid,
            "tool_input",
            dumps(
                {
                    "name": p.get("name"),
                    "arguments": p.get("arguments", p.get("input")),
                    "call_id": p.get("call_id"),
                }
            ),
            identity,
            fmt,
            native,
        )
    elif t in ("function_call_output", "custom_tool_call_output"):
        output = p.get("output")
        put_chunk(
            db,
            tid,
            rid,
            "tool_output",
            output if isinstance(output, str) else dumps(output),
            identity,
            fmt,
            native,
        )


def codex_export_conversation(db, sid, ordinal, c):
    native = str(c["id"])
    tid = "codex:" + native
    upsert_thread(
        db,
        tid,
        "codex",
        native,
        c.get("title") or "",
        archived=int(bool(c.get("archived"))),
        metadata={"export_format": "codex_web", "turn_count": len(c.get("turns", []))},
    )
    rid, changed = record(db, sid, ordinal, tid, c, salt="codex-export:1:")
    if changed:
        for i, turn in enumerate(c.get("turns", [])):
            for j, item in enumerate(turn.get("input_items", turn.get("output_items", []))):
                if isinstance(item, dict):
                    normalize_native(
                        db,
                        "codex",
                        Path("codex.json"),
                        tid,
                        rid,
                        item,
                        f"{sid}:{ordinal}:{i}:{j}",
                    )


def web_conversation(db, sid, ordinal, c):
    if "turns" in c and "mapping" not in c and "chat_messages" not in c:
        codex_export_conversation(db, sid, ordinal, c)
        return
    provider = "chatgpt" if "mapping" in c else "claude_web"
    native = str(c.get("uuid") or c.get("id") or c.get("conversation_id") or digest(dumps(c))[:32])
    tid = provider + ":" + native
    upsert_thread(
        db,
        tid,
        provider,
        native,
        c.get("name") or c.get("title") or "",
        created=chatgpt.timestamp(c.get("created_at") or c.get("create_time")),
        archived=int(bool(c.get("is_archived"))),
        metadata={k: v for k, v in c.items() if k not in ("chat_messages", "mapping")},
    )
    if provider == "chatgpt":
        db.execute(
            "UPDATE threads SET archived=? WHERE id=?",
            (int(bool(c.get("is_archived"))), tid),
        )
    rid, changed = record(
        db, sid, ordinal, tid, c, salt="chatgpt:2:" if provider == "chatgpt" else ""
    )
    if not changed:
        return
    if provider == "claude_web":
        for i, m in enumerate(c.get("chat_messages", [])):
            identity = f"{sid}:{ordinal}:{i}"
            role = "user" if m.get("sender") == "human" else "assistant"
            value = m.get("content") or m.get("text", "")
            content(
                db,
                tid,
                rid,
                value,
                role,
                identity,
                native=m.get("uuid"),
                parent=m.get("parent_message_uuid"),
            )
            for j, a in enumerate(m.get("attachments", [])):
                put_chunk(
                    db,
                    tid,
                    rid,
                    "attachment",
                    a.get("extracted_content", ""),
                    identity + ":attachment:" + str(j),
                )
    else:
        mapping = c.get("mapping") or {}
        ordered, active, status = chatgpt.graph(mapping, c.get("current_node"))
        folder = Path(
            db.execute("SELECT path FROM sources WHERE id=?", (sid,)).fetchone()[0]
        ).parent
        meta = {k: v for k, v in c.items() if k not in ("mapping", "chat_messages")}
        meta["_history_import"] = {"adapter": "chatgpt:2", "branch_status": status}
        db.execute("UPDATE threads SET metadata=? WHERE id=?", (dumps(meta), tid))
        for node_id in ordered:
            node = mapping[node_id]
            m = node.get("message")
            identity = f"{sid}:{ordinal}:{node_id}"
            if not isinstance(m, dict):
                continue
            is_active = not active or node_id in active
            for i, (origin, text) in enumerate(chatgpt.normalize(m)):
                put_chunk(
                    db,
                    tid,
                    rid,
                    origin,
                    text,
                    identity + ":" + str(i),
                    native=node_id,
                    parent=node.get("parent"),
                    active=is_active,
                )
            refs = list(chatgpt.pointers(m.get("content")))
            for a in (m.get("metadata") or {}).get("attachments") or []:
                if isinstance(a, dict) and a.get("id"):
                    refs.append((a["id"], a))
            for j, (pointer, ref) in enumerate(refs):
                asset = chatgpt.resolve_asset(
                    folder, pointer, ref.get("name"), ref.get("mime_type")
                )
                db.execute(
                    """INSERT OR REPLACE INTO assets(record_id,thread_id,native_message_id,pointer,filename,local_path,mime_type,metadata)
                 VALUES(?,?,?,?,?,?,?,?)""",
                    (
                        rid,
                        tid,
                        node_id,
                        pointer,
                        asset["filename"],
                        asset["local_path"],
                        asset["mime_type"],
                        dumps(ref),
                    ),
                )
                label = asset["filename"] or pointer
                put_chunk(
                    db,
                    tid,
                    rid,
                    "asset",
                    label,
                    identity + ":asset:" + str(j),
                    native=node_id,
                    parent=node.get("parent"),
                    active=is_active,
                )
                extracted = ref.get("extracted_content")
                if isinstance(extracted, str):
                    put_chunk(
                        db,
                        tid,
                        rid,
                        "attachment",
                        extracted,
                        identity + ":attachment:" + str(j),
                        native=node_id,
                        parent=node.get("parent"),
                        active=is_active,
                    )


def edges(path, offset):
    with path.open("rb") as f:
        prefix = f.read(min(offset, 4096))
        f.seek(max(0, offset - 4096))
        edge = f.read(min(offset, 4096))
    return digest(prefix), digest(edge)


def import_source(db, path, kind):
    stat = path.stat()
    old = db.execute("SELECT * FROM sources WHERE path=?", (str(path),)).fetchone()
    if (
        old
        and old["status"] == "ok"
        and old["size"] == stat.st_size
        and old["mtime"] == stat.st_mtime_ns
    ):
        return False
    if not old:
        db.execute("INSERT INTO sources(path,kind) VALUES(?,?)", (str(path), kind))
        old = db.execute("SELECT * FROM sources WHERE path=?", (str(path),)).fetchone()
    sid = old["id"]
    compressed = path.name.endswith(".zst")
    offset = old["offset"]
    append = False
    if kind != "export" and not compressed and offset and stat.st_size > old["size"]:
        append = edges(path, offset) == (old["prefix_hash"], old["edge_hash"])
    if kind == "export":
        with path.open("rb") as f:
            for ordinal, c in enumerate(ijson.items(f, "item", use_float=True)):
                if isinstance(c, dict):
                    web_conversation(db, sid, ordinal, c)
        count = locals().get("ordinal", -1) + 1
        stale = [
            r[0]
            for r in db.execute(
                "SELECT id FROM records WHERE source_id=? AND ordinal>=?", (sid, count)
            )
        ]
        for rid in stale:
            remove_record_chunks(db, rid)
            db.execute("DELETE FROM records WHERE id=?", (rid,))
        offset = 0
        tid = None
        prefix = edge = None
        partial = False
    else:
        if not append:
            remove_records(db, sid)
            offset = 0
        ordinal = db.execute(
            "SELECT coalesce(max(ordinal)+1,0) FROM records WHERE source_id=?", (sid,)
        ).fetchone()[0]
        tid = old["first_thread"] if append else None
        partial = False
        with path.open("rb") as source:
            reader = (
                io.BufferedReader(zstandard.ZstdDecompressor().stream_reader(source))
                if compressed
                else source
            )
            if append:
                reader.seek(offset)
            while True:
                line = reader.readline()
                if not line:
                    break
                if not line.endswith(b"\n") and not compressed:
                    partial = True
                    break
                offset += len(line)
                try:
                    r = json.loads(line)
                except ValueError:
                    db.execute(
                        "INSERT INTO records(source_id,ordinal,fingerprint,raw_zlib,error) VALUES(?,?,?,?,?)",
                        (
                            sid,
                            ordinal,
                            digest(line),
                            zlib.compress(line),
                            "Invalid JSON line",
                        ),
                    )
                    ordinal += 1
                    continue
                if not isinstance(r, dict):
                    ordinal += 1
                    continue
                if not tid:
                    native = path.stem
                    if kind == "codex":
                        native = (
                            r.get("payload", {}).get("id")
                            if r.get("type") == "session_meta"
                            else r.get("id")
                        )
                        native = native or path.name.removesuffix(".zst").removesuffix(".jsonl")
                    tid = kind + ":" + native
                    upsert_thread(
                        db,
                        tid,
                        kind,
                        native,
                        project=r.get("cwd", ""),
                        created=r.get("timestamp"),
                        archived=int("archived_sessions" in path.parts),
                    )
                rid, changed = record(db, sid, ordinal, tid, r)
                if changed:
                    normalize_native(db, kind, path, tid, rid, r, f"{sid}:{ordinal}")
                ordinal += 1
        prefix, edge = edges(path, offset) if not compressed else (None, None)
    db.execute(
        """UPDATE sources SET size=?,mtime=?,offset=?,prefix_hash=?,edge_hash=?,first_thread=?,status=? WHERE id=?""",
        (
            stat.st_size,
            stat.st_mtime_ns,
            offset,
            prefix,
            edge,
            tid,
            "partial" if partial else "ok",
            sid,
        ),
    )
    return True


def refresh_assets(db):
    rows = db.execute("""SELECT a.*,s.path source_path FROM assets a JOIN records r ON r.id=a.record_id JOIN sources s ON s.id=r.source_id
      WHERE a.local_path IS NULL OR a.filename IS NULL""").fetchall()
    updated = 0
    for a in rows:
        resolved = chatgpt.resolve_asset(
            Path(a["source_path"]).parent, a["pointer"], a["filename"], a["mime_type"]
        )
        if resolved["local_path"] != a["local_path"] or resolved["filename"] != a["filename"]:
            db.execute(
                "UPDATE assets SET filename=?,local_path=?,mime_type=? WHERE id=?",
                (
                    resolved["filename"],
                    resolved["local_path"],
                    resolved["mime_type"],
                    a["id"],
                ),
            )
            if resolved["filename"]:
                db.execute(
                    "UPDATE chunks SET text=?,text_hash=? WHERE record_id=? AND native_id=? AND origin='asset' AND text=?",
                    (
                        resolved["filename"],
                        digest(resolved["filename"]),
                        a["record_id"],
                        a["native_message_id"],
                        a["filename"] or a["pointer"],
                    ),
                )
            updated += 1
    return updated


def catalog(db, home):
    paths = list(home.glob("state_*.sqlite"))
    if not paths:
        return
    path = max(paths, key=lambda p: int(p.stem.split("_")[-1]))
    with contextlib.closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as src:
        src.row_factory = sqlite3.Row
        for r in src.execute("SELECT * FROM threads"):
            r = dict(r)
            tid = "codex:" + r["id"]
            if not db.execute("SELECT 1 FROM threads WHERE id=?", (tid,)).fetchone():
                continue
            title = r.get("name") or r.get("title") or ""
            db.execute(
                "UPDATE threads SET title=?,project=?,archived=? WHERE id=?",
                (title, r.get("cwd", ""), r.get("archived", 0), tid),
            )
    # Claude Code titles/cwd come from transcript records; no lossy hyphen decoding.


def discover(root=ROOT):
    ch = Path(os.environ.get("CLAUDE_CONFIG_DIR", str(Path.home() / ".claude"))).expanduser()
    co = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))).expanduser()
    files = []
    for kind, folder in [
        ("claude_code", ch / "projects"),
        ("codex", co / "sessions"),
        ("codex", co / "archived_sessions"),
    ]:
        if folder.exists():
            files += [
                (p, kind)
                for p in folder.rglob("*")
                if p.is_file() and (p.name.endswith(".jsonl") or p.name.endswith(".jsonl.zst"))
            ]
    # Only source export locations, never the generated comparison/report copies.
    for folder in [
        root,
        root / "extracted",
        root / "exports",
        root / ".history-imports",
    ]:
        if not folder.exists():
            continue
        candidates = (
            folder.glob("conversations*.json")
            if folder == root
            else folder.rglob("conversations*.json")
        )
        files += [(p, "export") for p in candidates if p.is_file()]
        if folder != root:
            files += [(p, "export") for p in folder.rglob("codex.json") if p.is_file()]
    extracted_archives = set()
    for folder in [root / "exports", root / "extracted"]:
        if folder.exists():
            for marker in folder.rglob(".archive-source.json"):
                try:
                    info = json.loads(marker.read_text())
                    archive = Path(info["path"])
                    if (
                        archive.exists()
                        and archive.stat().st_size == info["size"]
                        and archive.stat().st_mtime_ns == info["mtime"]
                    ):
                        extracted_archives.add(archive.resolve())
                except (ValueError, KeyError, OSError):
                    pass
    for p in root.glob("*.zip"):
        if p.resolve() in extracted_archives:
            continue
        if (root / "extracted" / p.stem).exists():
            continue
        with zipfile.ZipFile(p) as z:
            members = [
                i
                for i in z.infolist()
                if (
                    Path(i.filename).name.startswith("conversations")
                    or Path(i.filename).name == "codex.json"
                )
                and i.filename.endswith(".json")
            ]
            if not members:
                continue
            bundle = root / ".history-imports" / digest(str(p) + str(p.stat().st_mtime_ns))
            marker = bundle / ".archive-source.json"
            if not marker.exists():
                bundle.mkdir(parents=True, exist_ok=True)
                for info in z.infolist():
                    if not (bundle / info.filename).resolve().is_relative_to(bundle.resolve()):
                        raise ValueError("Unsafe ZIP member path")
                z.extractall(bundle)
                marker.write_text(
                    dumps(
                        {
                            "path": str(p.resolve()),
                            "size": p.stat().st_size,
                            "mtime": p.stat().st_mtime_ns,
                        }
                    )
                )
            files.extend((bundle / i.filename, "export") for i in members)
    return sorted(set(files)), co


def run(root=ROOT, db_path=DB_PATH, embeddings=True):
    started = time.monotonic()
    db = connect(db_path)
    initialize(db)
    changed = 0
    errors = []
    files, home = discover(root)
    print(f"Scanning {len(files)} source files", flush=True)
    for i, (path, kind) in enumerate(files):
        try:
            with db:
                if import_source(db, path, kind):
                    changed += 1
        except Exception as e:
            errors.append({"path": str(path), "error": type(e).__name__, "detail": str(e)[:250]})
        if (i + 1) % 50 == 0:
            print(f"  checked {i+1}/{len(files)}; changed {changed}", flush=True)
    with db:
        catalog(db, home)
        refresh_metadata(db)
        refresh_assets(db)
    vectors = embed_pending(db) if embeddings else 0
    with db:
        db.execute("INSERT OR REPLACE INTO settings VALUES('embedding_model',?)", (MODEL,))
    report = {
        "sources": len(files),
        "changed": changed,
        "vectors_added": vectors,
        "threads": db.execute("SELECT count(*) FROM threads").fetchone()[0],
        "chunks": db.execute("SELECT count(*) FROM chunks").fetchone()[0],
        "seconds": round(time.monotonic() - started, 2),
        "errors": errors,
    }
    (root / "history-last-run.json").write_text(json.dumps(report, indent=2))
    db.close()
    print(dumps(report), flush=True)
    return report


def main():
    with (ROOT / ".history-index.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another indexer is already running.")
        result = run()
        if result["errors"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
