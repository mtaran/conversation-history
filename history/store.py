"""Local trace archive, full-text retrieval and sqlite-vec semantic retrieval."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import zlib
from pathlib import Path

import sqlite_vec

ROOT = Path(
    os.environ.get("CONVERSATION_HISTORY_ROOT", str(Path(__file__).resolve().parent.parent))
).resolve()
DB_PATH = Path(os.environ.get("CONVERSATION_HISTORY_DB", str(ROOT / "history.sqlite3"))).resolve()
VERSION = "1"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
_model = None


def digest(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode()).hexdigest()


def dumps(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def model():
    global _model
    if _model is None:
        # Explicitly offline: histories and query text never leave this machine.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        import torch
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(4)
        cache = (
            Path.home()
            / ".cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots"
        )
        snapshot = cache / MODEL_REVISION
        location = str(snapshot) if (snapshot / "model.safetensors").exists() else MODEL
        _model = SentenceTransformer(
            location, revision=MODEL_REVISION, local_files_only=True, device="cpu"
        )
    return _model


def connect(path=DB_PATH, readonly=False):
    location = Path(path).resolve().as_uri() + "?mode=ro" if readonly else path
    db = sqlite3.connect(location, timeout=60, uri=readonly)
    db.row_factory = sqlite3.Row
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    db.execute("PRAGMA foreign_keys=ON")
    if readonly:
        db.execute("PRAGMA query_only=ON")
    else:
        db.execute("PRAGMA journal_mode=WAL")
    return db


def initialize(db):
    had_fts = db.execute("SELECT 1 FROM sqlite_master WHERE name='chunks_fts'").fetchone()
    db.executescript("""
    CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY,value TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS threads(
      id TEXT PRIMARY KEY,provider TEXT NOT NULL,native_id TEXT NOT NULL,
      title TEXT NOT NULL DEFAULT '',project TEXT NOT NULL DEFAULT '',created_at TEXT,
      parent_id TEXT,archived INTEGER NOT NULL DEFAULT 0,is_review INTEGER NOT NULL DEFAULT 0,
      metadata TEXT NOT NULL DEFAULT '{}');
    CREATE TABLE IF NOT EXISTS sources(
      id INTEGER PRIMARY KEY,path TEXT NOT NULL UNIQUE,kind TEXT NOT NULL,
      size INTEGER,mtime INTEGER,offset INTEGER NOT NULL DEFAULT 0,
      prefix_hash TEXT,edge_hash TEXT,first_thread TEXT,status TEXT NOT NULL DEFAULT 'new');
    CREATE TABLE IF NOT EXISTS records(
      id INTEGER PRIMARY KEY,source_id INTEGER NOT NULL REFERENCES sources(id),
      ordinal INTEGER NOT NULL,thread_id TEXT REFERENCES threads(id),fingerprint TEXT NOT NULL,
      raw_zlib BLOB NOT NULL,error TEXT,UNIQUE(source_id,ordinal));
    CREATE TABLE IF NOT EXISTS chunks(
      id INTEGER PRIMARY KEY,record_id INTEGER REFERENCES records(id),thread_id TEXT NOT NULL REFERENCES threads(id),
      identity TEXT NOT NULL UNIQUE,origin TEXT NOT NULL,text TEXT NOT NULL,
      text_hash TEXT NOT NULL,format TEXT NOT NULL DEFAULT '',native_id TEXT,parent_native TEXT,
      active INTEGER NOT NULL DEFAULT 1,embed_hash TEXT);
    CREATE TABLE IF NOT EXISTS assets(
      id INTEGER PRIMARY KEY,record_id INTEGER NOT NULL REFERENCES records(id) ON DELETE CASCADE,
      thread_id TEXT NOT NULL REFERENCES threads(id),native_message_id TEXT,pointer TEXT NOT NULL,
      filename TEXT,local_path TEXT,mime_type TEXT,metadata TEXT NOT NULL DEFAULT '{}',
      UNIQUE(record_id,native_message_id,pointer));
    CREATE INDEX IF NOT EXISTS assets_thread ON assets(thread_id);
    CREATE INDEX IF NOT EXISTS assets_record ON assets(record_id);
    CREATE INDEX IF NOT EXISTS chunks_thread ON chunks(thread_id);
    CREATE INDEX IF NOT EXISTS chunks_record ON chunks(record_id);
    CREATE INDEX IF NOT EXISTS chunks_mirrors ON chunks(thread_id,text_hash,format);
    CREATE INDEX IF NOT EXISTS records_source ON records(source_id);
    CREATE INDEX IF NOT EXISTS chunks_embeddings ON chunks(id) WHERE embed_hash IS NOT NULL AND active=1;
    CREATE INDEX IF NOT EXISTS threads_project ON threads(project);
    CREATE TABLE IF NOT EXISTS embedding_cache(hash TEXT PRIMARY KEY,vector BLOB NOT NULL);
    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text,content='chunks',content_rowid='id',tokenize='unicode61 remove_diacritics 2');
    CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
      INSERT INTO chunks_fts(rowid,text) VALUES(new.id,new.text);END;
    CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
      INSERT INTO chunks_fts(chunks_fts,rowid,text) VALUES('delete',old.id,old.text);END;
    CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE OF text ON chunks BEGIN
      INSERT INTO chunks_fts(chunks_fts,rowid,text) VALUES('delete',old.id,old.text);
      INSERT INTO chunks_fts(rowid,text) VALUES(new.id,new.text);END;
    CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vectors USING vec0(chunk_id INTEGER PRIMARY KEY,embedding float[384] distance_metric=cosine);
    CREATE VIEW IF NOT EXISTS searchable_chunks AS
      SELECT c.* FROM chunks c JOIN threads t ON t.id=c.thread_id
      WHERE c.active=1 AND t.is_review=0
      AND NOT(c.format='event' AND EXISTS(
        SELECT 1 FROM chunks r WHERE r.thread_id=c.thread_id AND r.text_hash=c.text_hash AND r.format='response'));
    """)
    if not had_fts:
        db.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
    v = db.execute("SELECT value FROM settings WHERE key='schema_version'").fetchone()
    if v and v[0] != VERSION:
        raise RuntimeError("Unsupported database schema; move history.sqlite3 aside to rebuild.")
    db.execute("INSERT OR IGNORE INTO settings VALUES('schema_version',?)", (VERSION,))
    db.commit()


def upsert_thread(
    db,
    tid,
    provider,
    native,
    title="",
    project="",
    created=None,
    parent=None,
    archived=0,
    review=0,
    metadata=None,
):
    db.execute(
        """INSERT INTO threads(id,provider,native_id,title,project,created_at,parent_id,archived,is_review,metadata)
    VALUES(?,?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET
    title=CASE WHEN excluded.title<>'' THEN excluded.title ELSE title END,
    project=CASE WHEN excluded.project<>'' THEN excluded.project ELSE project END,
    created_at=coalesce(created_at,excluded.created_at),parent_id=coalesce(excluded.parent_id,parent_id),
    archived=max(archived,excluded.archived),is_review=max(is_review,excluded.is_review),
    metadata=CASE WHEN excluded.metadata<>'{}' THEN excluded.metadata ELSE metadata END""",
        (
            tid,
            provider,
            native,
            title,
            project,
            created,
            parent,
            archived,
            review,
            dumps(metadata or {}),
        ),
    )


def remove_records(db, source_id):
    ids = db.execute(
        "SELECT c.id FROM chunks c JOIN records r ON r.id=c.record_id WHERE r.source_id=?",
        (source_id,),
    ).fetchall()
    db.executemany("DELETE FROM chunk_vectors WHERE chunk_id=?", [(r[0],) for r in ids])
    db.execute(
        "DELETE FROM chunks WHERE record_id IN (SELECT id FROM records WHERE source_id=?)",
        (source_id,),
    )
    db.execute("DELETE FROM records WHERE source_id=?", (source_id,))


def remove_record_chunks(db, rid):
    ids = db.execute("SELECT id FROM chunks WHERE record_id=?", (rid,)).fetchall()
    db.executemany("DELETE FROM chunk_vectors WHERE chunk_id=?", [(r[0],) for r in ids])
    db.execute("DELETE FROM chunks WHERE record_id=?", (rid,))


def put_chunk(
    db,
    thread,
    record,
    origin,
    text,
    identity,
    fmt="",
    native=None,
    parent=None,
    active=True,
):
    if not isinstance(text, str) or not text.strip():
        return
    # Stable content-based chunk IDs avoid re-embedding unchanged message bodies.
    for i, start in enumerate(range(0, len(text), 800)):
        part = text[start : start + 1000]
        key = digest(identity + ":" + str(i) + ":" + part)
        eligible = origin in ("user", "assistant", "metadata", "attachment")
        eh = digest(MODEL + ":v1:" + part) if eligible else None
        db.execute(
            """INSERT OR IGNORE INTO chunks(record_id,thread_id,identity,origin,text,text_hash,format,native_id,parent_native,active,embed_hash)
        VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
            (
                record,
                thread,
                key,
                origin,
                part,
                digest(part),
                fmt,
                native,
                parent,
                int(active),
                eh,
            ),
        )


def refresh_metadata(db):
    db.execute("""UPDATE threads SET title=coalesce((SELECT substr(c.text,1,160) FROM chunks c
      WHERE c.thread_id=threads.id AND c.origin='user' ORDER BY c.id LIMIT 1),'') WHERE title=''""")
    for t in db.execute("SELECT * FROM threads").fetchall():
        text = t["title"] + "\nProject: " + t["project"]
        identity = "metadata:" + t["id"]
        fingerprint = digest(text)
        old = db.execute("SELECT text_hash,id FROM chunks WHERE identity=?", (identity,)).fetchone()
        if old and old["text_hash"] == fingerprint:
            continue
        if old:
            db.execute("DELETE FROM chunk_vectors WHERE chunk_id=?", (old["id"],))
            db.execute("DELETE FROM chunks WHERE id=?", (old["id"],))
        if text.strip():
            db.execute(
                """INSERT INTO chunks(thread_id,identity,origin,text,text_hash,embed_hash) VALUES(?,?,'metadata',?,?,?)""",
                (t["id"], identity, text, fingerprint, digest(MODEL + ":v1:" + text)),
            )


def embed_pending(db, batch_size=128):
    # Vector storage is a persistent exact KNN index, not an approximate HNSW index.
    count = 0
    pending = db.execute("""SELECT c.id,c.text,c.embed_hash FROM chunks c INDEXED BY chunks_embeddings JOIN threads t ON t.id=c.thread_id
        LEFT JOIN chunk_vectors v ON v.chunk_id=c.id
        WHERE c.embed_hash IS NOT NULL AND c.active=1 AND t.is_review=0 AND v.chunk_id IS NULL""").fetchall()
    for start in range(0, len(pending), batch_size):
        rows = pending[start : start + batch_size]
        missing = {
            r["embed_hash"]: r["text"]
            for r in rows
            if not db.execute(
                "SELECT 1 FROM embedding_cache WHERE hash=?", (r["embed_hash"],)
            ).fetchone()
        }
        if missing:
            vectors = model().encode(
                list(missing.values()),
                batch_size=64,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            for h, v in zip(missing, vectors):
                db.execute(
                    "INSERT OR IGNORE INTO embedding_cache VALUES(?,?)",
                    (h, v.astype("float32").tobytes()),
                )
        for r in rows:
            blob = db.execute(
                "SELECT vector FROM embedding_cache WHERE hash=?", (r["embed_hash"],)
            ).fetchone()[0]
            db.execute("INSERT INTO chunk_vectors VALUES(?,?)", (r["id"], blob))
        count += len(rows)
        db.commit()
        if count % 1024 == 0:
            print(f"  vectors indexed: {count}", flush=True)
    return count


class HistoryIndex:
    def __init__(self, path=DB_PATH, readonly=False):
        self.db = connect(path, readonly=readonly)

    def close(self):
        self.db.close()

    def _scope(self, project, trace, thread_ids=None):
        clause = ""
        args = []
        if project:
            clause += " AND (t.project=? OR t.project LIKE ?)"
            args += [str(project).rstrip("/"), str(project).rstrip("/") + "/%"]
        if not trace:
            clause += " AND c.origin IN ('user','assistant','metadata','attachment','asset')"
        if thread_ids is not None:
            clause += " AND t.id IN (" + ",".join("?" for _ in thread_ids) + ")"
            args.extend(thread_ids)
        return clause, args

    def terms(self, query, limit=20, project=None, trace=False, thread_ids=None):
        """FTS5 expressions: Learner, russian AND games, \"spaced repetition\", arasaac*."""
        if thread_ids == []:
            return []
        scope, args = self._scope(project, trace, thread_ids)
        rows = self.db.execute(
            """SELECT t.*,c.id chunk_id,c.origin,c.text snippet,bm25(chunks_fts) score
          FROM chunks_fts JOIN searchable_chunks c ON c.id=chunks_fts.rowid JOIN threads t ON t.id=c.thread_id
          WHERE chunks_fts MATCH ?"""
            + scope
            + " ORDER BY score LIMIT 10000",
            [query, *args],
        ).fetchall()
        return self._group(rows, limit)

    def semantic(self, query, limit=20, project=None, thread_ids=None):
        if thread_ids == []:
            return []
        v = model().encode([query], normalize_embeddings=True)[0].astype("float32").tobytes()
        scope, args = self._scope(project, False, thread_ids)
        if project or thread_ids is not None:
            rows = self.db.execute(
                """SELECT t.*,c.id chunk_id,c.origin,c.text snippet,vec_distance_cosine(v.embedding,?) score
            FROM chunk_vectors v JOIN searchable_chunks c ON c.id=v.chunk_id JOIN threads t ON t.id=c.thread_id
            WHERE 1=1"""
                + scope
                + " ORDER BY score",
                [v, *args],
            ).fetchall()
        else:
            # Oversample progressively so long threads do not monopolize the result page.
            k = min(256, self.db.execute("SELECT count(*) FROM chunk_vectors").fetchone()[0])
            rows = []
            while k:
                rows = self.db.execute(
                    """SELECT t.*,c.id chunk_id,c.origin,c.text snippet,v.distance score
                FROM chunk_vectors v JOIN searchable_chunks c ON c.id=v.chunk_id JOIN threads t ON t.id=c.thread_id
                WHERE v.embedding MATCH ? AND k=? ORDER BY v.distance""",
                    (v, k),
                ).fetchall()
                if len({r["id"] for r in rows}) >= limit or k >= 4096:
                    break
                k = min(k * 2, 4096)
        return self._group(rows, limit)

    @staticmethod
    def _group(rows, limit):
        out = []
        seen = set()
        for r in rows:
            if r["id"] in seen:
                continue
            seen.add(r["id"])
            out.append(dict(r))
            if len(out) >= limit:
                break
        return out

    def hybrid(self, query, terms=None, limit=20, project=None, thread_ids=None):
        """Reciprocal-rank fusion of semantic and lexical thread results."""
        import re

        if terms is None:
            words = re.findall(r"[^\W_]+", query, flags=re.UNICODE)
            terms = " OR ".join('"' + w + '"' for w in words)
        lexical = (
            self.terms(terms, limit=max(100, limit), project=project, thread_ids=thread_ids)
            if terms
            else []
        )
        semantic = self.semantic(
            query, limit=max(100, limit), project=project, thread_ids=thread_ids
        )
        scores = {}
        items = {}
        for results in (lexical, semantic):
            for rank, r in enumerate(results, 1):
                items.setdefault(r["id"], r)
                scores[r["id"]] = scores.get(r["id"], 0) + 1 / (60 + rank)
        return [
            {**items[k], "fusion_score": scores[k]}
            for k in sorted(scores, key=scores.get, reverse=True)[:limit]
        ]

    def related_project(self, thread_id):
        """Expand a topic match to neighboring threads in the same workspace."""
        return [
            dict(r)
            for r in self.db.execute(
                """SELECT * FROM threads WHERE project=(SELECT project FROM threads WHERE id=?)
          AND project<>'' AND is_review=0 ORDER BY created_at""",
                (thread_id,),
            )
        ]

    def records(self, thread_id):
        return [
            json.loads(zlib.decompress(r[0]))
            for r in self.db.execute(
                "SELECT raw_zlib FROM records WHERE thread_id=? AND error IS NULL ORDER BY source_id,ordinal",
                (thread_id,),
            )
        ]
