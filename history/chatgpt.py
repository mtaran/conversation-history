"""ChatGPT export graph/content adapter. No private web API calls."""

import json
import mimetypes
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from urllib.parse import unquote


def timestamp(value):
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value, timezone.utc).isoformat()
        except (ValueError, OverflowError, OSError):
            return None
    return value if isinstance(value, str) else None


def graph(mapping, leaf):
    """Preserve all nodes; default to the active root-to-leaf path when valid."""
    active = set()
    p = leaf
    status = "active"
    if not leaf:
        status = "missing_leaf"
    while p:
        if p in active:
            status = "cycle"
            break
        if p not in mapping or not isinstance(mapping[p], dict):
            status = "broken_parent"
            break
        active.add(p)
        p = mapping[p].get("parent")
    if status != "active":
        active = set()  # Explicit fallback to all branches, never arbitrary loss.
    children = {}
    roots = []
    for k, n in mapping.items():
        if not isinstance(n, dict):
            continue
        parent = n.get("parent")
        if parent not in mapping:
            roots.append(k)
        else:
            children.setdefault(parent, []).append(k)

    def key(k):
        m = mapping[k].get("message") or {}
        t = m.get("create_time")
        return (t if isinstance(t, (float, int)) else 0, k)

    seen = set()
    ordered = []
    for root in sorted(roots, key=key) + list(mapping):
        stack = [root]
        while stack:
            k = stack.pop()
            if k in seen or not isinstance(mapping.get(k), dict):
                continue
            seen.add(k)
            ordered.append(k)
            stack.extend(reversed(sorted(children.get(k, []), key=key)))
    return ordered, active, status


def fragments(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [text for v in value for text in fragments(v)]
    if isinstance(value, dict):
        for key in ("text", "transcript", "transcription", "content", "parts"):
            if isinstance(value.get(key), (str, list, dict)):
                return fragments(value[key])
    return []


def normalize(message):
    """Yield (origin,text) blocks; do not pretend reasoning/tool calls are answers."""
    role = (message.get("author") or {}).get("role", "unknown")
    v = message.get("content") or {}
    meta = message.get("metadata") or {}
    if not isinstance(v, dict):
        v = {"parts": v}
    typ = v.get("content_type", "text")
    recipient = message.get("recipient")
    if role == "tool":
        yield "tool_output", "\n".join(fragments(v))
        return
    if role == "assistant" and recipient and recipient not in ("all", "assistant"):
        yield (
            "tool_input",
            json.dumps({"name": recipient, "content": v}, ensure_ascii=False),
        )
        return
    if typ == "thoughts":
        for item in v.get("thoughts", []):
            if not isinstance(item, dict):
                continue
            text = (
                item.get("content")
                or "\n".join(fragments(item.get("chunks")))
                or item.get("summary")
                or ""
            )
            if text:
                yield "thinking", text
        return
    if (
        typ == "reasoning_recap"
        or meta.get("is_reasoning")
        or meta.get("channel") in ("analysis", "commentary")
    ):
        yield "thinking", "\n".join(fragments(v))
        return
    origin = role if role in ("user", "assistant") else "context"
    if meta.get("is_visually_hidden_from_conversation"):
        origin = "context"
    text = "\n".join(fragments(v.get("parts", v.get("text", v.get("transcript", "")))))
    if text:
        yield origin, text


def pointers(value):
    if isinstance(value, dict):
        p = value.get("asset_pointer")
        if isinstance(p, str):
            yield p, value
        for v in value.values():
            yield from pointers(v)
    elif isinstance(value, list):
        for v in value:
            yield from pointers(v)


@lru_cache(maxsize=16)
def asset_catalog(folder):
    root = Path(folder)
    manifest = root / "conversation_asset_file_names.json"
    names = json.loads(manifest.read_text()) if manifest.exists() else {}
    library = root / "library_files.json"
    if library.exists():
        for f in json.loads(library.read_text()):
            if isinstance(f, dict) and isinstance(f.get("file_name"), str):
                for key in ("file_id", "id"):
                    if isinstance(f.get(key), str):
                        names.setdefault(f[key], f["file_name"])
    files = {p.name: p for p in root.rglob("*") if p.is_file()}
    return names, files


def resolve_asset(folder, pointer, name=None, mime=None):
    names, files = asset_catalog(str(folder))
    token = unquote(pointer.split("://")[-1]).split("?")[0].rsplit("/", 1)[-1]
    # Basename-only lookup confined to this extracted bundle; never fetch URLs.
    p = files.get(token) or files.get(token + ".dat")
    original = name or names.get(token) or names.get(token + ".dat")
    if not p and original:
        p = files.get(Path(original).name)
    if not p:
        matches = [v for k, v in files.items() if k.startswith(token + ".")]
        if len(matches) == 1:
            p = matches[0]
    if p and not mime:
        with p.open("rb") as f:
            magic = f.read(16)
        for prefix, typ in [
            (b"\x89PNG", "image/png"),
            (b"\xff\xd8\xff", "image/jpeg"),
            (b"%PDF", "application/pdf"),
            (b"GIF8", "image/gif"),
        ]:
            if magic.startswith(prefix):
                mime = typ
                break
    mime = mime or mimetypes.guess_type(original or "")[0]
    return {
        "pointer": pointer,
        "filename": original,
        "local_path": str(p.resolve()) if p else None,
        "mime_type": mime,
    }
