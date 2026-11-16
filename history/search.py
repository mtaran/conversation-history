"""Optional search CLI; the index builder itself needs no arguments."""

import argparse
import json

from .store import HistoryIndex


def main():
    p = argparse.ArgumentParser(description="Search the local conversation archive")
    p.add_argument("query")
    p.add_argument("--semantic", action="store_true")
    p.add_argument("--project")
    p.add_argument("--trace", action="store_true")
    p.add_argument("--limit", type=int, default=20)
    a = p.parse_args()
    index = HistoryIndex()
    try:
        if a.limit < 1:
            p.error("--limit must be positive")
        rows = (
            index.semantic(a.query, a.limit, a.project)
            if a.semantic
            else index.terms(a.query, a.limit, a.project, a.trace)
        )
        for r in rows:
            print(
                json.dumps(
                    {
                        k: r[k]
                        for k in [
                            "id",
                            "title",
                            "project",
                            "provider",
                            "score",
                            "origin",
                            "snippet",
                        ]
                    },
                    ensure_ascii=False,
                )
            )
    finally:
        index.close()


if __name__ == "__main__":
    main()
