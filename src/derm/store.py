"""SQLite-backed case store.

Analyses are persisted so the UI has a real history, and so aggregate statistics
(tier distribution, class distribution, mean confidence) can be computed over
everything the deployment has seen. Full-resolution renders are *not* stored -
only a small thumbnail plus the JSON payload with the image map stripped - which
keeps the database small enough to stay useful.

A connection is opened per call rather than shared. SQLite connections are not
safe to move between threads, and FastAPI will happily run handlers on different
threads from its worker pool.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .config import CASE_DB_PATH

_INIT_LOCK = threading.Lock()
_initialised: set[str] = set()

SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    id                     TEXT PRIMARY KEY,
    created_at             TEXT NOT NULL,
    filename               TEXT,
    top_code               TEXT NOT NULL,
    top_name               TEXT NOT NULL,
    confidence             REAL NOT NULL,
    tier                   TEXT NOT NULL,
    severity_score         REAL NOT NULL,
    malignancy_probability REAL NOT NULL,
    tds                    REAL,
    quality_score          REAL,
    requires_review        INTEGER NOT NULL DEFAULT 0,
    weights_status         TEXT,
    thumbnail              TEXT,
    payload                TEXT NOT NULL,
    notes                  TEXT
);
CREATE INDEX IF NOT EXISTS idx_cases_created_at ON cases(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_cases_tier       ON cases(tier);
CREATE INDEX IF NOT EXISTS idx_cases_top_code   ON cases(top_code);
"""


def _database_path(path: Path | None = None) -> Path:
    return Path(path) if path else CASE_DB_PATH


@contextmanager
def connect(path: Path | None = None) -> Iterator[sqlite3.Connection]:
    """Yield a configured connection, creating the schema on first use."""
    target = _database_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    key = str(target)
    if key not in _initialised:
        with _INIT_LOCK:
            if key not in _initialised:
                with sqlite3.connect(target) as setup:
                    setup.executescript(SCHEMA)
                _initialised.add(key)

    connection = sqlite3.connect(target, timeout=10.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def save_case(result: dict[str, Any], *, path: Path | None = None) -> str:
    """Persist an analysis result dictionary. Returns the case id."""
    payload = {k: v for k, v in result.items() if k != "images"}
    thumbnail = (result.get("images") or {}).get("original")

    severity = result.get("severity", {})
    prediction = result.get("prediction", {})
    morphology = result.get("morphology") or {}
    abcd = morphology.get("abcd") or {}

    row = (
        result["case_id"],
        result["created_at"],
        result.get("filename"),
        prediction.get("code", "unknown"),
        prediction.get("name", "unknown"),
        float(prediction.get("confidence", 0.0)),
        severity.get("tier", "INDETERMINATE"),
        float(severity.get("score", 0.0)),
        float(severity.get("malignancy_probability", 0.0)),
        float(abcd["tds"]) if "tds" in abcd else None,
        float((result.get("quality") or {}).get("score", 0.0)),
        1 if severity.get("requires_human_review") else 0,
        (result.get("model") or {}).get("weights_status"),
        thumbnail,
        json.dumps(payload, separators=(",", ":")),
        None,
    )

    with connect(path) as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO cases (
                id, created_at, filename, top_code, top_name, confidence, tier,
                severity_score, malignancy_probability, tds, quality_score,
                requires_review, weights_status, thumbnail, payload, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            row,
        )
    return result["case_id"]


def list_cases(
    *,
    limit: int = 50,
    offset: int = 0,
    tier: str | None = None,
    code: str | None = None,
    review_only: bool = False,
    path: Path | None = None,
) -> dict[str, Any]:
    """Return a page of case summaries, newest first."""
    clauses: list[str] = []
    params: list[Any] = []
    if tier:
        clauses.append("tier = ?")
        params.append(tier)
    if code:
        clauses.append("top_code = ?")
        params.append(code)
    if review_only:
        clauses.append("requires_review = 1")
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

    limit = max(1, min(int(limit), 200))
    offset = max(0, int(offset))

    with connect(path) as connection:
        total = connection.execute(
            f"SELECT COUNT(*) AS n FROM cases {where}", params
        ).fetchone()["n"]
        rows = connection.execute(
            f"""
            SELECT id, created_at, filename, top_code, top_name, confidence, tier,
                   severity_score, malignancy_probability, tds, quality_score,
                   requires_review, weights_status, thumbnail, notes
            FROM cases {where}
            ORDER BY created_at DESC, id DESC
            LIMIT ? OFFSET ?
            """,
            [*params, limit, offset],
        ).fetchall()

    return {
        "total": int(total),
        "limit": limit,
        "offset": offset,
        "items": [
            {
                **{k: row[k] for k in row.keys() if k != "requires_review"},
                "requires_review": bool(row["requires_review"]),
            }
            for row in rows
        ],
    }


def get_case(case_id: str, *, path: Path | None = None) -> dict[str, Any] | None:
    """Return the full stored payload for one case, or ``None``."""
    with connect(path) as connection:
        row = connection.execute(
            "SELECT payload, thumbnail, notes FROM cases WHERE id = ?", (case_id,)
        ).fetchone()
    if row is None:
        return None
    payload = json.loads(row["payload"])
    payload["images"] = {"original": row["thumbnail"]} if row["thumbnail"] else {}
    payload["notes"] = row["notes"]
    return payload


def set_notes(case_id: str, notes: str | None, *, path: Path | None = None) -> bool:
    """Attach or clear a free-text clinician note. Returns True if a row changed."""
    with connect(path) as connection:
        cursor = connection.execute(
            "UPDATE cases SET notes = ? WHERE id = ?", (notes, case_id)
        )
        return cursor.rowcount > 0


def delete_case(case_id: str, *, path: Path | None = None) -> bool:
    """Delete one case. Returns True if it existed."""
    with connect(path) as connection:
        cursor = connection.execute("DELETE FROM cases WHERE id = ?", (case_id,))
        return cursor.rowcount > 0


def clear_cases(*, path: Path | None = None) -> int:
    """Delete every case. Returns the number removed."""
    with connect(path) as connection:
        cursor = connection.execute("DELETE FROM cases")
        return cursor.rowcount


def stats(*, path: Path | None = None) -> dict[str, Any]:
    """Aggregate statistics across all stored cases."""
    with connect(path) as connection:
        totals = connection.execute(
            """
            SELECT COUNT(*)                AS total,
                   AVG(confidence)         AS mean_confidence,
                   AVG(severity_score)     AS mean_severity,
                   AVG(quality_score)      AS mean_quality,
                   SUM(requires_review)    AS flagged,
                   MIN(created_at)         AS first_case,
                   MAX(created_at)         AS last_case
            FROM cases
            """
        ).fetchone()
        by_tier = connection.execute(
            "SELECT tier, COUNT(*) AS n FROM cases GROUP BY tier ORDER BY n DESC"
        ).fetchall()
        by_class = connection.execute(
            "SELECT top_code, top_name, COUNT(*) AS n FROM cases "
            "GROUP BY top_code, top_name ORDER BY n DESC"
        ).fetchall()

    total = int(totals["total"] or 0)
    return {
        "total": total,
        "flagged_for_review": int(totals["flagged"] or 0),
        "mean_confidence": round(float(totals["mean_confidence"] or 0.0), 4),
        "mean_severity_score": round(float(totals["mean_severity"] or 0.0), 1),
        "mean_quality_score": round(float(totals["mean_quality"] or 0.0), 1),
        "first_case": totals["first_case"],
        "last_case": totals["last_case"],
        "by_tier": [{"tier": r["tier"], "count": r["n"]} for r in by_tier],
        "by_class": [
            {"code": r["top_code"], "name": r["top_name"], "count": r["n"]}
            for r in by_class
        ],
    }


__all__ = [
    "clear_cases",
    "connect",
    "delete_case",
    "get_case",
    "list_cases",
    "save_case",
    "set_notes",
    "stats",
]
