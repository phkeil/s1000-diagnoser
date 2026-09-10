"""SQLite manifest: the source of truth for every training audio segment.

Replaces a hand-curated directory of PNGs as src/train.py's data source. Tracks,
per segment: which source file and offset it comes from, its label, domain,
review/approval state, and train/val split assignment. A segment's
rendered_png_path is an optional cache (see ManifestDataset in src/data.py) -
the source file + start_time_seconds is what actually identifies a segment.

No ORM - plain stdlib sqlite3, matching this project's lightweight-dependencies
philosophy elsewhere (see requirements-api.txt's own comments).
"""

import logging
import math
import random
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# v2 added source_files' descriptive metadata columns (contributor,
# recording_device, ... - see SourceFileMetadata below). No migration system
# exists yet (see module docstring's "no ORM" note) - safe to change the
# CREATE TABLE directly rather than write an ALTER TABLE migration, since no
# schema-v1 database has ever been populated outside of tests.
SCHEMA_VERSION = 2

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS source_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT NOT NULL UNIQUE,
    source_url TEXT,
    domain TEXT NOT NULL CHECK(domain IN ('Garage','YouTube')),
    sample_rate INTEGER,
    duration_seconds REAL,
    contributor TEXT,
    recording_device TEXT,
    original_codec TEXT,
    exhaust_system TEXT,
    model_year INTEGER,
    kilometers_on_bike REAL,
    oil_type TEXT,
    kilometers_since_last_oilchange REAL,
    known_issues TEXT,
    notes TEXT,
    added_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS segments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_file_id INTEGER NOT NULL REFERENCES source_files(id) ON DELETE CASCADE,
    start_time_seconds REAL NOT NULL,
    duration_seconds REAL NOT NULL,
    rendered_png_path TEXT,
    render_config_hash TEXT,
    label TEXT NOT NULL DEFAULT 'unknown' CHECK(label IN ('healthy','defective','unknown')),
    domain TEXT NOT NULL CHECK(domain IN ('Garage','YouTube')),
    approved INTEGER NOT NULL DEFAULT 0 CHECK(approved IN (0,1)),
    split TEXT CHECK(split IN ('train','val')),
    split_assigned_at TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    CHECK((split IS NULL) OR (approved = 1)),
    UNIQUE(source_file_id, start_time_seconds)
);

CREATE INDEX IF NOT EXISTS idx_segments_source_file ON segments(source_file_id);
CREATE INDEX IF NOT EXISTS idx_segments_training ON segments(label, approved, split);
CREATE INDEX IF NOT EXISTS idx_segments_domain ON segments(domain);
"""


@dataclass
class ManifestSegment:
    """A training-ready segment: enough to render/load it without a live DB
    connection (plain dataclass, not a sqlite3.Row - must survive pickling
    across DataLoader(num_workers>0) worker processes)."""

    id: int
    source_file_id: int
    source_file_path: str
    start_time_seconds: float
    duration_seconds: float
    label: str
    domain: str
    rendered_png_path: Optional[str]


@dataclass
class SourceFileMetadata:
    """Optional descriptive metadata about a source recording - who
    contributed it and what state the bike/consumables were in, plus (for a
    manual upload) the container format it originally arrived in.

    Every field is nullable by design: a crawler-downloaded YouTube clip
    (Phase 3) isn't the crawler operator's own bike, so only original_codec
    (and maybe notes, e.g. the source video's title) will ever be populated
    for those rows - contributor/exhaust_system/oil_type/etc. simply stay
    NULL rather than being guessed at.

    exhaust_system is NULL for a bike's original/stock exhaust; a non-NULL
    value names the aftermarket system fitted instead.
    """

    contributor: Optional[str] = None
    recording_device: Optional[str] = None
    original_codec: Optional[str] = None
    exhaust_system: Optional[str] = None
    model_year: Optional[int] = None
    kilometers_on_bike: Optional[float] = None
    oil_type: Optional[str] = None
    kilometers_since_last_oilchange: Optional[float] = None
    known_issues: Optional[str] = None
    notes: Optional[str] = None


def get_connection(db_path: str) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")

    current_version = conn.execute("PRAGMA user_version").fetchone()[0]
    if current_version not in (0, SCHEMA_VERSION):
        logger.warning(
            "manifest.db schema version %d does not match expected %d - "
            "this code may not handle it correctly.",
            current_version,
            SCHEMA_VERSION,
        )
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Idempotent - safe to call on every connection open."""
    conn.executescript(_SCHEMA_SQL)
    if conn.execute("PRAGMA user_version").fetchone()[0] == 0:
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    conn.commit()


def get_or_create_source_file(
    conn: sqlite3.Connection,
    file_path: str,
    domain: str,
    source_url: Optional[str] = None,
    sample_rate: Optional[int] = None,
    duration_seconds: Optional[float] = None,
    metadata: Optional[SourceFileMetadata] = None,
) -> int:
    """metadata is only ever consulted the first time file_path is seen - an
    existing row's descriptive metadata isn't updated on a later call, same
    as its source_url/sample_rate/duration_seconds today."""
    existing = conn.execute(
        "SELECT id FROM source_files WHERE file_path = ?", (file_path,)
    ).fetchone()
    if existing is not None:
        return existing["id"]

    metadata = metadata or SourceFileMetadata()
    cur = conn.execute(
        """
        INSERT INTO source_files (
            file_path, source_url, domain, sample_rate, duration_seconds,
            contributor, recording_device, original_codec, exhaust_system,
            model_year, kilometers_on_bike, oil_type, kilometers_since_last_oilchange,
            known_issues, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            file_path,
            source_url,
            domain,
            sample_rate,
            duration_seconds,
            metadata.contributor,
            metadata.recording_device,
            metadata.original_codec,
            metadata.exhaust_system,
            metadata.model_year,
            metadata.kilometers_on_bike,
            metadata.oil_type,
            metadata.kilometers_since_last_oilchange,
            metadata.known_issues,
            metadata.notes,
        ),
    )
    conn.commit()
    return cur.lastrowid


def add_segment(
    conn: sqlite3.Connection,
    source_file_id: int,
    start_time_seconds: float,
    duration_seconds: float,
    label: str,
    domain: str,
    rendered_png_path: Optional[str] = None,
    approved: bool = False,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO segments (
            source_file_id, start_time_seconds, duration_seconds,
            rendered_png_path, label, domain, approved
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (source_file_id, start_time_seconds, duration_seconds, rendered_png_path, label, domain, int(approved)),
    )
    conn.commit()
    return cur.lastrowid


def bulk_add_segments(conn: sqlite3.Connection, rows: List[dict]) -> List[int]:
    """Same fields as add_segment, applied in one transaction."""
    ids = []
    with conn:
        for row in rows:
            cur = conn.execute(
                """
                INSERT INTO segments (
                    source_file_id, start_time_seconds, duration_seconds,
                    rendered_png_path, label, domain, approved
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["source_file_id"],
                    row["start_time_seconds"],
                    row["duration_seconds"],
                    row.get("rendered_png_path"),
                    row["label"],
                    row["domain"],
                    int(row.get("approved", False)),
                ),
            )
            ids.append(cur.lastrowid)
    return ids


def set_label(conn: sqlite3.Connection, segment_id: int, label: str) -> None:
    conn.execute(
        "UPDATE segments SET label = ?, updated_at = datetime('now') WHERE id = ?",
        (label, segment_id),
    )
    conn.commit()


def set_approved(conn: sqlite3.Connection, segment_id: int, approved: bool) -> None:
    conn.execute(
        "UPDATE segments SET approved = ?, updated_at = datetime('now') WHERE id = ?",
        (int(approved), segment_id),
    )
    conn.commit()


def set_split(conn: sqlite3.Connection, segment_id: int, split: str) -> None:
    conn.execute(
        "UPDATE segments SET split = ?, split_assigned_at = datetime('now'), updated_at = datetime('now') WHERE id = ?",
        (split, segment_id),
    )
    conn.commit()


def assign_pending_splits(
    conn: sqlite3.Connection, val_fraction: float = 0.15, seed: int = 42
) -> Dict[str, int]:
    """Assigns train/val to approved healthy segments that don't have a split
    yet - stratified per domain, deterministic given a fixed seed, and
    idempotent (a row keeps its split forever once assigned; call
    reset_all_splits() first to force a full re-split)."""
    pending = conn.execute(
        """
        SELECT id, domain FROM segments
        WHERE approved = 1 AND label = 'healthy' AND split IS NULL
        ORDER BY domain, id
        """
    ).fetchall()

    pending_by_domain: Dict[str, List[int]] = {}
    for row in pending:
        pending_by_domain.setdefault(row["domain"], []).append(row["id"])

    train_count = 0
    val_count = 0
    for domain, ids in pending_by_domain.items():
        shuffled = list(ids)
        random.Random(seed).shuffle(shuffled)

        n_val = math.ceil(len(shuffled) * val_fraction)
        val_ids = shuffled[len(shuffled) - n_val :] if n_val else []
        train_ids = shuffled[: len(shuffled) - n_val]

        if val_ids:
            conn.executemany(
                "UPDATE segments SET split = 'val', split_assigned_at = datetime('now'), "
                "updated_at = datetime('now') WHERE id = ?",
                [(seg_id,) for seg_id in val_ids],
            )
        if train_ids:
            conn.executemany(
                "UPDATE segments SET split = 'train', split_assigned_at = datetime('now'), "
                "updated_at = datetime('now') WHERE id = ?",
                [(seg_id,) for seg_id in train_ids],
            )
        train_count += len(train_ids)
        val_count += len(val_ids)

    conn.commit()
    return {"train": train_count, "val": val_count, "newly_assigned": train_count + val_count}


def get_training_segments(conn: sqlite3.Connection, split: str, label: str = "healthy") -> List[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM segments WHERE approved = 1 AND label = ? AND split = ? ORDER BY id",
        (label, split),
    ).fetchall()


def get_training_segment_objects(
    conn: sqlite3.Connection, split: str, label: str = "healthy"
) -> List[ManifestSegment]:
    rows = conn.execute(
        """
        SELECT s.id, s.source_file_id, sf.file_path AS source_file_path,
               s.start_time_seconds, s.duration_seconds, s.label, s.domain,
               s.rendered_png_path
        FROM segments s
        JOIN source_files sf ON sf.id = s.source_file_id
        WHERE s.approved = 1 AND s.label = ? AND s.split = ?
        ORDER BY s.id
        """,
        (label, split),
    ).fetchall()

    return [
        ManifestSegment(
            id=row["id"],
            source_file_id=row["source_file_id"],
            source_file_path=row["source_file_path"],
            start_time_seconds=row["start_time_seconds"],
            duration_seconds=row["duration_seconds"],
            label=row["label"],
            domain=row["domain"],
            rendered_png_path=row["rendered_png_path"],
        )
        for row in rows
    ]


def reset_all_splits(conn: sqlite3.Connection) -> int:
    """Explicit escape hatch - clears split (never label/approved) on every
    segment. Never called automatically; a real re-split decision."""
    cur = conn.execute(
        "UPDATE segments SET split = NULL, split_assigned_at = NULL, updated_at = datetime('now') "
        "WHERE split IS NOT NULL"
    )
    conn.commit()
    return cur.rowcount
