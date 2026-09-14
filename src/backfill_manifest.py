"""One-time backfill of the existing data/processed/dataset_idle_mel{,_defective}
PNGs into the SQLite manifest (src/manifest.py) that src/train.py reads from.

Run manually once: `python -m src.backfill_manifest`. Never invoked by CI or any
other code path - the data it walks lives under the gitignored data/ directory
and won't exist in a fresh checkout.

infer_domain() is used here, once, to bootstrap each source file's domain from
its filename - the one legitimate remaining use of that heuristic for
populating the manifest. After this backfill, nothing at train/serve time calls
it to determine a manifest row's domain again.
"""

import argparse
import logging
import re
import sqlite3
from pathlib import Path
from typing import Dict, Optional, Tuple

from src.config import Config, load_config
from src.inference import infer_domain
from src.manifest import SourceFileMetadata, add_segment, get_connection, get_or_create_source_file, init_db

logger = logging.getLogger(__name__)

SEGMENT_FILENAME_RE = re.compile(r"^(?P<stem>.+)_idle_(?P<start>\d+\.\d+)s\.(?:png|jpg|jpeg)$", re.IGNORECASE)
RAW_AUDIO_SUFFIXES = (".m4a", ".wav")

LABEL_DIRS = {
    "dataset_idle_mel": "healthy",
    "dataset_idle_mel_defective": "defective",
}


def _resolve_raw_audio(raw_dir: Path, stem: str) -> Optional[Path]:
    for suffix in RAW_AUDIO_SUFFIXES:
        candidate = raw_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def backfill(cfg: Config, manifest_db_path: Path) -> Dict[str, int]:
    conn = get_connection(str(manifest_db_path))
    init_db(conn)

    raw_dir = cfg.resolve_path("data/raw")
    processed_dir = cfg.resolve_path("data/processed")

    source_files_registered = 0
    segments_inserted = 0
    segments_skipped = 0
    source_file_cache: Dict[str, Tuple[int, str]] = {}

    for dir_name, label in LABEL_DIRS.items():
        png_dir = processed_dir / dir_name
        if not png_dir.is_dir():
            logger.warning("Skipping missing directory: %s", png_dir)
            continue

        for png_path in sorted(png_dir.iterdir()):
            match = SEGMENT_FILENAME_RE.match(png_path.name)
            if not match:
                logger.warning("Skipping unparseable filename: %s", png_path.name)
                segments_skipped += 1
                continue

            stem = match.group("stem")
            start_time = float(match.group("start"))

            raw_audio_path = _resolve_raw_audio(raw_dir, stem)
            if raw_audio_path is None:
                logger.warning("Skipping %s: no raw audio found for stem '%s'", png_path.name, stem)
                segments_skipped += 1
                continue

            raw_relative_path = str(raw_audio_path.relative_to(cfg.root_dir))

            if raw_relative_path not in source_file_cache:
                domain = infer_domain(stem, cfg)
                original_codec = raw_audio_path.suffix.lower().lstrip(".")
                source_file_id = get_or_create_source_file(
                    conn, raw_relative_path, domain, metadata=SourceFileMetadata(original_codec=original_codec)
                )
                source_file_cache[raw_relative_path] = (source_file_id, domain)
                source_files_registered += 1

            source_file_id, domain = source_file_cache[raw_relative_path]
            png_relative_path = str(png_path.relative_to(cfg.root_dir))

            try:
                add_segment(
                    conn,
                    source_file_id=source_file_id,
                    start_time_seconds=start_time,
                    duration_seconds=cfg.audio.segment_duration,
                    label=label,
                    domain=domain,
                    rendered_png_path=png_relative_path,
                    approved=True,
                )
                segments_inserted += 1
            except sqlite3.IntegrityError:
                logger.warning("Skipping duplicate segment: %s", png_path.name)
                segments_skipped += 1

    conn.close()

    return {
        "source_files_registered": source_files_registered,
        "segments_inserted": segments_inserted,
        "segments_skipped": segments_skipped,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-db",
        default=None,
        help="Path to the manifest DB (default: data/manifest.db under the repo root).",
    )
    args = parser.parse_args()

    cfg = load_config()
    manifest_db_path = Path(args.manifest_db) if args.manifest_db else cfg.resolve_path("data/manifest.db")

    summary = backfill(cfg, manifest_db_path)
    print(
        f"Backfill complete: {summary['source_files_registered']} source files registered, "
        f"{summary['segments_inserted']} segments inserted, {summary['segments_skipped']} skipped."
    )


if __name__ == "__main__":
    main()
