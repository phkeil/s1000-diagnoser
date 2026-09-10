import pickle
import sqlite3

import pytest

from src.manifest import (
    ManifestSegment,
    add_segment,
    assign_pending_splits,
    bulk_add_segments,
    get_connection,
    get_or_create_source_file,
    get_training_segment_objects,
    get_training_segments,
    init_db,
    reset_all_splits,
    set_split,
)


@pytest.fixture
def conn(tmp_path):
    connection = get_connection(str(tmp_path / "manifest.db"))
    init_db(connection)
    yield connection
    connection.close()


def _make_source_file(conn, path="data/raw/example.m4a", domain="YouTube"):
    return get_or_create_source_file(conn, path, domain)


def _seed_healthy_segments(conn, domain, count, source_file_path):
    source_file_id = _make_source_file(conn, path=source_file_path, domain=domain)
    return [
        add_segment(
            conn,
            source_file_id,
            start_time_seconds=i * 0.25,
            duration_seconds=1.0,
            label="healthy",
            domain=domain,
            approved=True,
        )
        for i in range(count)
    ]


def test_init_db_is_idempotent(conn):
    init_db(conn)  # second call must not raise

    tables = {
        row["name"] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    assert {"source_files", "segments"} <= tables


def test_get_or_create_source_file_returns_same_id_on_second_call(conn):
    first_id = _make_source_file(conn)
    second_id = get_or_create_source_file(conn, "data/raw/example.m4a", "YouTube")

    assert first_id == second_id


def test_add_segment_raises_on_duplicate_start_time(conn):
    source_file_id = _make_source_file(conn)
    add_segment(conn, source_file_id, 0.0, 1.0, "healthy", "YouTube")

    with pytest.raises(sqlite3.IntegrityError):
        add_segment(conn, source_file_id, 0.0, 1.0, "healthy", "YouTube")


def test_add_segment_rejects_invalid_label(conn):
    source_file_id = _make_source_file(conn)

    with pytest.raises(sqlite3.IntegrityError):
        add_segment(conn, source_file_id, 0.0, 1.0, "not-a-real-label", "YouTube")


def test_set_split_rejects_unapproved_segment(conn):
    source_file_id = _make_source_file(conn)
    segment_id = add_segment(conn, source_file_id, 0.0, 1.0, "healthy", "YouTube", approved=False)

    with pytest.raises(sqlite3.IntegrityError):
        set_split(conn, segment_id, "train")


def test_bulk_add_segments_inserts_all_rows_in_one_transaction(conn):
    source_file_id = _make_source_file(conn)
    rows = [
        {
            "source_file_id": source_file_id,
            "start_time_seconds": i * 0.25,
            "duration_seconds": 1.0,
            "label": "healthy",
            "domain": "YouTube",
        }
        for i in range(3)
    ]

    ids = bulk_add_segments(conn, rows)

    assert len(ids) == 3
    count = conn.execute("SELECT COUNT(*) AS n FROM segments").fetchone()["n"]
    assert count == 3


def test_assign_pending_splits_only_touches_approved_healthy_unassigned(conn):
    source_file_id = _make_source_file(conn)
    healthy_id = add_segment(conn, source_file_id, 0.0, 1.0, "healthy", "YouTube", approved=True)
    defective_id = add_segment(conn, source_file_id, 1.0, 1.0, "defective", "YouTube", approved=True)
    unapproved_id = add_segment(conn, source_file_id, 2.0, 1.0, "healthy", "YouTube", approved=False)

    assign_pending_splits(conn, val_fraction=0.5, seed=1)

    def split_of(segment_id):
        return conn.execute("SELECT split FROM segments WHERE id = ?", (segment_id,)).fetchone()["split"]

    assert split_of(healthy_id) in ("train", "val")
    assert split_of(defective_id) is None
    assert split_of(unapproved_id) is None


def test_assign_pending_splits_is_idempotent(conn):
    _seed_healthy_segments(conn, "YouTube", 10, "data/raw/yt.m4a")

    first = assign_pending_splits(conn, val_fraction=0.2, seed=1)
    second = assign_pending_splits(conn, val_fraction=0.2, seed=1)

    assert first["newly_assigned"] == 10
    assert second["newly_assigned"] == 0


def test_assign_pending_splits_is_deterministic_given_a_fixed_seed(conn):
    _seed_healthy_segments(conn, "YouTube", 20, "data/raw/yt.m4a")

    assign_pending_splits(conn, val_fraction=0.3, seed=7)
    first_assignment = {row["id"]: row["split"] for row in conn.execute("SELECT id, split FROM segments")}

    reset_all_splits(conn)
    assign_pending_splits(conn, val_fraction=0.3, seed=7)
    second_assignment = {row["id"]: row["split"] for row in conn.execute("SELECT id, split FROM segments")}

    assert first_assignment == second_assignment


def test_assign_pending_splits_stratifies_by_domain(conn):
    _seed_healthy_segments(conn, "Garage", 10, "data/raw/garage.m4a")
    _seed_healthy_segments(conn, "YouTube", 10, "data/raw/yt.m4a")

    assign_pending_splits(conn, val_fraction=0.2, seed=3)

    garage_val = conn.execute(
        "SELECT COUNT(*) AS n FROM segments WHERE domain = 'Garage' AND split = 'val'"
    ).fetchone()["n"]
    youtube_val = conn.execute(
        "SELECT COUNT(*) AS n FROM segments WHERE domain = 'YouTube' AND split = 'val'"
    ).fetchone()["n"]

    assert garage_val > 0
    assert youtube_val > 0


def test_reset_all_splits_clears_every_split_and_returns_count(conn):
    _seed_healthy_segments(conn, "YouTube", 6, "data/raw/yt.m4a")
    assign_pending_splits(conn, val_fraction=0.5, seed=1)

    affected = reset_all_splits(conn)

    remaining = conn.execute("SELECT COUNT(*) AS n FROM segments WHERE split IS NOT NULL").fetchone()["n"]
    assert affected == 6
    assert remaining == 0


def test_get_training_segments_filters_by_split_and_label(conn):
    _seed_healthy_segments(conn, "YouTube", 8, "data/raw/yt.m4a")
    assign_pending_splits(conn, val_fraction=0.25, seed=2)

    val_rows = get_training_segments(conn, split="val")
    train_rows = get_training_segments(conn, split="train")

    assert len(val_rows) + len(train_rows) == 8
    assert all(row["split"] == "val" for row in val_rows)
    assert all(row["split"] == "train" for row in train_rows)


def test_get_training_segment_objects_returns_manifest_segment_dataclasses(conn):
    _seed_healthy_segments(conn, "YouTube", 4, "data/raw/yt.m4a")
    assign_pending_splits(conn, val_fraction=0.5, seed=1)

    train_rows = get_training_segment_objects(conn, split="train")

    assert len(train_rows) > 0
    assert all(isinstance(row, ManifestSegment) for row in train_rows)
    assert all(row.source_file_path == "data/raw/yt.m4a" for row in train_rows)


def test_manifest_segment_is_picklable():
    segment = ManifestSegment(
        id=1,
        source_file_id=1,
        source_file_path="data/raw/yt.m4a",
        start_time_seconds=0.0,
        duration_seconds=1.0,
        label="healthy",
        domain="YouTube",
        rendered_png_path=None,
    )

    restored = pickle.loads(pickle.dumps(segment))

    assert restored == segment
