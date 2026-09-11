"""traversal rejected, retention honored, locks deferred (unit)."""

import pytest

from src.workspace import CleanupResult, OutputDirError, Workspace


def test_job_dir_layout(tmp_path) -> None:
    ws = Workspace(tmp_path / "ws")
    job = ws.job_dir("job-1")
    assert job.is_dir()
    for sub in ("renders", "ocr", "md"):
        assert (job / sub).is_dir()


def test_job_id_traversal_rejected(tmp_path) -> None:
    ws = Workspace(tmp_path / "ws")
    with pytest.raises(OutputDirError):
        ws.job_dir("../escape")
    with pytest.raises(OutputDirError):
        ws.job_dir("..")


def test_output_dir_requires_absolute(tmp_path) -> None:
    with pytest.raises(OutputDirError, match="absolute"):
        Workspace.validate_output_dir("relative/path")


def test_output_dir_missing_unless_create(tmp_path) -> None:
    missing = tmp_path / "nope"
    with pytest.raises(OutputDirError, match="does not exist"):
        Workspace.validate_output_dir(missing)
    created = Workspace.validate_output_dir(missing, create=True)
    assert created.is_dir()


def test_output_dir_rejects_file(tmp_path) -> None:
    file_path = tmp_path / "f.txt"
    file_path.write_text("x", encoding="utf-8")
    with pytest.raises(OutputDirError, match="not a directory"):
        Workspace.validate_output_dir(file_path)


def test_cleanup_removes_by_default(tmp_path) -> None:
    ws = Workspace(tmp_path / "ws")
    job = ws.job_dir("job-1")
    (job / "renders" / "p.png").write_bytes(b"data")
    result = ws.cleanup("job-1", keep=False)
    assert result == CleanupResult(removed=True, deferred=False)
    assert not job.exists()


def test_cleanup_keep_retains(tmp_path) -> None:
    ws = Workspace(tmp_path / "ws")
    job = ws.job_dir("job-1")
    result = ws.cleanup("job-1", keep=True)
    assert result == CleanupResult(removed=False, deferred=False)
    assert job.exists()


def test_locked_file_defers_then_reclaims(tmp_path) -> None:
    """Windows holds deletion of open files: defer, job unaffected, reclaim later."""
    ws = Workspace(tmp_path / "ws", max_retries=1)
    job = ws.job_dir("job-1")
    locked = job / "renders" / "p.png"
    locked.write_bytes(b"data")
    with locked.open("rb"):
        result = ws.cleanup("job-1", keep=False)
    # Either the platform locked (deferred) or it removed; never an exception,
    # and the completed job's outcome never depends on cleanup.
    assert isinstance(result, CleanupResult)
    if result.deferred:
        assert job.exists()
        assert (tmp_path / "ws" / ".pending-cleanup" / "job-1.txt").exists()
        reclaimed = ws.cleanup_deferred()
        assert reclaimed == ["job-1"]
        assert not job.exists()
    else:
        assert result.removed and not job.exists()
