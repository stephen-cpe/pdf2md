"""Workspace manager — per-job dirs, retention, validated output.

Layout (§6.3): <root>/jobs/{job_id}/{source.pdf,renders/,ocr/,md/}.
Windows hard DoD: cleanup retries PermissionError (open handle/Defender)
with exponential backoff up to MAX_CLEANUP_RETRIES, then defers to next
start via a marker file. Cleanup NEVER raises — a completed job stays
completed. Call cleanup_deferred() at application start.
"""

import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

_DEFERRED_DIRNAME = ".pending-cleanup"
_JOB_SUBDIRS = ("renders", "ocr", "md")


class OutputDirError(ValueError):
    """Output directory rejected (relative, missing, not a dir, unwritable)."""


@dataclass(frozen=True)
class CleanupResult:
    """removed=True when the tree is gone; deferred=True when parked for later."""

    removed: bool
    deferred: bool


class Workspace:
    """Rooted filesystem scope for job workspaces and deferred markers."""

    def __init__(self, root: Path, max_retries: int = 3) -> None:
        self._root = root.resolve()
        self._max_retries = max_retries
        self._root.mkdir(parents=True, exist_ok=True)

    def _scoped(self, *parts: str) -> Path:
        """Join under root; each part must be a single safe segment (NFR-5)."""
        for part in parts:
            if (
                not part
                or part in (".", "..")
                or "/" in part
                or "\\" in part
                or Path(part).is_absolute()
            ):
                raise OutputDirError(f"unsafe path segment: {part!r}")
        candidate = (self._root.joinpath(*parts)).resolve()
        if candidate != self._root and self._root not in candidate.parents:
            raise OutputDirError(f"path escapes workspace root: {parts!r}")
        return candidate

    def job_dir(self, job_id: str) -> Path:
        """Per-job dir with renders/ocr/md subdirs (creates on demand)."""
        directory = self._scoped("jobs", job_id)
        directory.mkdir(parents=True, exist_ok=True)
        for sub in _JOB_SUBDIRS:
            (directory / sub).mkdir(exist_ok=True)
        return directory

    @staticmethod
    def validate_output_dir(path: str | Path, create: bool = False) -> Path:
        """User output dir: absolute, a dir, writable (FR-JOB-2, NFR-5)."""
        candidate = Path(path)
        if not candidate.is_absolute():
            raise OutputDirError(f"output_dir must be absolute: {path!r}")
        resolved = candidate.resolve()
        if not resolved.exists():
            if not create:
                raise OutputDirError(f"output_dir does not exist: {resolved}")
            resolved.mkdir(parents=True, exist_ok=True)
        if not resolved.is_dir():
            raise OutputDirError(f"output_dir is not a directory: {resolved}")
        if not os.access(resolved, os.W_OK):
            raise OutputDirError(f"output_dir is not writable: {resolved}")
        return resolved

    def _remove_tree(self, target: Path) -> bool:
        """Best-effort rmtree with exponential backoff on Windows locks."""
        delay = 0.5
        for _ in range(self._max_retries):
            try:
                shutil.rmtree(target, ignore_errors=False)
                return True
            except PermissionError:
                time.sleep(delay)
                delay *= 2
        # Final attempt after the last backoff before deferring.
        try:
            shutil.rmtree(target, ignore_errors=False)
            return True
        except PermissionError:
            return False

    def _deferred_dir(self) -> Path:
        directory = self._root / _DEFERRED_DIRNAME
        directory.mkdir(exist_ok=True)
        return directory

    def cleanup(self, job_id: str, keep: bool) -> CleanupResult:
        """Apply retention: keep wins; else remove, deferring locked trees.

        Never raises for lock contention — check CleanupResult.deferred.
        """
        target = self._scoped("jobs", job_id)
        if keep or not target.exists():
            return CleanupResult(removed=False, deferred=False)
        if self._remove_tree(target):
            return CleanupResult(removed=True, deferred=False)
        (self._deferred_dir() / f"{job_id}.txt").write_text(str(target), encoding="utf-8")
        return CleanupResult(removed=False, deferred=True)

    def cleanup_deferred(self) -> list[str]:
        """Retry parked cleanups (call at startup); returns reclaimed job ids."""
        directory = self._root / _DEFERRED_DIRNAME
        if not directory.exists():
            return []
        reclaimed: list[str] = []
        for marker in sorted(directory.glob("*.txt")):
            target = Path(marker.read_text(encoding="utf-8"))
            try:
                scoped = target.resolve()
                if scoped != self._root and self._root not in scoped.parents:
                    continue
            except OSError:
                continue
            if not scoped.exists() or self._remove_tree(scoped):
                marker.unlink(missing_ok=True)
                reclaimed.append(marker.stem)
        return reclaimed


__all__ = ["CleanupResult", "OutputDirError", "Workspace"]
