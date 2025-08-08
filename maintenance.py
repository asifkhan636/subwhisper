import argparse
import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_PATTERNS = ("*.wav", "*.mp3", "*.json", "*.npy", "*.pt")


def prune_raw_intermediates(
    output_root: str = "runs",
    patterns: Iterable[str] = DEFAULT_PATTERNS,
    dry_run: bool = True,
) -> int:
    """Remove intermediate files from ``output_root``.

    Parameters
    ----------
    output_root:
        Root directory containing run outputs.
    patterns:
        Glob patterns of files to remove.
    dry_run:
        When ``True``, only log planned removals.

    Returns
    -------
    int
        Number of files removed or that would be removed.
    """
    root = Path(output_root)
    count = 0
    for pattern in patterns:
        for path in root.rglob(pattern):
            if dry_run:
                logger.info("Would remove %s", path)
            else:
                try:
                    path.unlink()
                except FileNotFoundError:
                    continue
            count += 1
    return count


def compress_old_runs(
    output_root: str = "runs",
    days: int = 30,
    dry_run: bool = True,
) -> int:
    """Archive run directories older than ``days`` days.

    Parameters
    ----------
    output_root:
        Root directory containing run subdirectories.
    days:
        Age threshold in days for compression.
    dry_run:
        When ``True``, only log actions without modifying files.

    Returns
    -------
    int
        Number of run directories archived or that would be archived.
    """
    root = Path(output_root)
    if not root.exists():
        logger.warning("Output root %s does not exist", root)
        return 0
    threshold = datetime.now() - timedelta(days=days)
    count = 0
    for run_dir in root.iterdir():
        if not run_dir.is_dir():
            continue
        mtime = datetime.fromtimestamp(run_dir.stat().st_mtime)
        if mtime > threshold:
            continue
        archive = run_dir.with_suffix(".tar.gz")
        logger.info("Compressing %s to %s", run_dir, archive)
        if dry_run:
            count += 1
            continue
        shutil.make_archive(str(run_dir), "gztar", root_dir=run_dir)
        shutil.rmtree(run_dir)
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compress old run outputs and prune intermediate files"
    )
    parser.add_argument(
        "--output-root",
        default="runs",
        help="Root directory containing run outputs",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Compress runs older than this many days",
    )
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=list(DEFAULT_PATTERNS),
        help="Glob patterns of intermediate files to remove",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview actions without modifying files",
    )
    group.add_argument(
        "--delete",
        action="store_true",
        help="Delete files after compression and pruning",
    )
    args = parser.parse_args()
    dry_run = not args.delete

    prune_raw_intermediates(args.output_root, args.patterns, dry_run=dry_run)
    compress_old_runs(args.output_root, args.days, dry_run=dry_run)


if __name__ == "__main__":  # pragma: no cover
    main()
