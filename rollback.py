import argparse
import shutil
from pathlib import Path


def restore(run_id: str, output_root: str = "runs", dest: str = "restored") -> None:
    """Copy SRT outputs from a previous run into ``dest``.

    Parameters
    ----------
    run_id:
        Identifier of the run to restore from.
    output_root:
        Root directory containing run subdirectories.
    dest:
        Destination directory where files will be copied.
    """
    run_dir = Path(output_root) / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory {run_dir} does not exist")

    dest_dir = Path(dest)
    count = 0
    for srt in run_dir.glob("**/*.srt"):
        rel = srt.relative_to(run_dir)
        target = dest_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(srt, target)
        count += 1
    print(f"Restored {count} files to {dest_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Restore outputs from a previous run")
    parser.add_argument("--run-id", required=True, help="Run identifier to restore")
    parser.add_argument("--output-root", default="runs", help="Root directory containing runs")
    parser.add_argument("--dest", default="restored", help="Directory to copy outputs into")
    args = parser.parse_args()
    restore(args.run_id, args.output_root, args.dest)


if __name__ == "__main__":  # pragma: no cover
    main()
