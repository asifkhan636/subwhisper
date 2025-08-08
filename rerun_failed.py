import csv
import json
from pathlib import Path
from typing import List

from experiment import SubtitleExperiment


def main() -> None:
    failed_path = Path("failed.csv")
    if not failed_path.exists():
        print("No failed episodes found")
        return

    with failed_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    remaining: List[dict] = []
    for row in rows:
        cfg = json.loads(row["config"])
        cfg["inputs"] = [row["file"]]
        cfg["run_id"] = f"{row['run_id']}_retry"
        exp = SubtitleExperiment(cfg)
        try:
            exp.run()
            exp.aggregate_results()
            exp.report()
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Rerun failed for {row['file']}: {exc}")
            remaining.append(row)

    if remaining:
        with failed_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=reader.fieldnames)
            writer.writeheader()
            writer.writerows(remaining)
    else:
        failed_path.unlink()


if __name__ == "__main__":  # pragma: no cover
    main()
