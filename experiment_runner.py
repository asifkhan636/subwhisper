import argparse
import json
import shutil
from itertools import product
from pathlib import Path
from typing import Any, Dict

import yaml

from experiment import SubtitleExperiment


def _load_config(path: Path) -> Dict[str, Any]:
    text = path.read_text()
    if path.suffix in {".yml", ".yaml"}:
        return yaml.safe_load(text)
    return json.loads(text)


def _set_nested(cfg: Dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    target = cfg
    for part in parts[:-1]:
        target = target.setdefault(part, {})
    target[parts[-1]] = value


def _sweep_configs(base: Dict[str, Any], grid: Dict[str, Any]):
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    for combo in product(*values):
        params = dict(zip(keys, combo))
        cfg = json.loads(json.dumps(base))  # deep copy
        for k, v in params.items():
            _set_nested(cfg, k, v)
        yield cfg, params


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Subtitle experiments")
    parser.add_argument("config", help="Path to YAML/JSON configuration")
    parser.add_argument("--sweep", action="store_true", help="Run parameter sweep")
    parser.add_argument(
        "--keep-run-dir",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Retain run directory after successful completion",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = _load_config(cfg_path)

    base = {k: v for k, v in cfg.items() if k not in {"grid"}}
    grid = cfg.get("grid", {})

    if args.sweep and grid:
        for idx, (conf, params) in enumerate(_sweep_configs(base, grid), 1):
            label = "_".join(f"{k.split('.')[-1]}-{v}" for k, v in params.items())
            run_id = f"run{idx}_{label}"
            conf["run_id"] = run_id
            keep = conf.get("retain_run_dir", args.keep_run_dir)
            exp = SubtitleExperiment(conf)
            success = False
            try:
                exp.run()
                exp.aggregate_results()
                exp.report()
                success = not exp.failures
            finally:
                if success and not keep:
                    shutil.rmtree(exp.run_dir, ignore_errors=True)
    else:
        run_id = base.get("run_id")
        if not run_id:
            run_id = "run"
        base["run_id"] = run_id
        keep = base.get("retain_run_dir", args.keep_run_dir)
        exp = SubtitleExperiment(base)
        success = False
        try:
            exp.run()
            exp.aggregate_results()
            exp.report()
            success = not exp.failures
        finally:
            if success and not keep:
                shutil.rmtree(exp.run_dir, ignore_errors=True)


if __name__ == "__main__":  # pragma: no cover
    main()
