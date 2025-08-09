from pathlib import Path

def test_no_sync_flag_absent():
    cli_path = Path(__file__).resolve().parents[1] / "subwhisper_cli.py"
    content = cli_path.read_text()
    assert "--no-sync" not in content
    assert "no_sync" not in content
    assert "--skip-music" in content
