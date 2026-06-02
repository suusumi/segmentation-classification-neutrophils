"""Smoke-тест служебных модулей проекта."""

from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    """Запускает базовую сквозную проверку конфигурации, логирования, JSON и seed."""
    from src.utils.config import load_yaml_config
    from src.utils.io import ensure_dir, read_json, write_json
    from src.utils.logger import get_logger
    from src.utils.seed import set_seed

    project_root = PROJECT_ROOT
    config_path = project_root / "configs" / "smoke_test.yaml"
    config = load_yaml_config(config_path)

    set_seed(
        seed=int(config["experiment"]["seed"]),
        deterministic=bool(config["experiment"]["deterministic"]),
    )

    log_dir = ensure_dir(project_root / config["paths"]["log_dir"])
    log_path = log_dir / "smoke.log"
    json_path = project_root / config["paths"]["json_path"]

    logger = get_logger("smoke_test", log_file=log_path)
    logger.info("Loaded config for experiment '%s'.", config["experiment"]["name"])
    logger.info("%s", config["run"]["message"])

    payload = {
        "experiment": config["experiment"]["name"],
        "seed": config["experiment"]["seed"],
        "message": config["run"]["message"],
        "log_file": str(log_path),
    }
    write_json(json_path, payload)

    loaded_payload = read_json(json_path)
    assert loaded_payload == payload
    assert log_path.exists()

    log_text = log_path.read_text(encoding="utf-8")
    assert "Smoke test completed successfully." in log_text

    # Дополнительно проверяем сгенерированный JSON стандартной библиотекой.
    raw_json = json.loads(json_path.read_text(encoding="utf-8"))
    assert raw_json["experiment"] == "smoke-test"

    print("Smoke test passed.")
    print(f"YAML: {config_path}")
    print(f"Log: {log_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
