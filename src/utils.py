from pathlib import Path
import os

def ensure_output_dirs(outputs_path):
    outputs_path = Path(outputs_path)
    (outputs_path / "plots").mkdir(parents=True, exist_ok=True)
    (outputs_path / "tables").mkdir(parents=True, exist_ok=True)
