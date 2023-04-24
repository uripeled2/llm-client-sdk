import json
import os
from pathlib import Path


def load_json_resource(path: str) -> dict:
    project_base_path = Path(__file__).parent.parent
    with open(os.path.join(f'{project_base_path}/resources/', path), "r") as f:
        return json.load(f)
