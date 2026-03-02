import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
STATE_FILE = BASE_DIR / "training_status.json"

def write_status(data):
    STATE_FILE.write_text(json.dumps(data), encoding="utf-8")

def read_status():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {
        "running": False,
        "epoch": 0,
        "total_epochs": 0,
        "progress": 0,
        "mensagem": ""
    }