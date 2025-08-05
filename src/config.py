from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "cleaned"
CHAR_MAP_PATH = BASE_DIR / "data" / "char_map.json"
MODEL_CHECKPOINT_DIR = BASE_DIR / "logs" / "checkpoints"
