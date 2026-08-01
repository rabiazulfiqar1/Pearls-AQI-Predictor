import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.logger import get_logger

logger = get_logger(__name__)

LOCAL_CSV_PATH = Path("data/processed/karachi_backfill_engineered.csv")

engineered = pd.read_csv(LOCAL_CSV_PATH)
engineered["timestamp"] = pd.to_datetime(engineered["timestamp"], utc=True)
print(engineered.dtypes)
