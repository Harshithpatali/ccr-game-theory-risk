from pathlib import Path
BASE_DIR=Path(__file__).resolve().parents[1]
DATA_DIR=BASE_DIR/"data"
DAILY_FILE=DATA_DIR/"daily_risk.csv"
TRADES_FILE=DATA_DIR/"trades.csv"
ALPHA=1.4
