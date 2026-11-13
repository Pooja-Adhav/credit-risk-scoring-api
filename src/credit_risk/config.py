"""Central configuration: paths, column names and business rules."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_DATA_PATH = DATA_DIR / "credit_default.csv"
MODEL_PATH = MODELS_DIR / "model.joblib"

RANDOM_STATE = 42
TARGET = "default"

# Repayment status (-2 .. 8), bill statements and payments for the last six months.
# Month 1 is the most recent one.
MONTHS = range(1, 7)
PAY_STATUS_COLS = [f"PAY_{m}" for m in MONTHS]
BILL_COLS = [f"BILL_AMT{m}" for m in MONTHS]
PAY_AMT_COLS = [f"PAY_AMT{m}" for m in MONTHS]

RAW_COLUMNS = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    *PAY_STATUS_COLS, *BILL_COLS, *PAY_AMT_COLS,
]

# Protected attributes are kept for fairness monitoring only and are never model inputs.
PROTECTED_COLS = ["SEX", "MARRIAGE"]

# Business assumption: missing a defaulter costs 5x more than wrongly declining a good client.
COST_FALSE_NEGATIVE = 5.0
COST_FALSE_POSITIVE = 1.0

# Probability-of-default cut-offs for risk bands (upper bounds).
RISK_BANDS = [(0.10, "A"), (0.20, "B"), (0.35, "C"), (0.55, "D"), (1.01, "E")]
