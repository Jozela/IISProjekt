import sys
import pandas as pd
from evidently import Report
from evidently.presets.dataset_stats import DataSummaryPreset
from evidently.presets.drift import DataDriftPreset
import os

# ── Paths ────────────────────────────────────────────────────────────────────
CURRENT_PATH   = "data/preprocessed/nesrece_v_cestnem_prometu.csv"
REFERENCE_PATH = "data/reference/nesrece_v_cestnem_prometu.csv"
REPORT_PATH    = "reports/nesrece_data_testing_report.html"

# ── Load current data ────────────────────────────────────────────────────────
current = pd.read_csv(CURRENT_PATH)

# Parse datetime columns so Evidently treats them correctly
for col in ["prijavaCas", "nastanekCas"]:
    current[col] = pd.to_datetime(current[col], errors="coerce")

# ── Bootstrap reference if missing ──────────────────────────────────────────
if not os.path.exists(REFERENCE_PATH):
    print(f"Reference file not found. Creating it at: {REFERENCE_PATH}")
    os.makedirs(os.path.dirname(REFERENCE_PATH), exist_ok=True)
    current.to_csv(REFERENCE_PATH, index=False)

reference = pd.read_csv(REFERENCE_PATH)
for col in ["prijavaCas", "nastanekCas"]:
    reference[col] = pd.to_datetime(reference[col], errors="coerce")

# ── Feature engineering (optional but useful for drift detection) ────────────
# Evidently detects drift better on numeric/categorical columns than raw timestamps.
# Adding derived features gives it more signal.
for df in [current, reference]:
    df["hour_of_day"]    = df["nastanekCas"].dt.hour
    df["day_of_week"]    = df["nastanekCas"].dt.dayofweek   # 0=Mon … 6=Sun
    df["response_delay_s"] = (
        df["prijavaCas"] - df["nastanekCas"]
    ).dt.total_seconds()

# Drop raw datetime columns – Evidently can't drift-test them directly
cols_to_drop = ["prijavaCas", "nastanekCas"]
ref_analysis = reference.drop(columns=cols_to_drop)
cur_analysis = current.drop(columns=cols_to_drop)

# ── Build & run report ───────────────────────────────────────────────────────
report = Report(
    [
        DataSummaryPreset(),
        DataDriftPreset(),
    ],
    include_tests=True,
)

result = report.run(reference_data=ref_analysis, current_data=cur_analysis)

# ── Save HTML report ─────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
result.save_html(REPORT_PATH)
print(f"Report saved to: {REPORT_PATH}")

# ── Check test results ───────────────────────────────────────────────────────
all_tests_passed = True
result_dict = result.dict()

if "tests" in result_dict:
    for test in result_dict["tests"]:
        status = test.get("status", "")
        if status not in ("SUCCESS", "WARNING"):   # treat WARNING as passing
            print(f"  ✗ FAILED  — {test.get('name', 'unknown test')}: {status}")
            all_tests_passed = False
        else:
            print(f"  ✓ {status:<7} — {test.get('name', 'unknown test')}")

# ── Exit & optionally promote current → reference ────────────────────────────
if not all_tests_passed:
    print("\nData tests FAILED. Reference data NOT updated.")
    sys.exit(1)
else:
    print("\nData tests PASSED. Promoting current data to reference.")
    os.remove(REFERENCE_PATH)
    # Re-read raw current (before derived columns) so reference stays clean
    pd.read_csv(CURRENT_PATH).to_csv(REFERENCE_PATH, index=False)
    sys.exit(0)