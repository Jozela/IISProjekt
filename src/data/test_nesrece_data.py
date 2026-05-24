import sys
import pandas as pd
from evidently import Report
from evidently.presets.dataset_stats import DataSummaryPreset
from evidently.presets.drift import DataDriftPreset
import os

CURRENT_PATH   = "data/preprocessed/nesrece_v_cestnem_prometu.csv"
REFERENCE_PATH = "data/reference/nesrece_v_cestnem_prometu.csv"
REPORT_PATH    = "reports/nesrece_data_testing_report.html"

current = pd.read_csv(CURRENT_PATH)

for col in ["prijavaCas", "nastanekCas"]:
    current[col] = pd.to_datetime(current[col], errors="coerce")

if not os.path.exists(REFERENCE_PATH):
    print(f"Reference file not found. Creating it at: {REFERENCE_PATH}")
    os.makedirs(os.path.dirname(REFERENCE_PATH), exist_ok=True)
    current.to_csv(REFERENCE_PATH, index=False)

reference = pd.read_csv(REFERENCE_PATH)
for col in ["prijavaCas", "nastanekCas"]:
    reference[col] = pd.to_datetime(reference[col], errors="coerce")

for df in [current, reference]:
    df["hour_of_day"]      = df["nastanekCas"].dt.hour
    df["day_of_week"]      = df["nastanekCas"].dt.dayofweek
    df["response_delay_s"] = (
        df["prijavaCas"] - df["nastanekCas"]
    ).dt.total_seconds()

cols_to_drop = ["prijavaCas", "nastanekCas"]
ref_analysis = reference.drop(columns=cols_to_drop)
cur_analysis = current.drop(columns=cols_to_drop)

report = Report(
    [DataSummaryPreset(), DataDriftPreset()],
    include_tests=True,
)
result = report.run(reference_data=ref_analysis, current_data=cur_analysis)

os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
result.save_html(REPORT_PATH)
print(f"Report saved to: {REPORT_PATH}")

all_tests_passed = True
result_dict = result.dict()

if "tests" in result_dict:
    for test in result_dict["tests"]:
        status = test.get("status", "")
        if status not in ("SUCCESS", "WARNING"):
            print(f"  FAILED  — {test.get('name', 'unknown test')}: {status}")
            all_tests_passed = False
        else:
            print(f"  {status} — {test.get('name', 'unknown test')}")

if not all_tests_passed:
    print("\nData tests FAILED. Reference data NOT updated.")
    sys.exit(1)
else:
    print("\nData tests PASSED. Promoting current data to reference.")
    os.remove(REFERENCE_PATH)
    pd.read_csv(CURRENT_PATH).to_csv(REFERENCE_PATH, index=False)
    sys.exit(0)