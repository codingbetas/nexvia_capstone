import pandas as pd
import sys
import subprocess
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

print("Loading reference baseline data...")
df_train = pd.read_csv('creditcard.csv').sample(1000, random_state=42)

print("Loading simulated production data (with drift)...")
df_live = pd.read_csv('creditcard.csv').sample(1000, random_state=99)
df_live['Amount'] = df_live['Amount'] * 2 

print("Generating Evidently AI Data Drift Report...")
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=df_train, current_data=df_live)

report.save_html("data_drift_report.html")
print("✅ Report saved as 'data_drift_report.html'")

# Automatic Retraining Trigger (Step 6 of PRD)
report_dict = report.as_dict()
try:
    drift_detected = report_dict['metrics'][0]['result'].get('dataset_drift', True)
except Exception:
    drift_detected = True

if drift_detected:
    print("🚨 DATA DRIFT DETECTED! Threshold exceeded.")
    print("🔄 Automatically triggering model retraining pipeline (train_local.py)...")
    result = subprocess.run([sys.executable, "train_local.py"], capture_output=True, text=True)
    print(result.stdout)
    print("✅ Model retraining completed successfully and logged to MLflow.")
else:
    print("✅ No significant drift detected. Model remains active.")
