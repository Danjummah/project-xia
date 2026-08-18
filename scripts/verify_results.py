from pathlib import Path
import csv, math
p = Path(__file__).resolve().parents[1] / 'results/project_xia_locked_test_results.csv'
with p.open(newline='', encoding='utf-8') as f:
    rows = {r['method']: r for r in csv.DictReader(f)}
r = rows.get('CS_SHAP_19')
if r is None: raise SystemExit('FAIL: CS_SHAP_19 row missing')
expected = {'feature_count':19, 'accuracy':.97751, 'macro_f1':.96815, 'balanced_accuracy':.96426, 'mcc':.97553, 'training_seconds':37.236}
for col, target in expected.items():
    value = float(r[col])
    if not math.isclose(value, target, rel_tol=0, abs_tol=5e-5): raise SystemExit(f'FAIL: {col}={value}')
if not math.isclose(float(rows['Full_42']['training_seconds']), 51.5566, rel_tol=0, abs_tol=5e-4): raise SystemExit('FAIL: Full_42 time')
print('PASS: principal locked-test results match the manuscript values.')
