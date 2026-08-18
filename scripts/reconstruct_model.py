from pathlib import Path
import hashlib
import zipfile

root = Path(__file__).resolve().parents[1]
models = root / "models"
parts = sorted(models.glob("project_xia_final_cs_shap_pipeline.zip.part-*"))
if not parts:
    raise SystemExit("No model archive parts were found.")
archive = models / "project_xia_final_cs_shap_pipeline.zip"
with archive.open("wb") as output:
    for part in parts:
        output.write(part.read_bytes())
with zipfile.ZipFile(archive) as bundle:
    bundle.extract("project_xia_final_cs_shap_pipeline.joblib", models)
model = models / "project_xia_final_cs_shap_pipeline.joblib"
expected = "7997757166a61de06a0a1415f9b03c7047ded251f2a2abfe20ec1d3d5ee36d96"
actual = hashlib.sha256(model.read_bytes()).hexdigest()
if actual != expected:
    raise SystemExit(f"SHA-256 mismatch: {actual}")
print(f"Restored and verified: {model}")
