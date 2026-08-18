# Security

Do not load `.joblib` files from an untrusted source. Pickle-based formats may execute arbitrary code during deserialization. Verify hashes, inspect source, and use an isolated environment when appropriate. CSVs and the notebook can be reviewed without loading models.
