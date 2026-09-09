# Requirement Traceability

| Requirement | Implementation | Output/Test |
|---|---|---|
| Data ingestion | `src/data/loader.py` | `tests/test_loader.py` |
| Advanced EDA | `src/reporting/data_quality_report.py` | `reports/data_quality_report.html` |
| Missing values | `src/preprocessing/missing_values.py` | `reports/missing_value_decisions.csv` |
| IQR/Z-score outliers | `src/preprocessing/outliers.py` | `reports/outlier_report.csv` |
| 3+ predictive features | `src/features/engineering.py` | `reports/feature_dictionary.csv` |
| Point-in-time correctness | `src/features/engineering.py` | `tests/test_features.py` |
| Leakage prevention | target excluded from feature engineering/encoding | `tests/test_features.py` |
| One-hot encoding | `src/preprocessing/encoding.py` | `tests/test_encoding.py` |
| Correlation analysis | `src/features/correlation.py` | correlation CSV reports |
| Pandera validation | `src/validation/schemas.py` | `reports/validation_failures.csv` |
| Feature store | `feature_store/feature_store.py` | registry YAML |
| CLI | `main.py` | execution summary |
| GUI | `app.py` | Streamlit showcase |
| Testing | `tests/` | `pytest -q` |
| Reproducibility | config + seed | generated artifacts |
