from pathlib import Path
import json,time
import pandas as pd
from src.data.loader import DatasetLoader
from src.data.validator import basic_integrity
from src.preprocessing.missing_values import MissingValueProcessor
from src.preprocessing.outliers import OutlierProcessor
from src.preprocessing.encoding import CategoricalEncoder
from src.features.engineering import FeatureEngineer
from src.features.correlation import CorrelationAnalyzer
from src.features.feature_registry import FeatureRegistry
from src.validation.schemas import validate_with_pandera, validation_failure_frame
from src.reporting.data_quality_report import generate_quality_report
from src.reporting.eda_report import generate_eda_report
from src.utils.paths import resolve_path

class PreprocessingPipeline:
    def __init__(self, config: dict, input_path=None):
        self.config=config; self.input_path=resolve_path(input_path or config["data"]["input_path"]); self.reports=resolve_path("reports"); self.figures=self.reports/"figures"
        self.steps=[]; self.original_rows=0; self.original_features=0; self.validation_status="NOT_RUN"; self.feature_registry=[]
    def _step(self,name,fn):
        start=time.perf_counter(); result=fn(); duration=time.perf_counter()-start; self.steps.append({"step":name,"status":"PASS","duration_seconds":round(duration,6)}); return result
    def run(self):
        df,meta=self._step("Data ingestion",lambda:DatasetLoader(self.input_path,self.config["target"]["column"]).load()); self.original_rows=len(df); self.original_features=len(df.columns)
        integrity=basic_integrity(df)
        if integrity: raise ValueError("Input integrity failure: "+"; ".join(integrity))
        before=df.copy()
        generate_quality_report(df,self.reports/"data_quality_report.html",self.figures,self.config["target"]["column"])
        self.steps.append({"step":"Quality analysis","status":"PASS"})
        missing=MissingValueProcessor(self.config["preprocessing"]["missing_threshold_drop"],self.config["preprocessing"]["missing_threshold_imputation"],self.config["preprocessing"]["random_state"])
        df,missing_report=self._step("Missing values",lambda:missing.transform(df)); missing_report.to_csv(self.reports/"missing_value_decisions.csv",index=False)
        out=OutlierProcessor(self.config["preprocessing"]["outlier_method"],self.config["preprocessing"]["zscore_threshold"]); df,out_report=self._step("Outliers",lambda:out.transform(df)); out_report.to_csv(self.reports/"outlier_report.csv",index=False)
        engineer=FeatureEngineer(self.config["features"]["entity_column"],self.config["features"]["timestamp_column"],self.config["features"]["amount_column"]); df=self._step("Feature engineering",lambda:engineer.transform(df)); self.feature_registry=engineer.registry
        if len(self.feature_registry)<self.config["features"]["minimum_engineered_features"]: raise ValueError("Minimum engineered feature requirement not met.")
        target=self.config["target"]["column"]
        # Keep identifiers/timestamps out of the predictor matrix; historical features already encode their useful information.
        id_col=self.config["features"]["entity_column"]
        time_col=self.config["features"]["timestamp_column"]
        drop_context=[c for c in [id_col,time_col] if c in df.columns and c != target]
        df=df.drop(columns=drop_context)
        encoder=CategoricalEncoder(); df,encoded=self._step("Categorical encoding",lambda:encoder.fit_transform(df,target))
        corr,corr_pairs,decisions,to_remove=self._step("Correlation",lambda:CorrelationAnalyzer(self.config["preprocessing"]["correlation_threshold"]).analyze(df,target))
        corr.to_csv(self.reports/"correlation_matrix.csv"); corr_pairs.to_csv(self.reports/"high_correlation_pairs.csv",index=False); decisions.to_csv(self.reports/"correlation_feature_decisions.csv",index=False)
        safe_remove=to_remove-{target}; df=df.drop(columns=list(safe_remove),errors="ignore")
        pd.DataFrame(self.feature_registry).to_csv(self.reports/"feature_dictionary.csv",index=False)
        registry=FeatureRegistry(resolve_path(self.config["feature_store"]["registry_path"])); registry.save(self.feature_registry)
        required=[target] + ([self.config["features"]["entity_column"]] if self.config["features"]["entity_column"] in df.columns else [])
        try:
            self._step("Pandera validation",lambda:validate_with_pandera(df,required,target,self.config["validation"]["lazy"])); self.validation_status="VALID"
            pd.DataFrame(columns=["error"]).to_csv(self.reports/"validation_failures.csv",index=False)
        except Exception as exc:
            self.validation_status="INVALID"; failure=validation_failure_frame(exc); failure.to_csv(self.reports/"validation_failures.csv",index=False); self.steps.append({"step":"Pandera validation","status":"FAIL","error":str(exc)})
            if self.config["validation"]["fail_on_invalid"]: raise
        generate_eda_report(before,df,self.reports/"eda_report.html",self.figures)
        output=resolve_path(self.config["data"]["output_path"]); output.parent.mkdir(parents=True,exist_ok=True); df.to_csv(output,index=False)
        summary={"original_rows":self.original_rows,"final_rows":len(df),"original_features":self.original_features,"final_features":len(df.columns),"missing_values_handled":int(before.isna().sum().sum()-df.isna().sum().sum()),"outliers_treated":int(out_report.outlier_count.sum()) if not out_report.empty else 0,"engineered_features":len(self.feature_registry),"highly_correlated_features_removed":len(safe_remove),"validation_status":self.validation_status,"final_dataset":str(output.relative_to(resolve_path('.'))),"steps":self.steps}
        self.reports.mkdir(exist_ok=True); (self.reports/"execution_summary.json").write_text(json.dumps(summary,indent=2,default=str),encoding="utf-8")
        trace=self._traceability(); (self.reports/"requirement_traceability.md").write_text(trace,encoding="utf-8")
        return df,summary
    def _traceability(self):
        return """# Requirement Traceability\n\n| Requirement | Implementation | Output/Test |\n|---|---|---|\n| Data ingestion | `src/data/loader.py` | `tests/test_loader.py` |\n| Advanced EDA | `src/reporting/data_quality_report.py` | `reports/data_quality_report.html` |\n| Missing values | `src/preprocessing/missing_values.py` | `reports/missing_value_decisions.csv` |\n| IQR/Z-score outliers | `src/preprocessing/outliers.py` | `reports/outlier_report.csv` |\n| 3+ predictive features | `src/features/engineering.py` | `reports/feature_dictionary.csv` |\n| Point-in-time correctness | `src/features/engineering.py` | `tests/test_features.py` |\n| Leakage prevention | target excluded from feature engineering/encoding | `tests/test_features.py` |\n| One-hot encoding | `src/preprocessing/encoding.py` | `tests/test_encoding.py` |\n| Correlation analysis | `src/features/correlation.py` | correlation CSV reports |\n| Pandera validation | `src/validation/schemas.py` | `reports/validation_failures.csv` |\n| Feature store | `feature_store/feature_store.py` | registry YAML |\n| CLI | `main.py` | execution summary |\n| GUI | `app.py` | Streamlit showcase |\n| Testing | `tests/` | `pytest -q` |\n| Reproducibility | config + seed | generated artifacts |\n"""
