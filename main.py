import yaml
from src.pipeline.preprocessing_pipeline import PreprocessingPipeline
from src.utils.paths import resolve_path
from src.utils.logging_config import configure_logging

def main():
    configure_logging(resolve_path("reports/pipeline.log"))
    config=yaml.safe_load(resolve_path("config/config.yaml").read_text(encoding="utf-8"))
    df,summary=PreprocessingPipeline(config).run()
    print("\n==================================================")
    print("PIPELINE COMPLETE")
    print(f"Original rows: {summary['original_rows']}")
    print(f"Final rows: {summary['final_rows']}")
    print(f"Original features: {summary['original_features']}")
    print(f"Final features: {summary['final_features']}")
    print(f"Missing values handled: {summary['missing_values_handled']}")
    print(f"Outliers treated: {summary['outliers_treated']}")
    print(f"Engineered features: {summary['engineered_features']}")
    print(f"Highly correlated features removed: {summary['highly_correlated_features_removed']}")
    print(f"Validation status: {summary['validation_status']}")
    print(f"Final dataset: {summary['final_dataset']}")
    print("==================================================")
if __name__ == "__main__": main()
