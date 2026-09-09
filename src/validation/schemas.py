import pandas as pd

def validate_with_pandera(df: pd.DataFrame, required_columns=None, target_column=None, lazy=True):
    try:
        import pandera.pandas as pa
    except Exception as exc:
        raise RuntimeError("Pandera is required for runtime schema validation. Install dependencies from requirements.txt; validation is not replaced with a fake fallback.") from exc
    required_columns=required_columns or []
    columns={c: pa.Column(df[c].dtype, nullable=True) for c in required_columns if c in df.columns}
    missing=[c for c in required_columns if c not in df.columns]
    if missing: raise ValueError(f"Missing required columns: {missing}")
    schema=pa.DataFrameSchema(columns, strict=False, coerce=False)
    if target_column and target_column in df.columns:
        schema=pa.DataFrameSchema({**columns,target_column:pa.Column(df[target_column].dtype, nullable=False)}, strict=False)
    return schema.validate(df,lazy=lazy)

def validation_failure_frame(exc) -> pd.DataFrame:
    rows=[]
    failure=getattr(exc,"failure_cases",None)
    if failure is not None:
        return failure.copy() if isinstance(failure,pd.DataFrame) else pd.DataFrame(failure)
    return pd.DataFrame([{"error":str(exc)}])
