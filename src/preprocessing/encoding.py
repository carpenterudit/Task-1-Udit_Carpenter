import pandas as pd
from sklearn.preprocessing import OneHotEncoder

class CategoricalEncoder:
    def __init__(self): self.encoder=None; self.columns=[]; self.feature_names=[]
    def fit_transform(self, df: pd.DataFrame, target: str | None=None) -> tuple[pd.DataFrame, list[str]]:
        self.columns=df.select_dtypes(include=["object","category","bool"]).columns.tolist()
        if target in self.columns: self.columns.remove(target)
        if not self.columns: return df.copy(), []
        self.encoder=OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        arr=self.encoder.fit_transform(df[self.columns])
        self.feature_names=self.encoder.get_feature_names_out(self.columns).tolist()
        numeric=df.drop(columns=self.columns)
        return pd.concat([numeric.reset_index(drop=True),pd.DataFrame(arr,columns=self.feature_names)],axis=1), self.feature_names
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.columns: return df.copy()
        arr=self.encoder.transform(df[self.columns]); numeric=df.drop(columns=self.columns)
        return pd.concat([numeric.reset_index(drop=True),pd.DataFrame(arr,columns=self.feature_names)],axis=1)
