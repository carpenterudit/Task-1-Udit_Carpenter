import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

class MissingValueProcessor:
    def __init__(self, drop_threshold=0.05, impute_threshold=0.20, random_state=42):
        self.drop_threshold=drop_threshold; self.impute_threshold=impute_threshold; self.random_state=random_state
        self.decisions=[]

    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        rows=[]
        for c in df.columns:
            n=int(df[c].isna().sum()); p=n/len(df) if len(df) else 0
            if n==0: method="none"; reason="No missing values."
            elif p < self.drop_threshold: method="record_delete_if_appropriate"; reason="Very low missingness; deletion is considered only when records are otherwise safe to remove."
            elif p <= self.impute_threshold: method="median" if pd.api.types.is_numeric_dtype(df[c]) and abs(df[c].skew(skipna=True))>1 else ("mean" if pd.api.types.is_numeric_dtype(df[c]) else "mode"); reason="Moderate missingness; statistical imputation selected based on type/skewness."
            else: method="knn" if pd.api.types.is_numeric_dtype(df[c]) else "mode"; reason="High missingness; KNN is used for numeric columns when multidimensional relationships are available; categorical fallback is mode."
            rows.append({"column":c,"missing_count":n,"missing_percentage":round(p*100,4),"selected_method":method,"reason":reason})
        self.decisions=rows
        return pd.DataFrame(rows)

    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        decisions=self.analyze(df)
        out=df.copy()
        numeric=[c for c in out.select_dtypes(include=np.number).columns if out[c].isna().any()]
        high_numeric=[r["column"] for r in self.decisions if r["selected_method"]=="knn" and r["column"] in numeric]
        if high_numeric:
            all_num=out.select_dtypes(include=np.number).columns.tolist()
            imputer=KNNImputer(n_neighbors=min(5,max(1,len(out)-1)))
            out[all_num]=imputer.fit_transform(out[all_num])
            for c in high_numeric: decisions.loc[decisions.column.eq(c),"selected_method"]="knn"
        for c in out.columns:
            if not out[c].isna().any(): continue
            method=decisions.loc[decisions.column.eq(c),"selected_method"].iloc[0]
            if pd.api.types.is_numeric_dtype(out[c]):
                if method=="median": fill=out[c].median()
                elif method=="mean": fill=out[c].mean()
                else: fill=out[c].median()
            else:
                mode=out[c].mode(dropna=True)
                fill=mode.iloc[0] if not mode.empty else "UNKNOWN"
            out[c]=out[c].fillna(fill)
        return out, decisions
