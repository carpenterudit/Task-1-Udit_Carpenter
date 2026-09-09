import numpy as np
import pandas as pd

class CorrelationAnalyzer:
    def __init__(self, threshold=0.8): self.threshold=threshold
    def analyze(self, df: pd.DataFrame, target: str | None=None):
        numeric=df.select_dtypes(include=np.number).columns.tolist()
        corr=df[numeric].corr(method="pearson")
        pairs=[]; decisions=[]; to_remove=set()
        for i,a in enumerate(numeric):
            for b in numeric[i+1:]:
                v=corr.loc[a,b]
                if pd.notna(v) and abs(v)>=self.threshold:
                    pairs.append({"feature_1":a,"feature_2":b,"correlation":float(v),"absolute_correlation":float(abs(v))})
                    if target and target in df.columns and pd.api.types.is_numeric_dtype(df[target]):
                        ca=abs(df[a].corr(df[target])); cb=abs(df[b].corr(df[target])); retain=a if (pd.notna(ca) and (pd.isna(cb) or ca>=cb)) else b
                    else: retain=a if df[a].nunique(dropna=False)>=df[b].nunique(dropna=False) else b
                    remove=b if retain==a else a
                    decisions.append({"feature_1":a,"feature_2":b,"correlation":float(v),"absolute_correlation":float(abs(v)),"retained_feature":retain,"removed_feature":remove,"reason":"Retained the stronger target relationship when target was available; otherwise retained the feature with greater observed variability/cardinality."})
                    to_remove.add(remove)
        return corr,pd.DataFrame(pairs),pd.DataFrame(decisions),to_remove
