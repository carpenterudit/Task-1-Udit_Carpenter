import numpy as np
import pandas as pd

class OutlierProcessor:
    def __init__(self, method="IQR", zscore_threshold=3.0): self.method=method.upper(); self.zscore_threshold=zscore_threshold
    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        rows=[]
        for c in df.select_dtypes(include=np.number).columns:
            s=df[c].dropna()
            if self.method=="ZSCORE":
                mean=s.mean(); sd=s.std(ddof=0); lower=mean-self.zscore_threshold*sd; upper=mean+self.zscore_threshold*sd
            else:
                q1=s.quantile(.25); q3=s.quantile(.75); iqr=q3-q1; lower=q1-1.5*iqr; upper=q3+1.5*iqr
            mask=(df[c]<lower)|(df[c]>upper)
            rows.append({"column":c,"lower_bound":float(lower),"upper_bound":float(upper),"outlier_count":int(mask.sum()),"outlier_percentage":round(mask.mean()*100,4),"treatment_method":"clip/winsorize"})
        return pd.DataFrame(rows)
    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
        report=self.analyze(df); out=df.copy()
        for r in report.itertuples(index=False): out[r.column]=out[r.column].clip(r.lower_bound,r.upper_bound)
        return out, report
