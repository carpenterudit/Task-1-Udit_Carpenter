from sklearn.preprocessing import StandardScaler
import pandas as pd
class NumericScaler:
    def __init__(self): self.scaler=None; self.columns=[]
    def fit_transform(self, df, exclude=None):
        exclude=set(exclude or []); self.columns=[c for c in df.select_dtypes(include="number").columns if c not in exclude]
        self.scaler=StandardScaler(); out=df.copy(); out[self.columns]=self.scaler.fit_transform(out[self.columns]); return out
    def transform(self, df):
        out=df.copy(); out[self.columns]=self.scaler.transform(out[self.columns]); return out
