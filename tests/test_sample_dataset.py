import pandas as pd
from pathlib import Path

def test_sample_dataset():
 p=Path('data/sample/transactions.csv'); d=pd.read_csv(p); assert len(d)>=1000; assert d.isna().sum().sum()>0; assert d.duplicated().sum()>0; assert {'customer_id','timestamp','amount','is_fraud'}.issubset(d.columns)
