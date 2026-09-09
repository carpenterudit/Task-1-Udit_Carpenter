import pandas as pd
from src.preprocessing.outliers import OutlierProcessor

def test_iqr_bounds_and_clipping():
 d=pd.DataFrame({'x':[1,2,3,4,100]}); p=OutlierProcessor(); r=p.analyze(d).iloc[0]; assert r.outlier_count==1; out,_=p.transform(d); assert out.x.max() < 100
