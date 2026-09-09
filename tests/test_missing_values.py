import pandas as pd
from src.preprocessing.missing_values import MissingValueProcessor

def test_mean_imputation():
 d=pd.DataFrame({'x':[1.,2.,None,4.]}); out,_=MissingValueProcessor().transform(d); assert out.x.isna().sum()==0 and out.x.iloc[2]==2.3333333333333335

def test_median_imputation():
 d=pd.DataFrame({'x':[1.,2.,100.,None]}); out,_=MissingValueProcessor().transform(d); assert out.x.isna().sum()==0

def test_knn_imputation():
 d=pd.DataFrame({'x':[1.,2.,3.,4.],'y':[2.,4.,None,8.]}); out,_=MissingValueProcessor(impute_threshold=.1).transform(d); assert out.isna().sum().sum()==0
