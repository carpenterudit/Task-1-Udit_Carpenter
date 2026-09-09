import pandas as pd
from src.features.correlation import CorrelationAnalyzer

def test_correlation_detection_and_selection():
 d=pd.DataFrame({'a':[1,2,3,4,5],'b':[2,4,6,8,10],'c':[5,4,3,2,1],'target':[1,2,3,4,5]}); corr,pairs,dec,remove=CorrelationAnalyzer(.8).analyze(d,'target'); assert len(pairs)>=2; assert len(remove)>=1
