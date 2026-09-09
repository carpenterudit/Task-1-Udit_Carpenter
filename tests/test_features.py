import pandas as pd
from src.features.engineering import FeatureEngineer

def test_features_and_point_in_time():
 d=pd.DataFrame({'customer_id':[1,1,1,2],'timestamp':['2025-01-01','2025-01-02','2025-01-04','2025-01-01'],'amount':[10.,20.,40.,5.],'is_fraud':[0,1,0,0]})
 o=FeatureEngineer().transform(d)
 o=o.sort_values(['customer_id','timestamp'])
 assert o.loc[o.customer_id.eq(1),'transaction_frequency'].tolist()==[0,1,2]
 assert o.loc[o.customer_id.eq(1),'average_transaction_amount'].tolist()==[0.,10.,15.]
 assert o.loc[o.customer_id.eq(1),'amount_deviation_from_average'].tolist()==[10.,10.,25.]
 assert len(o._is_copy if False else [x for x in o.columns if x not in d.columns])>=3

def test_no_target_feature_created():
 d=pd.DataFrame({'customer_id':[1,1],'timestamp':['2025-01-01','2025-01-02'],'amount':[10.,20.],'is_fraud':[1,0]})
 o=FeatureEngineer().transform(d); assert 'is_fraud' in o.columns and not any('fraud' in c.lower() for c in o.columns if c!='is_fraud')
