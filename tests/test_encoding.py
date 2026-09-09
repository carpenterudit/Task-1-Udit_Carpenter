import pandas as pd
from src.preprocessing.encoding import CategoricalEncoder

def test_one_hot_and_unknown():
 e=CategoricalEncoder(); train=pd.DataFrame({'cat':['a','b'],'x':[1,2],'target':[0,1]}); a,_=e.fit_transform(train,'target'); test=pd.DataFrame({'cat':['c'],'x':[3],'target':[0]}); b=e.transform(test); assert 'cat_a' in a and b.shape[1]==a.shape[1]
