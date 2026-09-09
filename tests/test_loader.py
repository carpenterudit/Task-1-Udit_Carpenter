from src.data.loader import DatasetLoader
from src.data.generate_sample import generate_sample

def test_loader(tmp_path):
 p=tmp_path/'x.csv'; generate_sample(p,200).to_csv(p,index=False); df,m=DatasetLoader(p,'is_fraud').load(); assert len(df)>0; assert m['target_detected']=='is_fraud'
