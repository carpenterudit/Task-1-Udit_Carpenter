import yaml, pytest
from src.data.generate_sample import generate_sample
from src.pipeline.preprocessing_pipeline import PreprocessingPipeline

def test_pipeline_artifacts(tmp_path):
 pytest.importorskip('pandera')
 p=tmp_path/'transactions.csv'; generate_sample(p,500)
 cfg=yaml.safe_load(open('config/config.yaml')); cfg['data']['output_path']=str(tmp_path/'final.csv'); cfg['feature_store']['registry_path']=str(tmp_path/'registry.yaml'); cfg['validation']['fail_on_invalid']=True
 out,s=PreprocessingPipeline(cfg,p).run(); assert len(out)>0 and s['engineered_features']>=3
