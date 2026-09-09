import pandas as pd, pytest
from src.validation.schemas import validate_with_pandera

def test_pandera_missing_dependency_is_explicit_or_validation_runs():
 d=pd.DataFrame({'is_fraud':[0,1]})
 try: validate_with_pandera(d,['is_fraud'],'is_fraud')
 except RuntimeError as e: assert 'Pandera is required' in str(e)
 except ImportError: pytest.fail('Pandera import should be handled explicitly')
