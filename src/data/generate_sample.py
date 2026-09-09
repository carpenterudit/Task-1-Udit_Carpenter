from pathlib import Path
import numpy as np
import pandas as pd

def generate_sample(path: str | Path, n: int = 2500, seed: int = 42) -> pd.DataFrame:
    rng=np.random.default_rng(seed)
    customers=rng.integers(10000,10500,size=n)
    ts=pd.Timestamp('2025-01-01')+pd.to_timedelta(rng.integers(0,90*24*3600,size=n),unit='s')
    amount=np.round(rng.lognormal(mean=3.7,sigma=.65,size=n),2)
    base=amount*1.12+rng.normal(0,8,size=n)
    df=pd.DataFrame({
      'transaction_id':np.arange(1,n+1), 'customer_id':customers, 'timestamp':ts,
      'amount':amount, 'amount_proxy':np.round(base,2),
      'merchant_category':rng.choice(['grocery','electronics','travel','fuel','restaurant','fashion'],size=n,p=[.25,.12,.10,.12,.26,.15]),
      'channel':rng.choice(['web','mobile','pos'],size=n,p=[.30,.45,.25]),
      'city':rng.choice(['Indore','Ujjain','Bhopal','Dewas','Delhi','Pune'],size=n),
      'customer_tenure_months':rng.integers(1,120,size=n),
    })
    risk=(df['amount']>150).astype(float)*.8 + (df['channel'].eq('web')).astype(float)*.35 + (df['merchant_category'].eq('travel')).astype(float)*.25 + rng.random(n)*.35
    df['is_fraud']=(risk>0.95).astype(int)
    # Missing numerical/categorical values.
    for col,frac in [('amount',.07),('amount_proxy',.09),('customer_tenure_months',.03),('merchant_category',.08),('city',.06)]:
        idx=rng.choice(n,size=int(n*frac),replace=False); df.loc[idx,col]=np.nan
    # Realistic numerical outliers.
    out_idx=rng.choice(n,size=18,replace=False); df.loc[out_idx,'amount']=df['amount'].median()*rng.uniform(8,18,size=len(out_idx))
    df.loc[out_idx,'amount_proxy']=df.loc[out_idx,'amount']*1.12
    # Duplicate records.
    dup_idx=rng.choice(df.index,size=30,replace=False); df=pd.concat([df,df.loc[dup_idx]],ignore_index=True)
    return df

if __name__=='__main__':
    out=Path(__file__).resolve().parents[2]/'data/sample/transactions.csv'; out.parent.mkdir(parents=True,exist_ok=True); generate_sample(out).to_csv(out,index=False); print(out)
