import numpy as np
import pandas as pd

class FeatureEngineer:
    def __init__(self, entity_column="customer_id", timestamp_column="timestamp", amount_column="amount"):
        self.entity_column=entity_column; self.timestamp_column=timestamp_column; self.amount_column=amount_column; self.registry=[]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out=df.copy()
        if self.timestamp_column in out.columns: out[self.timestamp_column]=pd.to_datetime(out[self.timestamp_column], errors="coerce")
        entity_ok=self.entity_column in out.columns; time_ok=self.timestamp_column in out.columns; amount_ok=self.amount_column in out.columns
        if entity_ok and time_ok:
            out=out.sort_values([self.entity_column,self.timestamp_column], kind="stable").copy()
            grp=out.groupby(self.entity_column, sort=False)
            out["transaction_frequency"]=grp.cumcount()
            if amount_ok:
                prior_sum=grp[self.amount_column].cumsum()-out[self.amount_column]
                prior_count=grp.cumcount()
                out["average_transaction_amount"]=prior_sum.div(prior_count.replace(0,np.nan)).fillna(0.0)
                out["amount_deviation_from_average"]=(out[self.amount_column]-out["average_transaction_amount"]).fillna(0.0)
                out["transaction_amount_log"]=np.log1p(out[self.amount_column].clip(lower=0))
            prev=grp[self.timestamp_column].shift(1)
            out["time_since_previous_transaction"]=(out[self.timestamp_column]-prev).dt.total_seconds().div(3600).fillna(0.0)
        elif amount_ok:
            out["transaction_amount_log"]=np.log1p(out[self.amount_column].clip(lower=0))
            out["amount_squared"]=np.square(out[self.amount_column])
            out["amount_deviation_from_median"]=out[self.amount_column]-out[self.amount_column].median()
        self.registry=self._build_registry(out, df.columns)
        return out

    def _build_registry(self, out, original_columns):
        names=[c for c in out.columns if c not in original_columns]
        templates={
          "transaction_frequency":("entity history before current event","Number of prior transactions for the entity","historical activity signal","low","yes"),
          "average_transaction_amount":("prior transaction amounts","Mean amount across prior events only","historical spending baseline","medium","yes"),
          "amount_deviation_from_average":("amount, prior average","current amount minus prior average","deviation from historical baseline","medium","yes"),
          "time_since_previous_transaction":("timestamp, prior timestamp","Current timestamp minus previous timestamp in hours","velocity/recency signal","medium","yes"),
          "transaction_amount_log":("amount","log1p(max(amount,0))","reduces right skew and compresses extreme values","medium","yes"),
          "amount_squared":("amount","amount²","nonlinear magnitude relationship","low","yes"),
          "amount_deviation_from_median":("amount","amount minus dataset median","robust deviation baseline","low","yes")}
        reg=[]
        for n in names:
            src,formula,exp,risk,pit=templates[n]
            reg.append({"feature_name":n,"source_columns":src,"formula":formula,"explanation":exp,"expected_predictive_value":risk,"leakage_risk":risk,"point_in_time_safe":pit})
        return reg
