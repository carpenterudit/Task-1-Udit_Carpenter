from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_quality_report(df: pd.DataFrame, out_html: Path, figures_dir: Path, target=None):
    figures_dir.mkdir(parents=True,exist_ok=True); out_html.parent.mkdir(parents=True,exist_ok=True)
    numeric=df.select_dtypes(include="number"); cat=df.select_dtypes(include=["object","category","bool"])
    summary=pd.DataFrame({"dtype":df.dtypes.astype(str),"missing_count":df.isna().sum(),"missing_percentage":df.isna().mean()*100,"unique_values":df.nunique(dropna=True),"minimum":numeric.min(),"maximum":numeric.max(),"mean":numeric.mean(),"median":numeric.median(),"std":numeric.std()}).fillna("")
    q=df.select_dtypes(include="number").quantile([.25,.5,.75]).T.rename(columns={.25:"q1",.5:"median_q",.75:"q3"})
    miss=df.isna().mean().sort_values(ascending=False)*100
    plt.figure(figsize=(10,5)); miss.plot.bar(); plt.ylabel("Missing %"); plt.tight_layout(); plt.savefig(figures_dir/"missing_values.png"); plt.close()
    for c in numeric.columns:
        fig,ax=plt.subplots(figsize=(7,4)); ax.hist(df[c].dropna(),bins=30); ax.set_title(f"Distribution: {c}"); fig.tight_layout(); fig.savefig(figures_dir/f"hist_{c}.png"); plt.close(fig)
        fig,ax=plt.subplots(figsize=(7,3)); sns.boxplot(x=df[c].dropna(), ax=ax); ax.set_title(f"Box plot: {c}"); fig.tight_layout(); fig.savefig(figures_dir/f"box_{c}.png"); plt.close(fig)
    for c in cat.columns:
        plt.figure(figsize=(7,4)); df[c].value_counts(dropna=False).head(15).plot.bar(); plt.title(f"Counts: {c}"); plt.tight_layout(); plt.savefig(figures_dir/f"cat_{c}.png"); plt.close()
    if len(numeric.columns)>=2:
        plt.figure(figsize=(10,8)); sns.heatmap(numeric.corr(),cmap="coolwarm",center=0); plt.tight_layout(); plt.savefig(figures_dir/"correlation_heatmap.png"); plt.close()
    if target and target in df.columns:
        plt.figure(figsize=(6,4)); df[target].value_counts(dropna=False).plot.bar(); plt.title("Target distribution"); plt.tight_layout(); plt.savefig(figures_dir/"target_distribution.png"); plt.close()
    html=f"<html><head><title>Data Quality Report</title><style>body{{font-family:Arial;margin:32px}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ddd;padding:7px}}th{{background:#f2f2f2}}</style></head><body><h1>Data Quality Report</h1><p>Rows: {len(df)} | Columns: {len(df.columns)} | Duplicates: {df.duplicated().sum()}</p><h2>Column summary</h2>{summary.to_html()}<h2>Quartiles</h2>{q.to_html()}</body></html>"
    out_html.write_text(html,encoding="utf-8")
    return summary
