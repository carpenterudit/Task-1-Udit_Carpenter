from pathlib import Path
import pandas as pd

def generate_eda_report(df_before: pd.DataFrame, df_after: pd.DataFrame, out_html: Path, figures_dir: Path):
    out_html.parent.mkdir(parents=True,exist_ok=True)
    html=f"<html><body style='font-family:Arial;margin:32px'><h1>EDA & Transformation Report</h1><h2>Before</h2>{df_before.describe(include='all').T.to_html()}<h2>After</h2>{df_after.describe(include='all').T.to_html()}<h2>Figures</h2><p>See reports/figures for missingness, distributions, box plots, target and correlation visualizations.</p></body></html>"
    out_html.write_text(html,encoding="utf-8")
