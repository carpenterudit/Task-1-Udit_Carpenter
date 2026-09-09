from pathlib import Path
import tempfile, yaml, pandas as pd
import streamlit as st
from src.pipeline.preprocessing_pipeline import PreprocessingPipeline
from src.utils.paths import resolve_path

st.set_page_config(page_title="Advanced EDA & Feature Engineering", page_icon="📊", layout="wide")
REPORTS=resolve_path("reports"); CONFIG_PATH=resolve_path("config/config.yaml")

def load_summary():
    p=REPORTS/"execution_summary.json"
    return __import__('json').loads(p.read_text()) if p.exists() else None

def run_pipeline(uploaded):
    if uploaded is None: return
    with tempfile.NamedTemporaryFile(delete=False,suffix=".csv") as f: f.write(uploaded.getbuffer()); path=f.name
    cfg=yaml.safe_load(CONFIG_PATH.read_text())
    cfg["data"]["output_path"]="data/processed/final_dataset.csv"
    return PreprocessingPipeline(cfg,input_path=path).run()

st.sidebar.title("PROJECT")
st.sidebar.caption("Advanced EDA & Feature Engineering")
page=st.sidebar.radio("Navigation",["Overview","Data Upload","Data Quality","Missing Values","Outliers","Feature Engineering","Correlation Analysis","Validation","Feature Store","Final Dataset","Pipeline Execution","About Project"])

if page=="Overview":
    st.title("Advanced EDA & Feature Engineering")
    st.write("Transform raw, chaotic tabular data into mathematically clean, validated, ML-ready data. Model training is intentionally not the primary objective.")
    st.markdown("**INPUT → DATA QUALITY → MISSING VALUES → OUTLIERS → FEATURE ENGINEERING → ENCODING → CORRELATION → VALIDATION → ML-READY DATA**")
    s=load_summary()
    if s:
        cols=st.columns(8)
        vals=[s['final_rows'],s['final_features'],s['missing_values_handled'],s['original_rows']-s['final_rows'],s['engineered_features'],s['outliers_treated'],s['highly_correlated_features_removed'],s['validation_status']]
        labels=['Rows','Columns','Missing Cells Fixed','Rows Removed','Engineered Features','Outliers','Correlated Removed','Validation']
        for c,l,v in zip(cols,labels,vals): c.metric(l,v)
        st.success("VALID" if s['validation_status']=="VALID" else "INVALID") if s['validation_status']=="VALID" else st.error("INVALID")
    else: st.info("Run the pipeline from the Data Upload page or `python main.py` first.")

elif page=="Data Upload":
    st.title("Data Upload")
    uploaded=st.file_uploader("Upload CSV",type=["csv"])
    if uploaded:
        df=pd.read_csv(uploaded); st.write(f"**{uploaded.name}** — {len(df):,} rows × {len(df.columns)} columns")
        st.dataframe(df.head(20),use_container_width=True)
        st.write("Numerical:",df.select_dtypes('number').columns.tolist()); st.write("Categorical:",df.select_dtypes(include=['object','category','bool']).columns.tolist())
        if st.button("Run Pipeline",type="primary"):
            try:
                with st.spinner("Executing real backend pipeline…"): result,summary=run_pipeline(uploaded)
                st.session_state['last_result']=result; st.session_state['last_summary']=summary; st.success("Pipeline completed")
            except Exception as e: st.exception(e)

elif page=="Data Quality":
    st.title("Data Quality")
    p=REPORTS/"data_quality_report.html"
    s=load_summary()
    if s:
        c1,c2,c3=st.columns(3); c1.metric("Original Rows",s['original_rows']); c2.metric("Original Features",s['original_features']); c3.metric("Duplicates",pd.read_csv(resolve_path('data/sample/transactions.csv')).duplicated().sum())
    if p.exists(): st.components.v1.html(p.read_text(),height=700,scrolling=True)
    else: st.info("Run pipeline first.")

elif page=="Missing Values":
    st.title("Missing Values")
    p=REPORTS/"missing_value_decisions.csv"
    if p.exists():
        d=pd.read_csv(p); st.dataframe(d,use_container_width=True); st.metric("Total values fixed",int(d.missing_count.sum()))
    else: st.info("Run pipeline first.")

elif page=="Outliers":
    st.title("Outlier Analysis")
    p=REPORTS/"outlier_report.csv"
    if p.exists(): st.dataframe(pd.read_csv(p),use_container_width=True); st.latex(r"IQR=Q_3-Q_1;\quad lower=Q_1-1.5IQR;\quad upper=Q_3+1.5IQR")
    else: st.info("Run pipeline first.")

elif page=="Feature Engineering":
    st.title("Feature Engineering")
    p=REPORTS/"feature_dictionary.csv"
    if p.exists():
        d=pd.read_csv(p); st.dataframe(d,use_container_width=True); st.success(f"At least 3 meaningful predictive features requirement: {'PASSED' if len(d)>=3 else 'FAILED'}")
    else: st.info("Run pipeline first.")

elif page=="Correlation Analysis":
    st.title("Correlation Analysis")
    p=REPORTS/"high_correlation_pairs.csv"
    if p.exists(): st.dataframe(pd.read_csv(p),use_container_width=True); st.write("Threshold:",yaml.safe_load(CONFIG_PATH.read_text())['preprocessing']['correlation_threshold'])
    else: st.info("Run pipeline first.")

elif page=="Validation":
    st.title("Pandera Validation")
    s=load_summary(); status=s['validation_status'] if s else 'NOT RUN'
    (st.success if status=='VALID' else st.error)(status)
    p=REPORTS/"validation_failures.csv"; st.dataframe(pd.read_csv(p),use_container_width=True) if p.exists() else st.info("No validation report yet.")

elif page=="Feature Store":
    st.title("Feature Store")
    st.markdown("**ONE SOURCE OF TRUTH**")
    st.code("Raw Event → Feature Definition → Offline Historical Feature → Training Dataset\nRaw Event → Feature Definition → Online Feature → Serving")
    p=resolve_path("feature_store/feature_definitions.yaml")
    if p.exists(): st.dataframe(pd.DataFrame(yaml.safe_load(p.read_text()).get('features',[])),use_container_width=True)
    st.write("Point-in-time correctness, training-serving consistency and Feast compatibility are documented in `feature_store/README.md`.")

elif page=="Final Dataset":
    st.title("Final Dataset")
    p=resolve_path("data/processed/final_dataset.csv")
    if p.exists():
        df=pd.read_csv(p); st.write(df.shape); st.dataframe(df.head(100),use_container_width=True); st.download_button("Download final_dataset.csv",p.read_bytes(),file_name="final_dataset.csv")
    else: st.info("Run pipeline first.")

elif page=="Pipeline Execution":
    st.title("Pipeline Execution")
    s=load_summary()
    if s:
        for x in s['steps']: st.write(f"**{x['step']}** — {x['status']}" + (f" — {x['duration_seconds']}s" if 'duration_seconds' in x else ""))
    else: st.info("No execution summary yet.")

elif page=="About Project":
    st.title("About Project")
    st.write("Core project: reusable data-science/data-engineering pipeline. Showcase: Streamlit GUI over the same backend.")
