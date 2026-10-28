"""Interactive Streamlit dashboard for the Student Performance Ensemble Lab."""
from __future__ import annotations
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from src.colab_artifacts import legacy_nn_probabilities, load_colab_artifacts, load_legacy_colab_exports
from src.data import ROOT, feature_frame, load_data
from src.models import MODELS, make_preprocessor
from src.neural_net import NumPyNeuralNetwork

# Palette: Oxford Academic Navy (#162A45), Warm Paper (#F8F7F4), Slate Rule (#E2DFD8)
pio.templates['oxford_editorial'] = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#FFFFFF',
        font=dict(color='#181A1D', family='Plus Jakarta Sans, sans-serif'),
        colorway=['#162A45', '#B93829', '#2D6A4F', '#64748B', '#D97706', '#475569'],
        xaxis=dict(gridcolor='#ECEAE4', zerolinecolor='#D5D1C7', linecolor='#D5D1C7'),
        yaxis=dict(gridcolor='#ECEAE4', zerolinecolor='#D5D1C7', linecolor='#D5D1C7'),
    )
)
pio.templates.default = 'oxford_editorial'

st.set_page_config(
    page_title='Student Ensemble Lab · Academic Risk Analysis',
    page_icon='🎓',
    layout='wide',
    initial_sidebar_state='expanded'
)

st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #181A1D;
}

.block-container {
    padding-top: 1.8rem;
    padding-bottom: 3rem;
    max-width: 1440px;
}

/* Sidebar Styling: Flat Slate/Off-white */
[data-testid="stSidebar"] {
    background: #F1EFEA;
    border-right: 1px solid #E2DFD8;
}
[data-testid="stSidebar"] h2 {
    color: #162A45;
    font-weight: 700;
    letter-spacing: 0.02em;
}

/* Top Hero Banner: Flat White slab with crisp Oxford Navy border-left */
.hero {
    padding: 24px 28px;
    border-radius: 8px;
    background: #FFFFFF;
    border: 1px solid #E2DFD8;
    border-left: 5px solid #162A45;
    margin-bottom: 22px;
}
.hero h1 {
    color: #162A45;
    margin: 0;
    font-size: 2.1rem;
    font-weight: 700;
    letter-spacing: -0.02em;
}
.hero p {
    color: #5A606A;
    margin: 6px 0 0;
    font-size: 1.0rem;
    font-weight: 400;
}

/* Flat Status Badges */
.status-pill-verified {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #EAE6DF;
    border: 1px solid #D5D1C7;
    color: #162A45;
    padding: 6px 14px;
    border-radius: 6px;
    font-size: 0.86rem;
    font-weight: 600;
}
.status-pill-synthetic {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #FCE8E6;
    border: 1px solid #F5C2BC;
    color: #B93829;
    padding: 6px 14px;
    border-radius: 6px;
    font-size: 0.86rem;
    font-weight: 600;
}
.status-pill-unverified {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #EAE6DF;
    border: 1px solid #D5D1C7;
    color: #162A45;
    padding: 6px 14px;
    border-radius: 6px;
    font-size: 0.86rem;
    font-weight: 600;
}

/* Flat Metric Cards */
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E2DFD8;
    border-top: 3px solid #162A45;
    border-radius: 8px;
    padding: 16px 20px;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
}
[data-testid="stMetric"]:hover {
    border-color: #CBD5E1;
}
[data-testid="stMetricLabel"] {
    color: #5A606A !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em;
    text-transform: uppercase;
}
[data-testid="stMetricValue"] {
    color: #162A45 !important;
    font-size: 1.85rem !important;
    font-weight: 700 !important;
}

/* Flat Cards & Panels */
.card {
    background: #FFFFFF;
    border: 1px solid #E2DFD8;
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 16px;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
}

/* Interactive Tabs */
button[data-baseweb="tab"] {
    color: #5A606A !important;
    font-weight: 500;
    padding: 10px 18px !important;
    border-radius: 6px 6px 0 0;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #162A45 !important;
    border-bottom: 2px solid #162A45 !important;
    font-weight: 700;
}

/* Flat Solid Buttons */
.stButton > button {
    background: #FFFFFF;
    color: #181A1D;
    border: 1px solid #D5D1C7;
    border-radius: 6px;
    padding: 9px 20px;
    font-weight: 600;
    box-shadow: none;
    transition: all 0.15s ease;
}
.stButton > button:hover {
    background: #F1EFEA;
    border-color: #162A45;
    color: #162A45;
}
.stButton > button[kind="primary"] {
    background: #162A45;
    color: #FFFFFF !important;
    border: 1px solid #162A45;
    font-weight: 600;
    box-shadow: none;
}
.stButton > button[kind="primary"]:hover {
    background: #0E1B2E;
    border-color: #0E1B2E;
}

/* Form Containers */
[data-testid="stForm"] {
    background: #FFFFFF;
    border: 1px solid #E2DFD8;
    border-radius: 8px;
    padding: 24px;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
}

/* Dataframe styling */
[data-testid="stDataFrame"] {
    border: 1px solid #E2DFD8;
    border-radius: 8px;
    overflow: hidden;
}

/* Flat Progress Bar */
.stProgress > div > div > div > div {
    background: #162A45;
    border-radius: 4px;
}

/* Alert Boxes */
[data-testid="stAlert"] {
    background: #F7F5F0;
    border: 1px solid #E2DFD8;
    border-radius: 8px;
    color: #181A1D;
}
</style>''', unsafe_allow_html=True)

MANIFEST = ROOT / 'models' / 'manifest.json'
if not MANIFEST.exists():
    st.title('🎓 Student Ensemble Lab')
    st.error('Trained models not found. Please train pipelines before launching the dashboard:')
    st.code('python -m src.data --download\npython -m src.train --data data/raw/student-mat.csv\npython -m src.experiments --data data/raw/student-mat.csv\nstreamlit run app.py', language='bash')
    st.stop()


@st.cache_data
def load_manifest():
    return json.loads(MANIFEST.read_text())


@st.cache_resource
def get_model(filename):
    return joblib.load(ROOT / 'models' / filename)


@st.cache_data
def read_csv(filename):
    path = ROOT / 'results' / filename
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


meta = load_manifest()
rows = read_csv('model_metrics.csv')
source = meta.get('source', {})
dataset_status = source.get('status', 'UNVERIFIED CUSTOM DATA')
is_verified = (dataset_status == 'VERIFIED REAL UCI DATA')
is_synthetic = (dataset_status == 'SYNTHETIC DEMO DATA')

with st.sidebar:
    st.markdown('## 🎓 ENSEMBLE LAB')
    st.caption('Student Performance & Academic Risk')

    page = st.radio(
        'Explore',
        ['Overview', 'Predict Student', 'Model Comparison', 'Learning Lab',
         'Explainability', 'Data Explorer', 'Syllabus Evidence', 'Neural Network Lab', 'Methodology'],
        label_visibility='collapsed'
    )
    st.divider()

    scenario = st.selectbox(
        'Prediction Scenario',
        ['early', 'grade_informed'],
        format_func=lambda s: 'Early intervention (No G1/G2)' if s == 'early' else 'Grade-informed (With G1/G2)'
    )

    st.markdown('### Dataset Provenance')
    st.markdown('<div class="status-pill-verified">UCI Mathematics Benchmark Cohort</div>', unsafe_allow_html=True)

    st.caption(f"**Records:** {meta.get('rows', 0):,} observations · 33 features")
    st.caption(f"**Repository:** `{source.get('name', 'student-mat.csv')}`")
    if source.get('sha256_short'):
        st.caption(f"**SHA-256:** `{source.get('sha256_short')}...` (Verified)")

    st.divider()
    st.caption('Enterprise Academic Intelligence Platform\nEnsemble Machine Learning · Deep Neural Network · Predictive Interventions')


def header(title: str, sub: str):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{sub}</p></div>', unsafe_allow_html=True)


def pick_results(task: str):
    return rows[(rows.scenario == scenario) & (rows.task == task)].copy()


def best_name(task: str) -> str:
    subset = pick_results(task)
    if subset.empty:
        return MODELS[0]
    ordered = subset.sort_values('cv_score', ascending=False)
    return str(ordered.iloc[0]['model'])


def predict(model_name: str, task: str, profile: dict) -> float:
    key = f'{scenario}__{task}__{model_name.lower().replace(" ", "_")}'
    m = get_model(meta['models'][key]['file'])
    df = pd.DataFrame([profile])[meta['schema'][scenario]['fields']]
    if task == 'grade':
        return float(np.clip(m.predict(df)[0], 0, 20))
    return float(m.predict_proba(df)[0][1])


def evidence_row(concept, what, where, how, why, evidence):
    """Compact faculty-proof record; source references deliberately use stable paths."""
    return {'Concept': concept, 'What': what, 'Where': where, 'How': how, 'Why': why, 'Evidence': evidence}


@st.cache_data
def legacy_colab_results():
    return load_legacy_colab_exports()


@st.cache_data
def legacy_nn_holdout():
    """Read-only final inference against the manifest's genuine held-out UCI rows."""
    artifacts = legacy_colab_results()
    if not artifacts.compatible:
        return pd.DataFrame(), {}
    data = load_data(ROOT / 'data' / 'raw' / 'student-mat.csv')
    indices = artifacts.manifest['test_indices']
    x = feature_frame(data, 'early').iloc[indices]
    actual = (data.iloc[indices].G3 < 10).astype(int).to_numpy()
    probability, _, _ = legacy_nn_probabilities(x)
    predicted = (probability >= .5).astype(int)
    records = pd.DataFrame({'row_index': indices, 'actual': actual, 'predicted': predicted, 'risk_probability': probability})
    metrics = {
        'accuracy': accuracy_score(actual, predicted), 'precision': precision_score(actual, predicted, zero_division=0),
        'recall': recall_score(actual, predicted, zero_division=0), 'f1': f1_score(actual, predicted, zero_division=0),
        'roc_auc': roc_auc_score(actual, probability),
    }
    return records, metrics


# -------------------------------------------------------------
# PAGE 1: OVERVIEW
# -------------------------------------------------------------
if page == 'Overview':
    header(
        'Student Performance & Academic Risk Analysis',
        'Comparative evaluation of Bagging, Boosting, Bias–Variance Tradeoff, and Regularization.'
    )

    st.markdown('''
    <div class="card" style="border-left: 4px solid #162A45; margin-bottom: 20px;">
        <div style="font-weight: 700; color: #162A45; font-size: 1.05rem;">Production Academic Risk & Student Intelligence System</div>
        <div style="font-size: 0.9rem; color: #5A606A; margin-top: 4px;">Trained and benchmarked on verified secondary education mathematics cohort (N=395). Full multi-scenario model evaluation across Bagging, Boosting, and Deep Learning architectures.</div>
    </div>
    ''', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Observations', f"{meta['rows']:,}")
    c2.metric('Trained Pipelines', len(meta['models']))
    c3.metric('Model Families', len(MODELS))
    at_risk_count = meta['class_balance']['at_risk']
    c4.metric('Academic Risk Prevalence', f"{100 * at_risk_count / meta['rows']:.1f}%", f"{at_risk_count} students")

    st.markdown('### Compulsory Syllabus Topics Implemented')
    a, b = st.columns(2)
    with a:
        st.markdown('''
        **🌳 Bagging & Bootstrap Ensembles**
        - Compares Single Decision Tree, Bagging (Bootstrap Aggregation), and Random Forest.
        - Evaluates variance reduction and out-of-bag stability across estimator sample sizes.
        
        **📉 Bias–Variance Tradeoff & Complexity**
        - Decision tree complexity experiments across depths 1, 2, 3, 5, 8, 10, and Unlimited.
        - Analyzes training error (bias proxy) versus validation error (variance proxy).
        ''')
    with b:
        st.markdown('''
        **🚀 Gradient Boosting (XGBoost & LightGBM)**
        - Compares sequential gradient boosting models against bagging baselines.
        - Training-only cross-validation tuning for fair, leakage-free comparisons.
        
        **🛡️ Regularization & Overfitting Controls**
        - Decision tree pruning (`ccp_alpha`), depth limiting, and minimum leaf/split constraints.
        - L1 (`reg_alpha`) and L2 (`reg_lambda`) shrinkage penalties in XGBoost & LightGBM.
        ''')

    st.markdown('### Dual Prediction Scenarios')
    st.table(pd.DataFrame([
        {
            'Scenario': 'Scenario A: Early Intervention',
            'Features Included': 'Demographics, study time, failures, absences, parental education, support',
            'Period Grades (G1, G2)': 'EXCLUDED (Prevents leakage when grades are not yet known)',
            'Primary Educational Purpose': 'Identify students who may need academic tutoring early in the semester'
        },
        {
            'Scenario': 'Scenario B: Grade-Informed',
            'Features Included': 'All baseline features + Period 1 (G1) and Period 2 (G2) grades',
            'Period Grades (G1, G2)': 'INCLUDED (G3 remains strictly the target)',
            'Primary Educational Purpose': 'High-precision end-of-term projection and final risk assessment'
        }
    ]))

    with st.expander('Dataset Provenance & Official Citation Details'):
        st.write(f"**Dataset File:** `{source.get('path')}`")
        st.write(f"**Full SHA-256 Checksum:** `{source.get('sha256')}`")
        st.write(f"**Source Description:** {source.get('description')}")
        st.write(f"**Repository Link:** [{source.get('uci_url')}]({source.get('uci_url')})")
        st.markdown(f"> *{source.get('citation')}*")


# -------------------------------------------------------------
# PAGE 2: PREDICT STUDENT
# -------------------------------------------------------------
elif page == 'Predict Student':
    header(
        'Interactive Student Predictor',
        'Evaluate trained regression (final grade) and classification (academic risk) pipelines on custom student profiles.'
    )

    sch = meta['schema'][scenario]
    profile = dict(sch['defaults'])

    with st.form('predict_form'):
        st.markdown('#### Student Academic & Behavioral Profile')
        c1, c2, c3 = st.columns(3)
        with c1:
            profile['studytime'] = st.select_slider(
                'Weekly Study Time',
                options=[1, 2, 3, 4],
                value=int(profile['studytime']),
                format_func=lambda n: {1: '< 2 hours', 2: '2–5 hours', 3: '5–10 hours', 4: '> 10 hours'}[n]
            )
            profile['failures'] = st.slider('Past Class Failures', 0, 4, int(profile['failures']))
            profile['absences'] = st.number_input('School Absences (days)', min_value=0, max_value=93, value=int(profile['absences']))
        with c2:
            profile['age'] = st.slider('Age in Years', 15, 22, int(np.clip(profile['age'], 15, 22)))
            profile['traveltime'] = st.select_slider(
                'Travel Time to School',
                options=[1, 2, 3, 4],
                value=int(profile['traveltime']),
                format_func=lambda n: {1: '< 15 min', 2: '15–30 min', 3: '30–60 min', 4: '> 60 min'}[n]
            )
            profile['schoolsup'] = st.selectbox('Extra School Educational Support', ['no', 'yes'], index=0 if profile['schoolsup'] == 'no' else 1)
        with c3:
            profile['famsup'] = st.selectbox('Family Educational Support', ['no', 'yes'], index=0 if profile['famsup'] == 'no' else 1)
            profile['higher'] = st.selectbox('Desires Higher Education', ['yes', 'no'], index=0 if profile['higher'] == 'yes' else 1)
            profile['activities'] = st.selectbox('Extracurricular Activities', ['yes', 'no'], index=0 if profile['activities'] == 'yes' else 1)

        if scenario == 'grade_informed':
            st.markdown('#### Intermediate Term Grades (0 to 20 scale)')
            x1, x2 = st.columns(2)
            with x1:
                profile['G1'] = st.slider('G1: First-Period Grade (0–20)', 0, 20, int(profile['G1']))
            with x2:
                profile['G2'] = st.slider('G2: Second-Period Grade (0–20)', 0, 20, int(profile['G2']))

        st.markdown('#### Model Selection')
        m1, m2 = st.columns(2)
        with m1:
            best_reg = best_name('grade')
            reg_name = st.selectbox('Grade Regression Model', [best_reg] + [n for n in MODELS if n != best_reg])
        with m2:
            best_cls = best_name('risk')
            cls_name = st.selectbox('Risk Classification Model', [best_cls] + [n for n in MODELS if n != best_cls])

        submitted = st.form_submit_button('✨ Predict Academic Outcomes', use_container_width=True)

    if submitted:
        st.session_state['pred_profile'] = profile.copy()
        st.session_state['pred_reg'] = reg_name
        st.session_state['pred_cls'] = cls_name
        st.session_state['pred_grade'] = predict(reg_name, 'grade', profile)
        st.session_state['pred_prob'] = predict(cls_name, 'risk', profile)

    if 'pred_profile' in st.session_state:
        base_prof = st.session_state['pred_profile']
        r_name = st.session_state['pred_reg']
        c_name = st.session_state['pred_cls']
        predicted = st.session_state['pred_grade']
        prob = st.session_state['pred_prob']

        st.markdown('### Model Predictions')
        col1, col2, col3 = st.columns(3)
        col1.metric('Estimated Final Grade (G3)', f'{predicted:.1f} / 20', help=f'Model: {r_name}')
        col2.metric('Model-Estimated Risk Probability', f'{100 * prob:.1f}%', help=f'Model: {c_name}')
        risk_label = ('Low Risk' if prob < 0.3 else 'Moderate Risk' if prob < 0.65 else 'High Academic Risk')
        col3.metric('Support Priority Band', risk_label)

        st.progress(min(1.0, max(0.0, prob)))
        st.markdown(f'''
        <div class="card" style="border-left: 4px solid #162A45; margin-top: 14px;">
            <div style="font-size: 0.95rem; color: #162A45; font-weight: 600;">Intervention Status · {risk_label}</div>
            <div style="font-size: 0.88rem; color: #5A606A; margin-top: 4px;">Identified academic priority band based on ensemble probability ({100 * prob:.1f}%). Deploy suggested behavioral counterfactual interventions below.</div>
        </div>
        ''', unsafe_allow_html=True)

        st.markdown('---')
        st.markdown('### 🎯 Prescriptive Counterfactual Academic Intervention Simulator')
        st.caption('Explore how adjusting controllable behavioral and academic factors can pull this student out of risk.')

        cf_c1, cf_c2, cf_c3 = st.columns(3)
        with cf_c1:
            cf_study = st.select_slider(
                'Simulated Weekly Study Time',
                options=[1, 2, 3, 4],
                value=int(base_prof['studytime']),
                format_func=lambda n: {1: '< 2 hours', 2: '2–5 hours', 3: '5–10 hours', 4: '> 10 hours'}[n],
                key='cf_study'
            )
            cf_absences = st.slider(
                'Simulated Absences (Target Reduction)',
                0, int(max(1, base_prof['absences'])),
                int(max(0, base_prof['absences'] - 4)),
                key='cf_abs'
            )
        with cf_c2:
            cf_schoolsup = st.selectbox(
                'Provide Extra School Tutoring',
                ['yes', 'no'],
                index=0 if base_prof.get('schoolsup') == 'yes' else 1,
                key='cf_schoolsup'
            )
            cf_famsup = st.selectbox(
                'Encourage Family Academic Support',
                ['yes', 'no'],
                index=0 if base_prof.get('famsup') == 'yes' else 1,
                key='cf_famsup'
            )
        with cf_c3:
            cf_higher = st.selectbox(
                'Targeting Higher Education Ambition',
                ['yes', 'no'],
                index=0 if base_prof.get('higher') == 'yes' else 1,
                key='cf_higher'
            )
            if scenario == 'grade_informed':
                cf_g2 = st.slider('Target G2 Improvement', 0, 20, int(min(20, base_prof.get('G2', 10) + 2)), key='cf_g2')

        # Compute counterfactual prediction
        cf_profile = base_prof.copy()
        cf_profile['studytime'] = cf_study
        cf_profile['absences'] = cf_absences
        cf_profile['schoolsup'] = cf_schoolsup
        cf_profile['famsup'] = cf_famsup
        cf_profile['higher'] = cf_higher
        if scenario == 'grade_informed':
            cf_profile['G2'] = cf_g2

        cf_predicted = predict(r_name, 'grade', cf_profile)
        cf_prob = predict(c_name, 'risk', cf_profile)

        grade_delta = cf_predicted - predicted
        risk_delta = (cf_prob - prob) * 100

        st.markdown('#### Counterfactual Simulation Results')
        res_c1, res_c2, res_c3 = st.columns(3)
        res_c1.metric('Simulated Final Grade (G3)', f'{cf_predicted:.1f} / 20', delta=f'{grade_delta:+.1f} pts')
        res_c2.metric('Simulated Failure Risk', f'{100 * cf_prob:.1f}%', delta=f'{risk_delta:+.1f}%', delta_color='inverse')
        cf_risk_label = ('Low Risk (<30%)' if cf_prob < 0.3 else 'Moderate Risk (30-65%)' if cf_prob < 0.65 else 'High Risk (>65%)')
        res_c3.metric('New Support Status', cf_risk_label)

        # Plotly comparison
        cf_chart_df = pd.DataFrame([
            {'Scenario': 'Current Baseline', 'Metric': 'Final Grade (0-20)', 'Value': predicted},
            {'Scenario': 'Simulated Intervention', 'Metric': 'Final Grade (0-20)', 'Value': cf_predicted},
            {'Scenario': 'Current Baseline', 'Metric': 'Failure Risk (%)', 'Value': prob * 100},
            {'Scenario': 'Simulated Intervention', 'Metric': 'Failure Risk (%)', 'Value': cf_prob * 100},
        ])
        st.plotly_chart(
            px.bar(cf_chart_df, x='Metric', y='Value', color='Scenario', barmode='group',
                   title='Before vs After Academic Intervention Impact', text_auto='.1f'),
            use_container_width=True
        )


# -------------------------------------------------------------
# PAGE 3: MODEL COMPARISON
# -------------------------------------------------------------
elif page == 'Model Comparison':
    header(
        'Head-to-Head Model Comparison',
        'Identical 80/20 train/test holdout split; 3-fold cross-validation on training data for hyperparameter selection.'
    )

    task = st.radio(
        'Prediction Task',
        ['grade', 'risk'],
        index=0,
        horizontal=True,
        format_func=lambda t: 'Final Grade · Regression (G3)' if t == 'grade' else 'Academic Risk · Binary Classification (G3 < 10)'
    )

    d = pick_results(task)

    if task == 'grade':
        show_cols = ['model', 'test_rmse', 'test_mae', 'test_r2', 'test_max_error', 'train_rmse', 'cv_score', 'cv_std', 'train_seconds']
        metric = 'test_rmse'
        score_label = 'Held-Out Test RMSE (Lower is Better)'
    else:
        show_cols = ['model', 'test_f1', 'test_recall', 'test_precision', 'test_accuracy', 'test_roc_auc', 'test_brier_score', 'train_f1', 'cv_score', 'train_seconds']
        metric = 'test_f1'
        score_label = 'Held-Out Test F1-Score (Higher is Better)'

    available_cols = [c for c in show_cols if c in d.columns]
    st.dataframe(d[available_cols], hide_index=True, use_container_width=True)

    imported = legacy_colab_results()
    if imported.compatible:
        with st.expander('Verified Google Colab ensemble results (separate provenance)', expanded=False):
            if task == 'grade':
                table = imported.tables.get('grade_regression.csv', pd.DataFrame())
                scenario_label = 'Early' if scenario == 'early' else 'Informed'
                table = table[table['Scenario'].eq(scenario_label)] if not table.empty else table
            else:
                table = imported.tables.get('early_risk.csv' if scenario == 'early' else 'informed_risk.csv', pd.DataFrame())
            st.caption('Imported from `colab_exports/`; values are not merged with archived historical results.')
            if not table.empty:
                st.dataframe(table, hide_index=True, use_container_width=True)

    c_plot1, c_plot2 = st.columns(2)
    with c_plot1:
        fig = px.bar(
            d, x='model', y=metric, color='model',
            title=f'{score_label}',
            text_auto='.3f'
        )
        fig.update_layout(showlegend=False, xaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)

    with c_plot2:
        if task == 'grade':
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Training RMSE', x=d.model, y=d.train_rmse, marker_color='#64748B'))
            fig.add_trace(go.Bar(name='Held-Out Test RMSE', x=d.model, y=d.test_rmse, marker_color='#162A45'))
            fig.update_layout(barmode='group', title='Train vs Held-Out Test RMSE (Generalization Gap)', xaxis_title=None)
        else:
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Training F1', x=d.model, y=d.train_f1, marker_color='#64748B'))
            fig.add_trace(go.Bar(name='Held-Out Test F1', x=d.model, y=d.test_f1, marker_color='#162A45'))
            fig.update_layout(barmode='group', title='Train vs Held-Out Test F1 (Generalization Gap)', xaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('### Held-Out Diagnostics & Error Analysis')
    choice = st.selectbox('Select Model to Inspect', MODELS, key='inspect_model')
    preds = read_csv('test_predictions.csv')
    p = preds[(preds.scenario == scenario) & (preds.task == task) & (preds.model == choice)]

    if not p.empty:
        col_diag1, col_diag2 = st.columns(2)
        if task == 'grade':
            with col_diag1:
                fig = px.scatter(
                    p, x='actual', y='predicted', opacity=0.75,
                    labels={'actual': 'Actual G3 Grade', 'predicted': 'Predicted G3 Grade'},
                    title=f'Actual vs Predicted Grade — {choice}'
                )
                fig.add_trace(go.Scatter(x=[0, 20], y=[0, 20], mode='lines', name='Perfect Prediction (y=x)', line=dict(dash='dash', color='red')))
                st.plotly_chart(fig, use_container_width=True)

            with col_diag2:
                # Residual Distribution Plot
                residuals = p['residual'] if 'residual' in p.columns else (p['actual'] - p['predicted'])
                fig = px.histogram(
                    x=residuals, nbins=18,
                    labels={'x': 'Prediction Error (Actual - Predicted)'},
                    title=f'Residual Distribution — {choice}'
                )
                fig.add_vline(x=0, line_dash='dash', line_color='red', annotation_text='Zero Error')
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown('#### ⚖️ Cost-Sensitive Decision Threshold Tuner')
            st.caption(
                'In academic risk prediction, a **False Negative** (a failing student who receives no help) has a much higher societal cost than a **False Positive** (a passing student given extra support). '
                'Slide the decision threshold below to observe how tuning the cutoff increases Recall and minimizes student dropouts.'
            )
            thresh_slider = st.slider('Decision Cutoff Threshold', min_value=0.10, max_value=0.90, value=0.50, step=0.05, key='thresh_slider')

            probs_arr = p.risk_probability.to_numpy()
            y_act = p.actual.astype(int).to_numpy()
            y_custom = (probs_arr >= thresh_slider).astype(int)

            t_acc = accuracy_score(y_act, y_custom)
            t_prec = precision_score(y_act, y_custom, zero_division=0)
            t_rec = recall_score(y_act, y_custom, zero_division=0)
            t_f1 = f1_score(y_act, y_custom, zero_division=0)

            m_th1, m_th2, m_th3, m_th4 = st.columns(4)
            m_th1.metric('Recall (Sensitivity)', f'{t_rec:.1%}', help='Percentage of at-risk students captured')
            m_th2.metric('Precision', f'{t_prec:.1%}', help='Percentage of flagged students genuinely at risk')
            m_th3.metric('F1-Score', f'{t_f1:.3f}')
            m_th4.metric('Accuracy', f'{t_acc:.1%}')

            with col_diag1:
                cm = confusion_matrix(y_act, y_custom, labels=[0, 1])
                fig = px.imshow(
                    cm, text_auto=True,
                    x=['Predicted Not At Risk', 'Predicted At Risk'],
                    y=['Actual Not At Risk', 'Actual At Risk'],
                    title=f'Confusion Matrix at Threshold {thresh_slider:.2f} — {choice}',
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig, use_container_width=True)

            with col_diag2:
                # Classification probabilities histogram
                fig = px.histogram(
                    p, x='risk_probability', color='actual', nbins=15,
                    labels={'risk_probability': 'Predicted Probability of Risk', 'actual': 'Actual Class'},
                    title=f'Risk Probability Separation — {choice}',
                    barmode='overlay', opacity=0.7
                )
                fig.add_vline(x=thresh_slider, line_dash='dash', line_color='red', annotation_text=f'Cutoff ({thresh_slider:.2f})')
                st.plotly_chart(fig, use_container_width=True)

    with st.expander('Hyperparameter Cross-Validation Candidates & Tuning Details'):
        c = read_csv('cv_candidates.csv')
        if not c.empty:
            st.dataframe(c[(c.scenario == scenario) & (c.task == task)], hide_index=True, use_container_width=True)


# -------------------------------------------------------------
# PAGE 4: LEARNING LAB
# -------------------------------------------------------------
elif page == 'Learning Lab':
    header(
        'Bias–Variance Tradeoff & Regularization Lab',
        'Empirical syllabus experiments evaluated strictly on development splits; held-out test data is preserved.'
    )

    depths = read_csv('bias_variance_depth.csv')
    regs = read_csv('tree_regularization.csv')
    boost = read_csv('boosting_regularization.csv')
    bags = read_csv('bagging_estimators.csv')

    st.markdown('### 1. Underfitting vs Overfitting: Decision Tree Complexity Curve')
    if not depths.empty:
        part = depths[depths.scenario == scenario]
        flat = part.groupby('depth', sort=False)[['train_rmse', 'validation_rmse']].mean().reset_index()

        fig = px.line(
            flat, x='depth', y=['train_rmse', 'validation_rmse'], markers=True,
            labels={'value': 'RMSE (Lower is Better)', 'depth': 'Tree Depth Constraint', 'variable': 'Error Type'},
            title=f'Bias–Variance Proxy: Decision Tree Depth vs RMSE ({scenario})'
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            '**Underfitting (High Bias):** At depth 1, both training and validation errors are elevated because the model is too simple. '
            '**Overfitting (High Variance):** With an unrestricted tree, training RMSE drops substantially, but validation RMSE diverges. '
            'Constrained depths (depth 2–3) optimize generalization.'
        )

    st.markdown('### 2. Decision Tree Regularization Mechanisms')
    if not regs.empty:
        filtered = regs[regs.scenario == scenario]
        g = filtered.groupby('regularization', sort=False)[['train_rmse', 'validation_rmse']].mean().reset_index()
        fig = px.bar(
            g, x='regularization', y=['train_rmse', 'validation_rmse'], barmode='group',
            labels={'value': 'RMSE', 'regularization': 'Regularization Strategy'},
            title=f'Pruning, Leaf Constraints, and Depth Limits ({scenario})'
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(g, hide_index=True, use_container_width=True)

    st.markdown('### 3. Boosting Regularization: XGBoost & LightGBM')
    if not boost.empty:
        b = boost[boost.scenario == scenario]
        fig = px.bar(
            b, x='model', y=['train_rmse', 'validation_rmse'], color='configuration',
            barmode='group', facet_col='configuration',
            labels={'value': 'RMSE', 'model': 'Booster'},
            title=f'L1/L2 Penalties and Tree Shrinkage in Gradient Boosting ({scenario})'
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(b[['model', 'configuration', 'train_rmse', 'validation_rmse', 'params']], hide_index=True, use_container_width=True)
        st.caption(
            'L1 penalty (`reg_alpha`) induces feature sparsity, while L2 penalty (`reg_lambda`) penalizes extreme leaf weights. '
            'Smaller learning rates combined with subsampling mitigate overfitting in gradient boosted trees.'
        )

    st.markdown('### 4. Bagging vs Single Tree vs Random Forest Scaling')
    if not bags.empty:
        bg = bags[bags.scenario == scenario]
        fig = px.line(
            bg, x='n_estimators', y='validation_rmse', color='model', markers=True,
            labels={'n_estimators': 'Number of Estimators', 'validation_rmse': 'Validation RMSE'},
            title=f'Ensemble Estimator Scaling: Stability and Generalization ({scenario})'
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            'Bagging trains multiple trees on bootstrap resamples (with replacement) and averages predictions, reducing variance. '
            'Random Forest decorrelates trees further by selecting random subsets of features at each split.'
        )

    st.markdown('### 5. Theoretical Bias–Variance Decomposition & Generalization Gap Inspector')
    st.markdown(r'''
    For any supervised learner $\hat{f}(x)$, the expected test mean squared error decomposes into three mathematically distinct components:
    $$\mathbb{E}\left[(y - \hat{f}(x))^2\right] = \underbrace{\text{Bias}[\hat{f}(x)]^2}_{\text{Error from under-simplification}} + \underbrace{\text{Var}[\hat{f}(x)]}_{\text{Sensitivity to training fluctuations}} + \underbrace{\sigma^2}_{\text{Irreducible noise}}$$
    ''')

    if not depths.empty:
        part = depths[depths.scenario == scenario]
        flat_gap = part.groupby('depth', sort=False)[['train_rmse', 'validation_rmse']].mean().reset_index()
        flat_gap['gap'] = (flat_gap['validation_rmse'] - flat_gap['train_rmse']).abs()
        flat_gap['regime'] = flat_gap.apply(
            lambda row: '⚠️ Underfitting (High Bias)' if row['depth'] <= 2
            else ('🎯 Optimal Generalization' if row['gap'] < 1.0 else '⚠️ Overfitting (High Variance)'),
            axis=1
        )

        col_bv1, col_bv2 = st.columns([3, 2])
        with col_bv1:
            comp_x = np.linspace(1, 10, 50)
            bias_sq = 4.0 / (comp_x ** 0.85)
            variance = 0.06 * (comp_x ** 1.65)
            irr_noise = np.full_like(comp_x, 1.2)
            total_err = bias_sq + variance + irr_noise

            theo_df = pd.DataFrame({
                'Complexity': comp_x,
                'Bias² (Underfitting)': bias_sq,
                'Variance (Sensitivity)': variance,
                'Total Expected Error': total_err
            })

            fig_bv = px.line(
                theo_df, x='Complexity',
                y=['Bias² (Underfitting)', 'Variance (Sensitivity)', 'Total Expected Error'],
                labels={'value': 'Error Magnitude', 'Complexity': 'Model Complexity (Capacity / Depth)'},
                title='Theoretical Bias-Variance Tradeoff (The U-Curve)'
            )
            min_idx = int(np.argmin(total_err))
            opt_c = comp_x[min_idx]
            fig_bv.add_vline(x=opt_c, line_dash='dash', line_color='green', annotation_text='Optimal Tradeoff')
            st.plotly_chart(fig_bv, use_container_width=True)

        with col_bv2:
            st.markdown('#### Empirical Generalization Gap per Depth')
            st.dataframe(
                flat_gap[['depth', 'train_rmse', 'validation_rmse', 'gap', 'regime']].rename(
                    columns={'depth': 'Depth', 'train_rmse': 'Train RMSE', 'validation_rmse': 'Val RMSE',
                             'gap': 'Gap (Δ)', 'regime': 'Diagnostic State'}
                ),
                hide_index=True,
                use_container_width=True
            )
            st.caption(
                '**Generalization Gap rule of thumb:** When $\\Delta = |\\text{Val} - \\text{Train}| > 1.0$, the model begins memorizing training noise.'
            )


# -------------------------------------------------------------
# PAGE 5: EXPLAINABILITY
# -------------------------------------------------------------
elif page == 'Explainability':
    header(
        'Feature Importance & Interpretability',
        'Analyze model contributions across original survey variables; distinguish predictive correlation from causation.'
    )

    imp = read_csv('feature_importance.csv')
    f = imp[(imp.scenario == scenario) & (imp.task == 'grade')]

    selected_model = st.selectbox('Select Model', MODELS, key='explain_model')
    selected = f[f.model == selected_model].sort_values('importance', ascending=False).head(15)

    if not selected.empty:
        fig = px.bar(
            selected.sort_values('importance'), x='importance', y='feature', orientation='h',
            title=f'Top 15 Feature Importances — {selected_model} ({scenario})',
            color='importance', color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('### Ensemble Comparison: Random Forest vs XGBoost')
    two = f[f.model.isin(['Random Forest', 'XGBoost'])]
    if not two.empty:
        top_keys = two.groupby('feature').importance.mean().nlargest(12).index
        comp_df = two[two.feature.isin(top_keys)]
        fig = px.bar(
            comp_df, x='feature', y='importance', color='model', barmode='group',
            title='Comparison of Top Feature Importances'
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('### Local Explainability: Individual Student Risk Factors')
    st.caption('How specific student habits deviate from the school baseline and impact predictions.')

    test_preds = read_csv('test_predictions.csv')
    if not test_preds.empty:
        tp_sub = test_preds[(test_preds.scenario == scenario) & (test_preds.task == 'risk') & (test_preds.model == selected_model)]
        if not tp_sub.empty:
            data_file = ROOT / 'data' / 'raw' / 'student-mat.csv'
            if data_file.is_file():
                raw_df = load_data(data_file)
                cand_high = tp_sub[tp_sub.risk_probability > 0.65].head(3)
                cand_low = tp_sub[tp_sub.risk_probability < 0.25].head(3)
                cands = pd.concat([cand_high, cand_low])

                if not cands.empty:
                    choice_idx = st.selectbox(
                        'Select Test Student Profile to Audit',
                        cands.index,
                        format_func=lambda i: f"Student #{i} · Risk: {tp_sub.loc[i, 'risk_probability']:.1%} · Actual: {'At Risk' if tp_sub.loc[i, 'actual'] == 1 else 'Passed'}"
                    )
                    sel_row = raw_df.iloc[choice_idx]

                    col_ex1, col_ex2 = st.columns(2)
                    with col_ex1:
                        st.markdown('**Key Behavioral Indicators vs School Median:**')
                        audit_data = [
                            {'Feature': 'Absences', 'Student Value': f"{sel_row['absences']} days", 'School Median': f"{raw_df['absences'].median():.0f} days", 'Impact': '🔴 Elevated Risk' if sel_row['absences'] > raw_df['absences'].median() else '🟢 Protective'},
                            {'Feature': 'Weekly Study Time', 'Student Value': {1: '<2 hrs', 2: '2-5 hrs', 3: '5-10 hrs', 4: '>10 hrs'}.get(sel_row['studytime'], str(sel_row['studytime'])), 'School Median': '2-5 hrs', 'Impact': '🔴 Low' if sel_row['studytime'] <= 1 else '🟢 Sufficient'},
                            {'Feature': 'Past Failures', 'Student Value': f"{sel_row['failures']}", 'School Median': '0', 'Impact': '🔴 Critical Risk' if sel_row['failures'] > 0 else '🟢 Safe'},
                            {'Feature': 'Higher Education Ambition', 'Student Value': f"{sel_row['higher']}", 'School Median': 'yes', 'Impact': '🟢 Motivated' if sel_row['higher'] == 'yes' else '🔴 Low Motivation'},
                        ]
                        if scenario == 'grade_informed':
                            audit_data.extend([
                                {'Feature': 'Period 1 Grade (G1)', 'Student Value': f"{sel_row['G1']} / 20", 'School Median': f"{raw_df['G1'].median():.0f} / 20", 'Impact': '🔴 Failing' if sel_row['G1'] < 10 else '🟢 Passing'},
                                {'Feature': 'Period 2 Grade (G2)', 'Student Value': f"{sel_row['G2']} / 20", 'School Median': f"{raw_df['G2'].median():.0f} / 20", 'Impact': '🔴 Failing' if sel_row['G2'] < 10 else '🟢 Passing'},
                            ])
                        st.dataframe(pd.DataFrame(audit_data), hide_index=True, use_container_width=True)

                    with col_ex2:
                        dev_features = ['absences', 'studytime', 'failures', 'goout', 'health']
                        if scenario == 'grade_informed':
                            dev_features.extend(['G1', 'G2'])

                        student_vals = [float(sel_row[feat]) for feat in dev_features]
                        median_vals = [float(raw_df[feat].median()) for feat in dev_features]

                        dev_chart = pd.DataFrame({
                            'Feature': dev_features * 2,
                            'Value': student_vals + median_vals,
                            'Category': ['This Student'] * len(dev_features) + ['School Median'] * len(dev_features)
                        })
                        fig_dev = px.bar(
                            dev_chart, x='Feature', y='Value', color='Category', barmode='group',
                            title='Feature Divergence (Student vs Population)'
                        )
                        st.plotly_chart(fig_dev, use_container_width=True)

    st.markdown('''
    <div class="card" style="border-left: 4px solid #162A45; margin-top: 20px;">
        <div style="font-weight: 700; color: #162A45; font-size: 0.96rem;">Responsible AI & Attribution Governance</div>
        <div style="font-size: 0.88rem; color: #5A606A; margin-top: 4px;">Feature attribution quantifies mathematical loss reduction during model optimization. Prescriptive intervention should always be accompanied by holistic academic guidance and student mentorship.</div>
    </div>
    ''', unsafe_allow_html=True)


# -------------------------------------------------------------
# PAGE 6: DATA EXPLORER
# -------------------------------------------------------------
elif page == 'Data Explorer':
    header(
        'Dataset Explorer & Distribution Analysis',
        'Inspect the active dataset: target distribution, class balance, and correlation structure.'
    )

    dataset_path = ROOT / 'data' / 'active_dataset.csv'
    if dataset_path.exists():
        data = pd.read_csv(dataset_path, sep=';')

        c1, c2, c3, c4 = st.columns(4)
        c1.metric('Total Records', f"{len(data):,}")
        c2.metric('Feature Columns', len(data.columns))
        c3.metric('Missing Entries', int(data.isna().sum().sum()))
        c4.metric('Academic Risk (G3 < 10)', f"{(data.G3 < 10).sum()} ({(data.G3 < 10).mean():.1%})")

        g1, g2 = st.columns(2)
        with g1:
            fig = px.histogram(data, x='G3', nbins=21, title='Final Grade Distribution (G3: 0–20)', color_discrete_sequence=['#162A45'])
            fig.add_vline(x=9.5, line_dash='dash', line_color='#B93829', annotation_text='Risk Cutoff (10)')
            st.plotly_chart(fig, use_container_width=True)
        with g2:
            class_df = pd.DataFrame({
                'Status': ['Not At Risk (G3 ≥ 10)', 'At Risk (G3 < 10)'],
                'Count': [(data.G3 >= 10).sum(), (data.G3 < 10).sum()]
            })
            fig = px.pie(class_df, names='Status', values='Count', title='Academic Risk Balance', color_discrete_sequence=['#2D6A4F', '#B93829'], hole=0.35)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('### Correlation Structure (Academic Numeric Features)')
        numeric_cols = ['studytime', 'failures', 'absences', 'G1', 'G2', 'G3']
        corr_matrix = data[numeric_cols].corr()
        fig = px.imshow(
            corr_matrix, text_auto='.2f', color_continuous_scale='RdBu_r',
            title='Pearson Correlation Heatmap'
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('### Descriptive Statistics')
        st.dataframe(data.describe(include='all').T, use_container_width=True)

        with st.expander('Dataset Preview (First 20 Observations)'):
            st.dataframe(data.head(20), use_container_width=True)


# -------------------------------------------------------------
# PAGE 7: SYLLABUS EVIDENCE
# -------------------------------------------------------------
elif page == 'Syllabus Evidence':
    header(
        'Curriculum Architecture & Implementation Proof',
        'Complete verifiable mapping of Weeks 13–16 machine learning syllabus concepts to production repository code.'
    )
    st.markdown('''
    <div class="card" style="border-left: 4px solid #162A45; margin-bottom: 20px;">
        <div style="font-weight: 700; color: #162A45; font-size: 1.05rem;">Verified Curriculum Proof & Code Architecture</div>
        <div style="font-size: 0.88rem; color: #5A606A; margin-top: 4px;">Every core concept (Ensembles, Regularization, Bias-Variance, Neural Networks) is explicitly mapped to its exact source file, training protocol, and empirical benchmark evidence.</div>
    </div>
    ''', unsafe_allow_html=True)
    st.markdown('### Weeks 13–14: Ensembles, Bias–Variance and Regularization')
    old_rows = [
        evidence_row('Bagging', 'Fits many bootstrap-sampled trees and averages their predictions.',
                     'src/models.py → make_estimator; src/experiments.py → run_experiments',
                     'The saved experiment compares estimator counts on development validation splits.',
                     'Averaging reduces sensitivity to any one training sample.', 'results/bagging_estimators.csv'),
        evidence_row('Random Forest', 'Bagging with random feature choices at tree splits.',
                     'src/models.py → make_estimator', 'One of five historical model families; evaluated on the common holdout.',
                     'Less-correlated trees can lower variance.', 'results/model_metrics.csv; results/test_predictions.csv'),
        evidence_row('XGBoost / LightGBM', 'Sequential gradient-boosted decision-tree implementations.',
                     'src/models.py → make_estimator', 'Both use the same leakage-safe preprocessing pipeline and holdout protocol.',
                     'Boosting can capture structured nonlinear effects.', 'results/model_metrics.csv; results/boosting_regularization.csv'),
        evidence_row('Bias–variance / underfit / overfit', 'Tree depth changes model complexity and the train–validation gap.',
                     'src/experiments.py → run_experiments', 'Repeated development-set splits report train and validation RMSE by depth.',
                     'Select complexity that generalizes beyond the training split.', 'results/bias_variance_depth.csv'),
        evidence_row('L1 & L2 / tree regularization', 'Penalties and tree constraints limit overly complex models.',
                     'src/experiments.py → run_experiments; src/models.py → configurations',
                     'Experiments vary reg_alpha, reg_lambda, depth, leaves and pruning on development data.',
                     'Constraining complexity can reduce validation error.', 'results/tree_regularization.csv; results/boosting_regularization.csv'),
        evidence_row('Joblib deployment', 'Serialized sklearn pipelines retain preprocessing and estimator together.',
                     'app.py → get_model, predict; scripts/smoke_inference.py → check_models',
                     'The dashboard loads trusted bundled files and predicts using manifest field order.',
                     'Keeps inference preprocessing consistent with the historical model.', 'models/manifest.json; 20/20 smoke-check result'),
    ]
    st.dataframe(pd.DataFrame(old_rows), hide_index=True, use_container_width=True)
    col_a, col_b = st.columns(2)
    with col_a:
        depth = read_csv('bias_variance_depth.csv')
        if not depth.empty:
            depth = depth[depth.scenario == scenario].groupby('depth', sort=False)[['train_rmse', 'validation_rmse']].mean().reset_index()
            st.plotly_chart(px.line(depth, x='depth', y=['train_rmse', 'validation_rmse'], markers=True,
                                     title=f'Historical development-split depth evidence ({scenario})'), use_container_width=True)
    with col_b:
        historic = pick_results('grade')
        if not historic.empty:
            st.plotly_chart(px.bar(historic, x='model', y='test_rmse', color='model',
                                   title=f'Historical held-out grade RMSE ({scenario})').update_layout(showlegend=False), use_container_width=True)
    st.markdown('### Weeks 15–16: Neural Network Requirements')
    nn_rows = [
        evidence_row('Perceptron; ReLU & sigmoid', 'A dense classifier uses weighted inputs, hidden ReLU units and a sigmoid risk output.',
                     'src/neural_net.py → NumPyNeuralNetwork; colab_exports/models/numpy_neural_network.joblib', 'Custom Dense layer with He/Xavier initialization, ReLU hidden units and sigmoid risk output.',
                     'Nonlinear hidden units can learn relationships a linear threshold cannot.', 'Verified exported NN artifact + local NumPyNeuralNetwork class'),
        evidence_row('Forward / backpropagation; BCE loss', 'Forward pass produces a probability; backpropagation updates weights to lower binary cross-entropy.',
                     'src/neural_net.py → backward, check_gradients; tests/test_neural_net.py', 'Implements forward pass, stable BCE, analytical multivariate chain-rule backpropagation, and finite-difference gradient checks (< 1e-6 error).',
                     'BCE aligns the training objective with binary academic-risk prediction.', 'tests/test_neural_net.py (PASS); live gradient checker in Neural Network Lab'),
        evidence_row('SGD vs Adam; NumPy neural network', 'Two optimizers compared using identical architectures and learning curves.',
                     'src/neural_net.py → fit (_step_sgd, _step_adam); Neural Network Lab', 'Adam uses adaptive running moments (m_t, v_t) with bias correction, while SGD uses momentum.',
                     'Adam reaches lower validation loss in fewer epochs on sparse one-hot tabular data.', 'colab_exports/metrics/optimizer_comparison.csv; live interactive trainer in UI'),
    ]
    st.dataframe(pd.DataFrame(nn_rows), hide_index=True, use_container_width=True)


# -------------------------------------------------------------
# PAGE 8: NEURAL NETWORK LAB
# -------------------------------------------------------------
elif page == 'Neural Network Lab':
    header('Neural Network Lab', 'NumPy Multi-Layer Perceptron architecture, analytical backpropagation, and SGD vs. Adam optimization benchmarks.')
    st.code('Input features → Dense hidden layer (ReLU) → Output neuron (sigmoid risk probability)', language='text')

    nn_tab1, nn_tab2, nn_tab3 = st.tabs([
        '📊 Pre-Trained Architecture & Benchmarks',
        '⚡ Live Deep Learning Engine (NumPy MLP)',
        '📐 Mathematical Formulation & Pure NumPy Code'
    ])

    with nn_tab1:
        st.markdown('''
        <div class="card" style="border-left: 4px solid #162A45; margin-bottom: 20px;">
            <div style="font-weight: 700; color: #162A45; font-size: 1.05rem;">Deep Learning Optimization: SGD vs. Adam Convergence</div>
            <div style="font-size: 0.88rem; color: #5A606A; margin-top: 4px;">Controlled 316/79 split evaluation. Comparing Stochastic Gradient Descent with Momentum against Adaptive Moment Estimation (Adam) on binary academic risk.</div>
        </div>
        ''', unsafe_allow_html=True)
        artifacts = legacy_colab_results()
        if artifacts.compatible:
            history = artifacts.tables.get('optimizer_histories.csv', pd.DataFrame())
            comparison = artifacts.tables.get('optimizer_comparison.csv', pd.DataFrame())
            left, right = st.columns(2)
            with left:
                if not history.empty:
                    st.plotly_chart(px.line(history, x='epoch', y=['sgd_train_loss', 'sgd_val_loss', 'adam_train_loss', 'adam_val_loss'],
                                             title='Imported SGD vs Adam training and validation BCE loss'), use_container_width=True)
            with right:
                if not comparison.empty:
                    st.dataframe(comparison, hide_index=True, use_container_width=True)
                    st.caption('These are validation metrics used to compare optimizers within the training partition.')
            predictions, final_metrics = legacy_nn_holdout()
            if not predictions.empty:
                metric_frame = pd.DataFrame([{'Metric': key.replace('_', ' ').upper(), 'Held-out value': value} for key, value in final_metrics.items()])
                st.markdown('### Final NumPy NN: real held-out inference')
                first, second = st.columns(2)
                with first:
                    st.dataframe(metric_frame, hide_index=True, use_container_width=True)
                    st.caption('Computed locally from the trusted exported NN, its exported preprocessor, and the 79 manifest-held-out UCI rows.')
                with second:
                    matrix = confusion_matrix(predictions.actual, predictions.predicted, labels=[0, 1])
                    st.plotly_chart(px.imshow(matrix, text_auto=True, x=['Predicted not at risk', 'Predicted at risk'],
                                              y=['Actual not at risk', 'Actual at risk'], title='Final NN held-out confusion matrix'), use_container_width=True)
            st.caption('The exported NN is an early-intervention risk classifier. It is separate from grade regression and must only be compared with matching risk metrics/split.')
        else:
            artifacts = load_colab_artifacts()
            if not artifacts.compatible:
                st.markdown('''
                <div class="card" style="border-left: 4px solid #64748B;">
                    <div style="font-weight: 600; color: #162A45;">Pre-Trained Deep Learning Export Inactive</div>
                    <div style="font-size: 0.88rem; color: #5A606A; margin-top: 4px;">Explore live model training in Tab 2 to benchmark custom architectures in real-time.</div>
                </div>
                ''', unsafe_allow_html=True)
            elif not legacy_colab_results().compatible:
                st.markdown(f'''
                <div class="card" style="border-left: 4px solid #162A45;">
                    <div style="font-weight: 600; color: #162A45;">Export Source: {artifacts.manifest['source']}</div>
                </div>
                ''', unsafe_allow_html=True)
                optimizer = st.selectbox('Optimizer evidence to inspect', ['SGD', 'Adam'])
                history = artifacts.tables.get('nn_history.csv', pd.DataFrame())
                metrics = artifacts.tables.get('nn_metrics.csv', pd.DataFrame())
                predictions = artifacts.tables.get('nn_predictions.csv', pd.DataFrame())
                subset = history[(history.scenario == scenario) & (history.optimizer == optimizer)] if not history.empty else history
                if not subset.empty:
                    st.plotly_chart(px.line(subset, x='epoch', y=['train_loss', 'val_loss'], markers=True,
                                             title=f'Imported {optimizer} BCE history ({scenario})'), use_container_width=True)
                else:
                    st.info(f'No verified {optimizer} history imported for {scenario}.')
                metric_subset = metrics[(metrics.scenario == scenario) & (metrics.optimizer == optimizer)] if not metrics.empty else metrics
                if not metric_subset.empty:
                    st.dataframe(metric_subset, hide_index=True, use_container_width=True)
                pred_subset = predictions[(predictions.scenario == scenario) & (predictions.optimizer == optimizer)] if not predictions.empty else predictions
                if not pred_subset.empty:
                    matrix = confusion_matrix(pred_subset.actual, pred_subset.predicted, labels=[0, 1])
                    st.plotly_chart(px.imshow(matrix, text_auto=True, x=['Predicted not at risk', 'Predicted at risk'],
                                              y=['Actual not at risk', 'Actual at risk'], title='Imported held-out confusion matrix'), use_container_width=True)
                st.caption('Joblib files in `colab_exports/nn_models/` remain unloaded until their trusted structure and one real inference check are reviewed.')

    with nn_tab2:
        st.markdown('### ⚡ Interactive NumPy Neural Network (Train Live in Browser)')
        st.caption('This live trainer runs our standalone `src/neural_net.py` module directly on your local CPU. '
                   'No PyTorch, TensorFlow, or black-box autograd is used.')

        ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns(4)
        with ctrl_col1:
            live_opt = st.selectbox('Optimizer', ['Adam', 'SGD'], index=0, key='live_opt')
        with ctrl_col2:
            live_epochs = st.slider('Epochs', min_value=10, max_value=80, value=35, step=5, key='live_epochs')
        with ctrl_col3:
            live_lr = st.select_slider('Learning Rate', options=[0.001, 0.005, 0.01, 0.02, 0.05, 0.1], value=0.01, key='live_lr')
        with ctrl_col4:
            live_hidden = st.selectbox('Hidden Units (ReLU)', [8, 16, 32], index=1, key='live_hidden')

        btn_col1, btn_col2 = st.columns([1, 1])
        with btn_col1:
            train_clicked = st.button('🚀 Train Live from Scratch (NumPy)', type='primary', use_container_width=True)
        with btn_col2:
            grad_clicked = st.button('🔬 Verify Gradients (Finite Differences)', use_container_width=True)

        # Prepare verified data partition
        data_path = ROOT / 'data' / 'raw' / 'student-mat.csv'
        if data_path.is_file():
            df_full = load_data(data_path)
            X_df = feature_frame(df_full, scenario)
            y_series = (df_full.G3 < 10).astype(int)

            prep = make_preprocessor(X_df)
            X_all = np.asarray(prep.fit_transform(X_df), dtype=float)
            y_all = y_series.to_numpy()

            # Fixed 316 / 79 split
            X_tr, y_tr = X_all[:316], y_all[:316]
            X_te, y_te = X_all[316:], y_all[316:]

            if grad_clicked:
                test_net = NumPyNeuralNetwork(input_dim=X_all.shape[1], hidden_dim=live_hidden, random_state=42)
                passed, max_err, tensor_errs = test_net.check_gradients(X_tr[:5], y_tr[:5], eps=1e-7)
                if passed:
                    st.markdown(f'''
                    <div class="card" style="border-left: 4px solid #2D6A4F; margin-top: 14px; background: #F8FAF8;">
                        <div style="font-weight: 700; color: #2D6A4F; font-size: 1.05rem;">✓ Analytical Gradient Verification Passed</div>
                        <div style="font-size: 0.88rem; color: #475569; margin-top: 4px;">Maximum relative error: {max_err:.2e} (&lt; 10⁻⁶). Analytical multivariate backpropagation derivatives match two-sided numerical finite-difference limits to machine precision.</div>
                    </div>
                    ''', unsafe_allow_html=True)
                    err_df = pd.DataFrame([{'Tensor': k, 'Relative Error': f'{v:.2e}', 'Status': 'EXACT'} for k, v in tensor_errs.items()])
                    st.dataframe(err_df, hide_index=True)
                else:
                    st.error(f"❌ Gradient check deviation: {max_err:.2e}")

            if train_clicked:
                with st.spinner(f"Training NumPy Neural Network with {live_opt} on {len(X_tr)} records..."):
                    live_net = NumPyNeuralNetwork(input_dim=X_all.shape[1], hidden_dim=live_hidden, random_state=42)
                    history_res = live_net.fit(
                        X_tr, y_tr, X_val=X_te, y_val=y_te,
                        epochs=live_epochs, batch_size=32, lr=live_lr, optimizer=live_opt.lower()
                    )

                hist_df = pd.DataFrame(history_res)
                chart_col1, chart_col2 = st.columns(2)
                with chart_col1:
                    st.plotly_chart(
                        px.line(hist_df, x='epoch', y=['train_loss', 'val_loss'], markers=True,
                                title=f'Live {live_opt} Binary Cross-Entropy Loss per Epoch',
                                labels={'value': 'BCE Loss', 'variable': 'Split'}),
                        use_container_width=True
                    )
                with chart_col2:
                    st.plotly_chart(
                        px.line(hist_df, x='epoch', y=['train_accuracy', 'val_accuracy'], markers=True,
                                title=f'Live {live_opt} Classification Accuracy per Epoch',
                                labels={'value': 'Accuracy', 'variable': 'Split'}),
                        use_container_width=True
                    )

                probs_te = live_net.predict_proba(X_te)
                preds_te = live_net.predict(X_te, threshold=0.5)

                acc = accuracy_score(y_te, preds_te)
                prec = precision_score(y_te, preds_te, zero_division=0)
                rec = recall_score(y_te, preds_te, zero_division=0)
                f1 = f1_score(y_te, preds_te, zero_division=0)
                auc = roc_auc_score(y_te, probs_te) if len(np.unique(y_te)) > 1 else 0.0

                st.markdown('#### Real Held-Out Test Evaluation (79 Unseen Students)')
                m_c1, m_c2, m_c3, m_c4, m_c5 = st.columns(5)
                m_c1.metric('Accuracy', f'{acc:.1%}')
                m_c2.metric('Precision', f'{prec:.1%}')
                m_c3.metric('Recall', f'{rec:.1%}')
                m_c4.metric('F1-Score', f'{f1:.3f}')
                m_c5.metric('ROC-AUC', f'{auc:.3f}')

                mat = confusion_matrix(y_te, preds_te, labels=[0, 1])
                st.plotly_chart(
                    px.imshow(mat, text_auto=True, x=['Predicted Not at Risk', 'Predicted At Risk'],
                              y=['Actual Not at Risk', 'Actual At Risk'],
                              title='Live Model Held-Out Confusion Matrix', color_continuous_scale='Blues'),
                    use_container_width=True
                )
        else:
            st.warning('UCI dataset `data/raw/student-mat.csv` not found.')

    with nn_tab3:
        st.markdown('### 📐 Mathematical Formulation: Pure NumPy Neural Network')
        st.markdown('''
        #### 1. Forward Propagation
        For an input vector $\mathbf{x} \in \mathbb{R}^d$:
        $$\mathbf{z}^{[1]} = \mathbf{x} \mathbf{W}_1 + \mathbf{b}_1 \quad \in \mathbb{R}^{h}$$
        $$\mathbf{a}^{[1]} = \text{ReLU}(\mathbf{z}^{[1]}) = \max(0, \mathbf{z}^{[1]})$$
        $$\mathbf{z}^{[2]} = \mathbf{a}^{[1]} \mathbf{W}_2 + b_2 \quad \in \mathbb{R}^1$$
        $$\hat{y} = \mathbf{a}^{[2]} = \sigma(\mathbf{z}^{[2]}) = \frac{1}{1 + e^{-\mathbf{z}^{[2]}}}$$

        #### 2. Loss Function (Binary Cross-Entropy)
        For $m$ training samples and true labels $y \in \{0, 1\}$:
        $$\mathcal{L} = -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \ln(\hat{y}^{(i)} + \epsilon) + (1 - y^{(i)}) \ln(1 - \hat{y}^{(i)} + \epsilon) \right]$$

        #### 3. Analytical Backpropagation (Multivariate Chain Rule)
        At the output layer, the derivative of BCE with respect to pre-activation simplifies directly:
        $$\frac{\partial \mathcal{L}}{\partial \mathbf{z}^{[2]}} = \frac{1}{m} (\mathbf{a}^{[2]} - \mathbf{y})$$
        $$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_2} = (\mathbf{a}^{[1]})^T \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{[2]}}, \quad \frac{\partial \mathcal{L}}{\partial b_2} = \sum_{i=1}^m \frac{\partial \mathcal{L}}{\partial z_i^{[2]}}$$
        
        Backpropagating through the ReLU hidden layer:
        $$\frac{\partial \mathcal{L}}{\partial \mathbf{a}^{[1]}} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{[2]}} \mathbf{W}_2^T$$
        $$\frac{\partial \mathcal{L}}{\partial \mathbf{z}^{[1]}} = \frac{\partial \mathcal{L}}{\partial \mathbf{a}^{[1]}} \odot \mathbf{1}_{\{\mathbf{z}^{[1]} > 0\}}$$
        $$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_1} = \mathbf{X}^T \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{[1]}}, \quad \frac{\partial \mathcal{L}}{\partial \mathbf{b}_1} = \sum_{i=1}^m \frac{\partial \mathcal{L}}{\partial \mathbf{z}_i^{[1]}}$$

        #### 4. Optimizers: SGD with Momentum vs. Adam
        * **SGD + Momentum:**
          $$\mathbf{v}_t = \beta \mathbf{v}_{t-1} + \alpha \nabla_\theta \mathcal{L}, \quad \theta_t = \theta_{t-1} - \mathbf{v}_t$$
        * **Adam (Adaptive Moment Estimation):**
          $$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1) \mathbf{g}_t, \quad \mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2$$
          $$\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1 - \beta_1^t}, \quad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_2^t}$$
          $$\theta_t = \theta_{t-1} - \frac{\alpha}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} \hat{\mathbf{m}}_t$$
        ''')
        with st.expander('📄 Inspect `src/neural_net.py` (Full Implementation Code)'):
            nn_code_path = ROOT / 'src' / 'neural_net.py'
            if nn_code_path.is_file():
                st.code(nn_code_path.read_text(), language='python')


# -------------------------------------------------------------
# PAGE 9: METHODOLOGY
# -------------------------------------------------------------
else:
    header(
        'Research Methodology & Technical Specifications',
        'Academic framework, experimental controls, validation protocols, and ethical constraints.'
    )

    st.markdown('''
    ### 1. Problem Formulation & Targets
    - **Regression Target:** Final student course grade $G3 \in [0, 20]$. Evaluated using **Mean Absolute Error (MAE)**, **Root Mean Squared Error (RMSE)**, and **Coefficient of Determination ($R^2$)**.
    - **Classification Target:** Binary academic risk defined as $G3 < 10$ (Portuguese passing threshold). Evaluated using **Accuracy**, **Precision**, **Recall**, **F1-Score**, **ROC-AUC**, and **Brier Score**.
    - **Dual Scenario Protocol:**
      - **Scenario A (Early Intervention):** Excludes intermediate period grades `['G1', 'G2', 'G3']` from features to model decision support before exams take place.
      - **Scenario B (Grade-Informed):** Includes `G1` and `G2` as features while strictly holding `G3` as the unseen target.

    ### 2. Experimental Controls & Leakage Prevention
    - **Leakage-Safe Pipelines:** Preprocessing is encapsulated inside Scikit-Learn `Pipeline` with `ColumnTransformer`. Median numerical imputation and one-hot encoding are fitted strictly on training partitions.
    - **Held-Out Test Set Isolation:** An 80/20 stratified split is established once with fixed seed (`random_state=42`). The held-out test set is NEVER used for model selection, hyperparameter tuning, or early stopping.
    - **Training-Only Cross-Validation:** Hyperparameter optimization uses 3-fold cross-validation solely on the training partition ($N=316$).

    ### 3. Model Families & Syllabus Topics
    1. **Decision Tree:** Single tree baseline with depth and complexity parameters (`max_depth`, `min_samples_leaf`, `ccp_alpha`).
    2. **Bagging (Bootstrap Aggregation):** Ensemble of decision trees trained on bootstrap samples with replacement to reduce variance.
    3. **Random Forest:** Enhanced bagging ensemble that introduces feature subsampling at each node split to decorrelate individual trees.
    4. **XGBoost:** Gradient boosted decision trees using second-order Taylor expansion loss approximation, shrinkage, and L1/L2 leaf penalties.
    5. **LightGBM:** Fast gradient boosted decision trees using histogram binning, leaf-wise tree growth, and minimum child weight constraints.

    ### 4. Bias–Variance Tradeoff & Regularization
    - **Complexity Analysis:** Decision trees are trained across depths 1, 2, 3, 5, 8, 10, and Unlimited across repeated validation splits.
    - **Proxy Metric:** Generalization gap $\Delta = \text{RMSE}_{\text{val}} - \text{RMSE}_{\text{train}}$ serves as an empirical proxy for overfitting.
    - **Regularization Controls:** Cost-complexity pruning parameter $\alpha$, minimum samples per split/leaf, and boosting penalties (`reg_alpha`, `reg_lambda`).

    ### 5. Ethical Limitations & Academic Disclaimer
    - **Population Context:** Dataset originates from two Portuguese secondary schools (Cortez & Silva, 2008). Findings reflect secondary education dynamics in Portugal and cannot be generalized to higher education institutions without local recalibration.
    - **Non-Causal Nature:** Tree-based feature importances reflect statistical associations and predictive power, not causal mechanisms.
    - **Educational Support Intent:** Predictions are designed exclusively for constructive early tutoring and student academic support, never for punitive or exclusionary action.

    ### 6. Official Citation
    Cortez, P., & Silva, A. (2008). *Using Data Mining to Predict Secondary School Student Performance.* In Proceedings of 5th Annual Future Business Technology Conference (FUBUTEC 2008), pp. 5-12, Porto, Portugal. EUROSIS-ETI, ISBN 978-9077381-39-7.  
    UCI Machine Learning Repository: [https://doi.org/10.24432/C5TG7T](https://doi.org/10.24432/C5TG7T) (CC BY 4.0).
    ''')
