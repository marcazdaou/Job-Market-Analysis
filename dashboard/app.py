"""
Job Market Analytics Dashboard

This is the third and final step in the pipeline. It reads from the SQLite
database and the trained model, then renders an interactive web dashboard
using Streamlit.

Launch with:
    streamlit run dashboard/app.py
Then open: http://localhost:8501
"""

import streamlit as st
import sqlite3
import pandas as pd
import json
import pickle
import os
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from datetime import datetime, timedelta

# ── File paths ────────────────────────────────────────────────────────────────
# BASE is the directory this file lives in (dashboard/).
# All paths go one level up ("../") to reach the project root.
BASE  = os.path.dirname(__file__)
DB    = os.path.join(BASE, "../data/jobs.db")              # SQLite database
MODEL = os.path.join(BASE, "../models/salary_model.pkl")   # trained model
META  = os.path.join(BASE, "../models/model_meta.json")    # model metrics

# ── Page configuration ────────────────────────────────────────────────────────
# This must be the first Streamlit call in the file.
# layout="wide" uses the full browser width instead of a narrow centered column.
st.set_page_config(
    page_title="Job Market Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS styling ────────────────────────────────────────────────────────
# Streamlit renders a React app in the browser. We inject raw CSS to override
# default styles. unsafe_allow_html=True is required to render HTML/CSS strings.
st.markdown("""
<style>
    /* Main page background */
    .main { background-color: #0f1117; }

    /* KPI stat cards at the top of the dashboard */
    .metric-card {
        background: linear-gradient(135deg, #0f2744, #0d3260);
        border: 1px solid #1d6fa4;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #60a5fa; }
    .metric-label { font-size: 0.85rem; color: #94a3b8; margin-top: 4px; }

    /* Blue left-border section title style */
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #e2e8f0;
        border-left: 4px solid #60a5fa;
        padding-left: 12px;
        margin: 24px 0 16px 0;
    }

    /* Sidebar background — both selectors needed for Streamlit version compatibility */
    .stSidebar { background-color: #1a3a5c; }
    section[data-testid="stSidebar"] { background-color: #1a3a5c; }

    /* Sidebar labels and headings */
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4,
    section[data-testid="stSidebar"] span { color: #ffffff !important; }

    /* Selectbox — dark background, white text, blue border */
    section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {
        background-color: #0f2744 !important;
        border-color: #60a5fa !important;
        color: #ffffff !important;
    }
    section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] svg {
        fill: #ffffff !important;
    }

    /* Radio buttons */
    section[data-testid="stSidebar"] .stRadio div { color: #ffffff !important; }
</style>
""", unsafe_allow_html=True)


# ── Cached data loading functions ─────────────────────────────────────────────
# @st.cache_data caches the return value so the function only re-runs when
# the data changes (or after ttl seconds). Without caching, the database would
# be re-queried on every user interaction, making the dashboard slow.

@st.cache_data(ttl=300)  # cache for 5 minutes (300 seconds)
def load_jobs():
    """
    Loads all job records from SQLite into a pandas DataFrame.
    Also parses the skills column (stored as a JSON string) into a Python list,
    and converts the date_posted column from a string to a datetime object.
    Both conversions are needed by the charts and filters below.
    """
    conn = sqlite3.connect(DB)
    df = pd.read_sql("SELECT * FROM jobs", conn)
    conn.close()
    # skills is stored as a JSON string like '["Python", "SQL", "AWS"]'
    # json.loads converts it back to a Python list for easy processing
    df["skills_list"] = df["skills"].apply(lambda s: json.loads(s) if s else [])
    # Convert "2024-03-15" string to a proper datetime for date comparisons
    df["date_posted"] = pd.to_datetime(df["date_posted"])
    return df


@st.cache_data(ttl=300)
def load_aggregated():
    """
    Runs SQL queries with GROUP BY, HAVING, and CASE statements to pre-compute
    global market statistics. These are used for charts that show overall trends
    rather than filtered data.

    We use SQL here instead of pandas because:
    - It demonstrates proper use of SQL aggregation (relevant for data engineering roles)
    - GROUP BY and HAVING in SQL are more efficient for these specific summaries
    - The results are independent of the sidebar filters (global market view)

    Returns three DataFrames:
    - title_stats:       avg/min/max salary per job title
    - location_stats:    posting count and avg salary per location (min 10 postings)
    - experience_bands:  avg salary grouped into 4 experience bands
    """
    conn = sqlite3.connect(DB)

    # Average, min, and max salary per job title across all records
    title_stats = pd.read_sql("""
        SELECT
            title,
            COUNT(*)                              AS postings,
            ROUND(AVG(salary_avg), 0)             AS avg_salary,
            ROUND(MIN(salary_avg), 0)             AS min_salary,
            ROUND(MAX(salary_avg), 0)             AS max_salary,
            ROUND(AVG(experience), 1)             AS avg_experience
        FROM jobs
        GROUP BY title
        ORDER BY avg_salary DESC
    """, conn)

    # Top 12 locations by posting count; HAVING filters out locations with < 10 postings
    # to avoid showing locations with just 1-2 records (statistically unreliable)
    location_stats = pd.read_sql("""
        SELECT
            location,
            COUNT(*)                                    AS postings,
            ROUND(AVG(salary_avg), 0)                  AS avg_salary,
            ROUND(SUM(remote) * 100.0 / COUNT(*), 1)   AS remote_pct
        FROM jobs
        GROUP BY location
        HAVING COUNT(*) >= 10
        ORDER BY postings DESC
        LIMIT 12
    """, conn)

    # CASE statement groups continuous experience years into 4 readable bands.
    # ORDER BY MIN(experience) sorts bands in logical order (0-2, 3-5, 6-9, 10+)
    # instead of alphabetical order which would be wrong.
    experience_bands = pd.read_sql("""
        SELECT
            CASE
                WHEN experience <= 2  THEN '0-2 yrs'
                WHEN experience <= 5  THEN '3-5 yrs'
                WHEN experience <= 9  THEN '6-9 yrs'
                ELSE                       '10+ yrs'
            END                           AS band,
            COUNT(*)                      AS postings,
            ROUND(AVG(salary_avg), 0)     AS avg_salary
        FROM jobs
        GROUP BY band
        ORDER BY MIN(experience)
    """, conn)

    conn.close()
    return title_stats, location_stats, experience_bands


@st.cache_resource  # cache_resource (not cache_data) is used for large objects like ML models
def load_model():
    """
    Loads the trained model from disk. Returns (None, None) if the model file
    does not exist yet, so the dashboard can show a helpful message instead of crashing.

    The model file contains a dict with:
    - "model":        the fitted scikit-learn model object
    - "model_name":   which model won the comparison (e.g. "Random Forest")
    - "feature_cols": the exact list of column names the model was trained on
    """
    if not os.path.exists(MODEL):
        return None, None
    with open(MODEL, "rb") as f:
        obj = pickle.load(f)
    # Load the companion JSON file with performance metrics and comparison results
    meta = json.load(open(META)) if os.path.exists(META) else {}
    return obj, meta


# ── Sidebar filters ───────────────────────────────────────────────────────────
# Everything inside "with st.sidebar:" renders in the left panel.
# The user sets these values and they filter the dataset shown in the main area.
with st.sidebar:
    st.markdown("## Filters")
    df_all = load_jobs()  # full unfiltered dataset — always loaded for filter options

    # Dropdown to filter by a specific job title, or show all titles
    title_opts = ["All"] + sorted(df_all["title"].unique().tolist())
    sel_title  = st.selectbox("Job Title", title_opts)

    # Dropdown to filter by city/location
    loc_opts = ["All"] + sorted(df_all["location"].unique().tolist())
    sel_loc  = st.selectbox("Location", loc_opts)

    # Radio button: show all jobs, only remote, or only on-site
    remote_opt = st.radio("Work Type", ["All", "Remote", "On-site"])

    # Slider to set the minimum and maximum salary to show.
    # The range is determined by the actual min/max in the database.
    sal_min = int(df_all["salary_avg"].min())
    sal_max = int(df_all["salary_avg"].max())
    sel_sal = st.slider("Salary Range ($)", sal_min, sal_max, (sal_min, sal_max), step=5000)

    st.markdown("---")
    st.markdown("#### Date Range")
    # How many days back to show postings (default: last 30 days)
    days_back = st.slider("Days back", 7, 60, 30)
    cutoff = datetime.now() - timedelta(days=days_back)


# ── Apply filters ─────────────────────────────────────────────────────────────
# Start from the full dataset and progressively narrow it down based on
# what the user selected in the sidebar. All charts below use this filtered "df".
df = df_all.copy()
if sel_title != "All":
    df = df[df["title"] == sel_title]
if sel_loc != "All":
    df = df[df["location"] == sel_loc]
if remote_opt == "Remote":
    df = df[df["remote"] == 1]
elif remote_opt == "On-site":
    df = df[df["remote"] == 0]
df = df[(df["salary_avg"] >= sel_sal[0]) & (df["salary_avg"] <= sel_sal[1])]
df = df[df["date_posted"] >= pd.Timestamp(cutoff)]

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("# Job Market Analytics Dashboard")
st.markdown(
    "*Market intelligence for Data Engineering and Analytics roles* &nbsp;|&nbsp; "
    "Marc Abi Zeid Daou &nbsp;|&nbsp; "
    "[Live Demo](https://job-market-analysis-ndmqelnezlfa64hyyhhwkv.streamlit.app) &nbsp;|&nbsp; "
    "[GitHub](https://github.com/marcazdaou/Job-Market-Analysis)"
)
st.markdown("---")

# Stop rendering if the filters result in zero records — avoids blank/broken charts
if df.empty:
    st.warning("No jobs match the selected filters. Try broadening your search.")
    st.stop()

# ── KPI metric cards ──────────────────────────────────────────────────────────
# Four summary numbers at the top of the dashboard.
# st.columns(4) creates four equal-width columns side by side.
col1, col2, col3, col4 = st.columns(4)

with col1:
    # Total number of job postings matching the current filters
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{len(df):,}</div>
        <div class="metric-label">Total Postings</div>
    </div>""", unsafe_allow_html=True)

with col2:
    # Average salary across all filtered postings
    avg_sal = df["salary_avg"].mean()
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">${avg_sal:,.0f}</div>
        <div class="metric-label">Average Salary</div>
    </div>""", unsafe_allow_html=True)

with col3:
    # Percentage of filtered postings that allow remote work
    remote_pct = df["remote"].mean() * 100
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{remote_pct:.1f}%</div>
        <div class="metric-label">Remote Positions</div>
    </div>""", unsafe_allow_html=True)

with col4:
    # Number of unique company names in the filtered results
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{df['company'].nunique()}</div>
        <div class="metric-label">Companies Hiring</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Row 1: Salary by title + Top skills ──────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.markdown('<div class="section-header">Average Salary by Job Title</div>', unsafe_allow_html=True)
    # Group by title, compute mean salary, sort ascending so highest is at top of horizontal bar chart
    sal_data = (
        df.groupby("title")["salary_avg"]
        .mean()
        .sort_values(ascending=True)
        .reset_index()
    )
    fig = px.bar(
        sal_data, x="salary_avg", y="title", orientation="h",
        color="salary_avg", color_continuous_scale="Blues",  # darker = higher salary
        labels={"salary_avg": "Avg Salary ($)", "title": ""},
        template="plotly_dark"
    )
    fig.update_layout(
        height=420, showlegend=False, coloraxis_showscale=False,
        margin=dict(l=0, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
    )
    fig.update_traces(marker_line_width=0)  # remove bar outlines for a cleaner look
    st.plotly_chart(fig, use_container_width=True)

with col_b:
    st.markdown('<div class="section-header">Top In-Demand Skills</div>', unsafe_allow_html=True)
    # Flatten all skills from all filtered jobs into one big list, then count occurrences.
    # The list comprehension loops over each row's skills_list and extends a flat list.
    all_skills = [s for sl in df["skills_list"] for s in sl]
    skill_counts = Counter(all_skills).most_common(20)  # top 20 most common skills
    skill_df = pd.DataFrame(skill_counts, columns=["skill", "count"])
    fig2 = px.bar(
        skill_df, x="count", y="skill", orientation="h",
        color_discrete_sequence=["#a78bfa"],  # single color so all bars are equally visible
        labels={"count": "Job Postings", "skill": ""},
        template="plotly_dark"
    )
    fig2.update_layout(
        height=420, showlegend=False,
        margin=dict(l=0, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── Row 2: Location + Salary distribution ─────────────────────────────────────
col_c, col_d = st.columns(2)

with col_c:
    st.markdown('<div class="section-header">Postings by Location</div>', unsafe_allow_html=True)
    # Group by location, count postings and compute average salary per location.
    # Color each bar by average salary so you can see which cities pay more.
    loc_data = (
        df.groupby("location")
        .agg(count=("id", "count"), avg_salary=("salary_avg", "mean"))
        .reset_index()
        .sort_values("count", ascending=False)
        .head(12)  # only show top 12 to avoid crowding the x-axis
    )
    fig3 = px.bar(
        loc_data, x="location", y="count",
        color="avg_salary", color_continuous_scale="Teal",
        labels={"count": "Postings", "location": "", "avg_salary": "Avg Salary ($)"},
        template="plotly_dark"
    )
    fig3.update_layout(
        height=360,
        margin=dict(l=0, r=10, t=10, b=60),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis_tickangle=-35  # angle the labels so they don't overlap
    )
    st.plotly_chart(fig3, use_container_width=True)

with col_d:
    st.markdown('<div class="section-header">Salary Distribution</div>', unsafe_allow_html=True)
    # Histogram showing how salaries are distributed across all filtered postings.
    # nbins=40 means we divide the salary range into 40 buckets.
    fig4 = px.histogram(
        df, x="salary_avg", nbins=40,
        color_discrete_sequence=["#7c8dfc"],
        labels={"salary_avg": "Annual Salary ($)"},
        template="plotly_dark"
    )
    # Add a dashed vertical line at the mean salary for reference
    fig4.add_vline(
        x=df["salary_avg"].mean(), line_dash="dash", line_color="#f59e0b",
        annotation_text=f"Mean: ${df['salary_avg'].mean():,.0f}",
        annotation_position="top right"
    )
    fig4.update_layout(
        height=360,
        margin=dict(l=0, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig4, use_container_width=True)

# ── Row 3: Postings over time + Experience vs Salary ─────────────────────────
col_e, col_f = st.columns(2)

with col_e:
    st.markdown('<div class="section-header">Postings Over Time</div>', unsafe_allow_html=True)
    # Count how many jobs were posted on each date, then plot as an area chart.
    # This shows whether the job market is growing or shrinking over the period.
    time_df = df.groupby("date_posted").size().reset_index(name="count")
    fig5 = px.area(
        time_df, x="date_posted", y="count",
        color_discrete_sequence=["#7c8dfc"],
        labels={"date_posted": "Date", "count": "New Postings"},
        template="plotly_dark"
    )
    fig5.update_layout(
        height=320,
        margin=dict(l=0, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig5, use_container_width=True)

with col_f:
    st.markdown('<div class="section-header">Experience vs. Salary</div>', unsafe_allow_html=True)
    # Scatter plot: each dot is one job posting.
    # X = years of experience required, Y = salary.
    # Color = job title, so you can see different roles' salary curves.
    # We sample up to 500 points to keep the chart fast and readable.
    fig6 = px.scatter(
        df.sample(min(500, len(df))),
        x="experience", y="salary_avg",
        color="title", size_max=8, opacity=0.7,
        labels={"experience": "Years of Experience", "salary_avg": "Salary ($)", "title": "Role"},
        template="plotly_dark"
    )
    fig6.update_layout(
        height=320,
        margin=dict(l=0, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(size=9))
    )
    st.plotly_chart(fig6, use_container_width=True)

# ── Skill Salary Premium ──────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">Skill Salary Premium</div>', unsafe_allow_html=True)
st.markdown(
    "For each skill, the bar shows the average salary across all postings that require it. "
    "Skills appearing in fewer than 30 postings are excluded to ensure statistical reliability."
)

# Build a flat list of (skill, salary) pairs — one row per skill per job posting.
# Example: if a job requires Python and SQL and pays $130k,
# we add {"skill": "Python", "salary": 130000} and {"skill": "SQL", "salary": 130000}.
skill_salary_rows = [
    {"skill": skill, "salary": row["salary_avg"]}
    for _, row in df.iterrows()
    for skill in row["skills_list"]
]
if skill_salary_rows:
    skill_sal_df = pd.DataFrame(skill_salary_rows)
    premium = (
        skill_sal_df.groupby("skill")["salary"]
        .agg(avg_salary="mean", count="count")
        .reset_index()
        .query("count >= 30")          # exclude rare skills (unreliable average)
        .sort_values("avg_salary", ascending=True)
        .tail(20)                      # show top 20 highest-paying skills
    )
    fig_prem = px.bar(
        premium, x="avg_salary", y="skill", orientation="h",
        color="avg_salary", color_continuous_scale="Greens",
        labels={"avg_salary": "Avg Salary in Postings Requiring Skill ($)", "skill": ""},
        template="plotly_dark"
    )
    fig_prem.update_layout(
        height=480, showlegend=False, coloraxis_showscale=False,
        margin=dict(l=0, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_prem, use_container_width=True)

# ── Salary by Experience Band (SQL-aggregated global view) ────────────────────
# This chart uses the pre-computed SQL aggregation from load_aggregated().
# It is NOT filtered by the sidebar — it always shows the global dataset.
# That is intentional: it gives a market-wide view of how salary grows with experience.
_, _, experience_bands = load_aggregated()
st.markdown('<div class="section-header">Salary by Experience Band</div>', unsafe_allow_html=True)
st.markdown("Global market view — shows average salary across all postings grouped by experience level.")

fig_bands = px.bar(
    experience_bands, x="band", y="avg_salary",
    color="avg_salary", color_continuous_scale="Blues",
    text="postings",  # show posting count on top of each bar
    labels={"band": "Experience Level", "avg_salary": "Average Salary ($)", "postings": "Postings"},
    template="plotly_dark"
)
fig_bands.update_traces(texttemplate="%{text} postings", textposition="outside")
fig_bands.update_layout(
    height=320, showlegend=False, coloraxis_showscale=False,
    margin=dict(l=0, r=10, t=30, b=10),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
)
st.plotly_chart(fig_bands, use_container_width=True)

# ── Salary Estimator ──────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## Salary Estimator")
st.markdown("Enter a candidate profile to generate a salary estimate using the trained model.")

model_obj, model_meta = load_model()

# Display model performance metrics above the estimator form
if model_meta:
    best_name = model_meta.get("best_model", "Gradient Boosting")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Selected Model", best_name)
    m2.metric("Test MAE", f"${model_meta.get('mae', 0):,.0f}")      # average dollar error on test set
    m3.metric("Test R²", f"{model_meta.get('r2', 0):.3f}")          # 1.0 = perfect, 0 = no better than mean
    m4.metric("CV R²", f"{model_meta.get('cv_r2_mean', 0):.3f} +/- {model_meta.get('cv_r2_std', 0):.3f}")

if model_obj:
    model     = model_obj["model"]       # the fitted scikit-learn model
    feat_cols = model_obj["feature_cols"]  # ordered list of feature names used during training

    # st.form groups widgets so they only trigger a rerun when the submit button is clicked,
    # not on every individual widget change
    with st.form("predict_form"):
        pc1, pc2, pc3 = st.columns(3)
        with pc1:
            p_title = st.selectbox("Job Title", sorted(df_all["title"].unique()))
            p_exp   = st.slider("Years of Experience", 0, 15, 3)
        with pc2:
            p_loc    = st.selectbox("Location", sorted(df_all["location"].unique()))
            p_remote = st.checkbox("Remote Position", value=False)
        with pc3:
            p_skills = st.multiselect(
                "Skills",
                options=sorted({s for sl in df_all["skills_list"] for s in sl}),
                default=["Python", "SQL", "AWS"]
            )

        submitted = st.form_submit_button("Estimate Salary", use_container_width=True)

    if submitted:
        # Build the feature vector the model expects.
        # Start with all zeros (assuming no skills, no location, etc.)
        # then set the relevant fields to 1 or the actual value.
        row = {c: 0 for c in feat_cols}
        row["experience"]  = p_exp
        row["remote"]      = int(p_remote)
        row["skill_count"] = len(p_skills)

        # Assign seniority score the same way train.py does
        if any(x in p_title for x in ["Principal", "Staff"]):
            row["seniority"] = 4
        elif any(x in p_title for x in ["Senior", "Lead"]):
            row["seniority"] = 3
        else:
            row["seniority"] = 2

        # Turn on the location flag for the selected city (if it was in top 10 during training)
        safe_loc = p_loc.replace(" ", "_").replace(",", "")
        if f"loc_{safe_loc}" in row:
            row[f"loc_{safe_loc}"] = 1

        # Turn on each selected skill's binary flag column
        for sk in p_skills:
            col_name = f"skill_{sk.replace(' ', '_').replace('-', '_')}"
            if col_name in row:
                row[col_name] = 1

        # Convert to a DataFrame in the exact column order the model was trained on
        X_input = pd.DataFrame([row])[feat_cols].fillna(0)
        pred    = model.predict(X_input)[0]  # returns an array; [0] gets the single value

        # Show three versions: the raw prediction plus +/- 10% bounds
        res1, res2, res3 = st.columns(3)
        res1.metric("Predicted Salary",       f"${pred:,.0f}")
        res2.metric("Conservative Estimate",  f"${pred * 0.9:,.0f}")
        res3.metric("Optimistic Estimate",    f"${pred * 1.1:,.0f}")

        # Skill gap analysis: find the top 10 most common skills across all jobs
        # and highlight which ones the user did not select
        all_sk  = [s for sl in df_all["skills_list"] for s in sl]
        top_sk  = [s for s, _ in Counter(all_sk).most_common(10)]
        missing = [s for s in top_sk if s not in p_skills]
        if missing:
            st.info(f"High-demand skills not in profile: **{', '.join(missing[:5])}**")

    # ── Model Comparison chart ────────────────────────────────────────────────
    # Shows how all three candidate models performed so the viewer can see why
    # the winning model was chosen. Data comes from model_meta.json.
    comparison = model_meta.get("model_comparison") if model_meta else None
    if comparison:
        st.markdown("---")
        st.markdown("### Model Comparison")
        st.markdown(
            "Three regression models were trained and evaluated using 5-fold cross-validation. "
            "The model with the highest mean CV R² was selected to prevent overfitting on the test set."
        )

        # Convert the nested comparison dict into a DataFrame for easy charting
        comp_df = (
            pd.DataFrame(comparison)
            .T.reset_index()
            .rename(columns={"index": "Model"})
            .sort_values("cv_r2_mean", ascending=True)  # ascending so best model appears at top of horizontal bar
        )

        # Horizontal bar chart: one bar per model, length = CV R², error bars = CV std dev
        fig_comp = px.bar(
            comp_df, x="cv_r2_mean", y="Model", orientation="h",
            error_x="cv_r2_std",  # error bars show how much scores varied across the 5 folds
            color="cv_r2_mean", color_continuous_scale="Blues",
            labels={"cv_r2_mean": "Mean CV R² Score", "Model": ""},
            template="plotly_dark"
        )
        fig_comp.update_layout(
            height=260, showlegend=False, coloraxis_showscale=False,
            margin=dict(l=0, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        # Formatted table showing all metrics for all three models
        display_comp = comp_df[["Model", "mae", "r2", "cv_r2_mean", "cv_r2_std", "train_time_s"]].copy()
        display_comp.columns = ["Model", "MAE ($)", "R²", "CV R² Mean", "CV R² Std", "Train Time (s)"]
        display_comp["MAE ($)"] = display_comp["MAE ($)"].apply(lambda x: f"${float(x):,.0f}")
        st.dataframe(display_comp, use_container_width=True, hide_index=True)

else:
    # Model files don't exist yet — tell the user what to run
    st.info("Run `python models/train.py` to train the salary prediction model.")

# ── Raw data table ────────────────────────────────────────────────────────────
# Collapsible section — hidden by default to avoid overwhelming the dashboard.
# Shows the actual filtered rows so the user can inspect individual postings.
st.markdown("---")
with st.expander("View Raw Job Data"):
    display_df = df[["title", "company", "location", "salary_avg", "experience", "remote", "date_posted"]].copy()
    display_df["salary_avg"] = display_df["salary_avg"].apply(lambda x: f"${x:,.0f}")
    display_df["remote"]     = display_df["remote"].map({1: "Yes", 0: "No"})
    display_df.columns       = ["Title", "Company", "Location", "Avg Salary", "Experience (yrs)", "Remote", "Date Posted"]
    st.dataframe(display_df.sort_values("Date Posted", ascending=False), use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><small>"
    "Marc Abi Zeid Daou &nbsp;|&nbsp; Job Market Analytics Dashboard &nbsp;|&nbsp; "
    "Python · Streamlit · Scikit-learn"
    "</small></center>",
    unsafe_allow_html=True
)
