"""Step 3 of 3: the public Streamlit dashboard.

Reads the committed artifacts only, so the hosted app needs no setup:
    data/jobs_clean.parquet   cleaned postings      (pipeline/collector.py)
    data/data_summary.json    cleaning funnel       (pipeline/collector.py)
    models/salary_model.pkl   trained salary model  (models/train.py)
    models/model_meta.json    evaluation results    (models/train.py)

Layout: a header, sidebar filters, then six tabs:
    Overview · Salaries · Skills · Explore a role · Salary predictor (ML) · About the data

Run locally:  streamlit run dashboard/app.py
"""
import json
import pickle
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
from features import OTHER_FAMILY, SENIORITY_ORDER, model_frame, role_family, seniority  # noqa: E402

DATA = ROOT / "data" / "jobs_clean.parquet"
SUMMARY = ROOT / "data" / "data_summary.json"
MODEL = ROOT / "models" / "salary_model.pkl"
META = ROOT / "models" / "model_meta.json"
STATS = ROOT / "analysis" / "results" / "statistics.json"          # analysis/salary_statistics.py
LABEL_CHECK = ROOT / "analysis" / "results" / "label_validation.json"  # analysis/label_validation.py

AUTHOR = "Marc Abi Zeid Daou"
LINKS = {"GitHub": "https://github.com/marcazdaou/Job-Market-Analysis",
         "LinkedIn": "https://www.linkedin.com/in/marc-abi-zeid-daou/",
         "Portfolio": "https://marcazdaou.github.io/my-portfolio/"}

MIN_GROUP = 30  # medians from fewer salaried postings than this are hidden (too noisy)

st.set_page_config(page_title="Job Market Analytics", layout="wide")

# ══ Visual style ══════════════════════════════════════════════════════════════
# Colors come from a palette validated for color-vision deficiency. Blue is the
# single-series color; orange is only used as the second series in comparisons.
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, CARD = "#e9e8e3", "#c3c2b7", "#ffffff"
BLUE, BLUE_LIGHT, ORANGE, RED = "#2a78d6", "#b7d3f6", "#eb6834", "#e34948"
NAVY = "#0d366b"
BLUE_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]  # ordinal: light -> dark
FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'

# One Plotly template shared by every chart: hairline grid, white card surface,
# muted axis text, and a clean white tooltip.
_axis = dict(gridcolor=GRID, linecolor=AXIS, zeroline=False, ticks="",
             tickfont=dict(color=MUTED, size=12), title=dict(font=dict(color=INK_2, size=12)))
pio.templates["jobs"] = go.layout.Template(layout=dict(
    font=dict(family=FONT, size=13, color=INK_2),
    paper_bgcolor=CARD, plot_bgcolor=CARD,
    xaxis=_axis, yaxis=_axis, colorway=[BLUE, ORANGE],
    hoverlabel=dict(bgcolor="white", bordercolor=GRID, font=dict(family=FONT, color=INK, size=13)),
    margin=dict(l=8, r=24, t=8, b=8), bargap=0.3, barcornerradius=4, showlegend=False,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(color=INK_2)),
))
pio.templates.default = "jobs"
PLOT_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"]}

# Page-level CSS: soft background, a gradient header, white "cards" around charts.
st.markdown(f"""
<style>
  .stApp {{ background: #f7f7f5; }}
  .block-container {{ padding-top: 3.2rem; max-width: 1300px; }}
  header[data-testid="stHeader"] {{ background: transparent; }}
  h1, h2, h3 {{ color: {INK}; letter-spacing: -0.01em; }}

  /* Header: navy-to-teal gradient panel. */
  .masthead {{ text-align: center; padding: 40px 28px 30px; margin-bottom: 10px; border-radius: 14px;
               background: linear-gradient(120deg, #0d2340 0%, #163f6e 50%, #1b6a80 100%); }}
  .masthead .eyebrow {{ font-size: 0.75rem; letter-spacing: 0.2em; text-transform: uppercase;
                        color: #a9c3dd; font-weight: 600; }}
  .masthead h1 {{ font-size: 2.45rem; font-weight: 750; line-height: 1.18; margin: 12px auto 12px;
                  max-width: 880px; padding: 0; color: #ffffff; }}
  .masthead .num {{ color: #7cc4ff; }}   /* headline number in sky blue */
  .masthead .sub {{ font-size: 1.04rem; color: #d6e2ee; max-width: 740px; margin: 0 auto; line-height: 1.55; }}

  /* Key stats: plain number/label pairs, no lines or boxes */
  .masthead .stats {{ display: flex; justify-content: center; flex-wrap: wrap; gap: 6px 44px;
                      margin: 26px auto 0; max-width: 880px; }}
  .masthead .stat .v {{ font-size: 1.3rem; font-weight: 700; color: #ffffff; }}
  .masthead .stat .l {{ font-size: 0.72rem; color: #a9c3dd; text-transform: uppercase; letter-spacing: 0.08em; }}

  /* Author name in warm gold so it stands apart from the blue background */
  .masthead .author {{ margin-top: 26px; font-size: 1.12rem; color: #f2c66d; font-weight: 700;
                       letter-spacing: 0.02em; }}
  .masthead .links {{ margin-top: 10px; display: flex; justify-content: center; gap: 10px; flex-wrap: wrap; }}
  .masthead .links a {{ color: #0d2340; text-decoration: none; font-size: 0.86rem; font-weight: 650;
                        padding: 6px 16px; border-radius: 6px; background: #ffffff; }}
  .masthead .links a:hover {{ background: #f2c66d; }}
  @media (max-width: 640px) {{   /* phones: smaller headline and tighter stats */
    .masthead {{ padding: 28px 16px 22px; }}
    .masthead h1 {{ font-size: 1.75rem; }}
    .masthead .stats {{ gap: 6px 22px; }}
  }}

  /* "What's in this data" boxes: light blue panel with a blue left edge */
  .scope {{ background: #eef4fb; border-left: 4px solid {BLUE}; border-radius: 8px; padding: 14px 16px;
            height: 100%; }}
  .scope .q {{ font-size: 0.76rem; text-transform: uppercase; letter-spacing: 0.08em; color: {NAVY};
               font-weight: 700; }}
  .scope .a {{ font-size: 1.02rem; font-weight: 700; color: {INK}; margin: 4px 0 4px; }}
  .scope .d {{ font-size: 0.85rem; color: {INK_2}; line-height: 1.45; }}

  /* Key findings ("the story") */
  .finding {{ background: {CARD}; border: 1px solid rgba(11,11,11,0.07); border-radius: 14px;
              padding: 16px 18px; height: 100%; box-shadow: 0 1px 3px rgba(16,24,40,0.05); }}
  .finding .q {{ font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em; color: {BLUE};
                 font-weight: 650; }}
  .finding .a {{ font-size: 1.35rem; font-weight: 700; color: {INK}; margin: 6px 0 4px; line-height: 1.25; }}
  .finding .d {{ font-size: 0.86rem; color: {INK_2}; line-height: 1.45; }}
  .finding .go {{ font-size: 0.78rem; color: {MUTED}; margin-top: 8px; }}

  /* Cards: every st.container(border=True) becomes a white rounded card */
  div[data-testid="stVerticalBlockBorderWrapper"],
  div[data-testid="stVerticalBlock"][class*="border"] {{
      background: {CARD}; border-radius: 14px !important; border-color: rgba(11,11,11,0.07) !important;
      box-shadow: 0 1px 3px rgba(16,24,40,0.05); }}

  /* KPI tiles */
  .kpi {{ background: {CARD}; border: 1px solid rgba(11,11,11,0.07); border-radius: 14px;
          padding: 14px 16px; box-shadow: 0 1px 3px rgba(16,24,40,0.05); height: 100%; }}
  .kpi-value {{ font-size: 1.6rem; font-weight: 650; color: {INK}; line-height: 1.2; }}
  .kpi-label {{ font-size: 0.82rem; color: {INK_2}; margin-top: 2px; }}
  .kpi-note  {{ font-size: 0.74rem; color: {MUTED}; }}

  .chart-title {{ font-weight: 650; color: {INK}; font-size: 1.02rem; }}
  .chart-sub {{ color: {INK_2}; font-size: 0.84rem; margin: 2px 0 6px 0; }}

  /* Tabs: larger, spaced labels */
  button[data-baseweb="tab"] p {{ font-size: 0.98rem; font-weight: 550; }}
  div[data-baseweb="tab-list"] {{ gap: 18px; }}
  section[data-testid="stSidebar"] {{ background: #ffffff; }}
  .footer {{ text-align: center; color: {MUTED}; font-size: 0.82rem; margin: 28px 0 8px; }}
  .footer a {{ color: {INK_2}; }}
</style>""", unsafe_allow_html=True)


# ══ Data loading (cached once per server process) ═════════════════════════════
@st.cache_resource
def load_jobs() -> pd.DataFrame:
    jobs = pd.read_parquet(DATA)
    jobs["title_lc"] = jobs["title"].str.lower()  # pre-lowered once for fast title search
    return jobs


@st.cache_resource
def load_skills_long() -> pd.DataFrame:
    """One row per (posting, skill): turns skill questions into fast group-bys."""
    jobs = load_jobs()
    long = jobs[["skills", "salary_mid", "role_family"]].explode("skills").dropna(subset=["skills"])
    return long.rename(columns={"skills": "skill"}).rename_axis("row").reset_index()


@st.cache_resource
def load_model():
    if not MODEL.exists() or not META.exists():
        return None, None
    with MODEL.open("rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], json.loads(META.read_text(encoding="utf-8"))


@st.cache_resource
def iso3_lookup() -> dict:
    """Country name -> ISO-3 code for the world map.

    Plotly's bundled gapminder table covers most countries; the rest of the
    countries in this dataset are added by hand.
    """
    import plotly.express as px
    table = px.data.gapminder()[["country", "iso_alpha"]].drop_duplicates()
    lookup = dict(zip(table["country"], table["iso_alpha"]))
    lookup.update({"United Arab Emirates": "ARE", "South Korea": "KOR", "Czechia": "CZE", "Ukraine": "UKR",
                   "Hong Kong": "HKG", "Estonia": "EST", "Lithuania": "LTU", "Latvia": "LVA", "Slovakia": "SVK"})
    return lookup


@st.cache_resource
def load_summary() -> dict:
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


@st.cache_resource
def load_json(path: Path) -> dict:
    """Optional result file (statistics, label accuracy); empty dict if it hasn't been produced."""
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


# ══ Small helpers ═════════════════════════════════════════════════════════════
def usd(x: float) -> str:
    """Compact dollars for chart labels: 123456 -> '$123k' (halves round up, e.g. 82,500 -> '$83k')."""
    return f"${int(x / 1000 + 0.5):,}k"


@contextmanager
def card(title: str, subtitle: str = ""):
    """White card with a title; everything inside the `with` block renders in it."""
    with st.container(border=True):
        st.markdown(f'<div class="chart-title">{title}</div>', unsafe_allow_html=True)
        if subtitle:
            st.markdown(f'<div class="chart-sub">{subtitle}</div>', unsafe_allow_html=True)
        yield


def show(fig: go.Figure, table: pd.DataFrame | None = None):
    """Render a chart, followed by its data as a collapsible table."""
    st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG)
    if table is not None:
        with st.expander("View data table"):
            st.dataframe(table, width="stretch", hide_index=True)


def kpi(col, value: str, label: str, note: str = ""):
    col.markdown(f'<div class="kpi"><div class="kpi-value">{value}</div>'
                 f'<div class="kpi-label">{label}</div><div class="kpi-note">{note}</div></div>',
                 unsafe_allow_html=True)


def ranked_bar(df: pd.DataFrame, value: str, label: str, hover: str, height: int, headroom: float = 1.18):
    """Horizontal single-color bar chart, largest value on top.

    An optional "_text" column is drawn at the bar ends. Every other column
    (except the label) is passed to the tooltip as customdata, in column order,
    so tooltips can show extra facts instead of repeating the visible label.
    """
    df = df.sort_values(value)
    colors = np.where(df[label].astype(str).eq("Other"), AXIS, BLUE)  # catch-all group in gray
    fig = go.Figure(go.Bar(
        x=df[value], y=df[label], orientation="h", marker_color=colors,
        text=df["_text"] if "_text" in df else None, textposition="outside",
        textfont=dict(color=INK_2, size=12), cliponaxis=False,
        customdata=df.drop(columns=[label, "_text"], errors="ignore"),
        hovertemplate=hover + "<extra></extra>"))
    fig.update_layout(height=height, yaxis=dict(gridcolor="rgba(0,0,0,0)"),
                      xaxis=dict(range=[0, df[value].max() * headroom]))
    return fig


def salary_stats(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Count, quartiles and median salary per group; small groups are dropped."""
    g = df.dropna(subset=["salary_mid"]).groupby(by, observed=True)["salary_mid"]
    stats = pd.DataFrame({"n": g.size(), "p25": g.quantile(0.25),
                          "median": g.median(), "p75": g.quantile(0.75)}).reset_index()
    return stats[stats["n"] >= MIN_GROUP]


def ci_text(ci: dict | None, key) -> str:
    """'95% CI $155k–$165k' for a group from the bootstrap results, or '' if unavailable."""
    item = (ci or {}).get(str(key))
    return f"95% CI {usd(item['ci_low'])}–{usd(item['ci_high'])}" if item else ""


def median_iqr_chart(stats: pd.DataFrame, label: str, height: int, ci: dict | None = None):
    """Median (dot) with the middle 50% of salaries (bar): typical pay AND spread.

    `ci` (optional) maps each group to its bootstrap confidence interval for the
    median; it is shown in the tooltip.
    """
    stats = stats.sort_values("median").copy()
    stats["ci"] = stats[label].map(lambda k: ci_text(ci, k))
    fig = go.Figure()
    fig.add_bar(x=stats["p75"] - stats["p25"], base=stats["p25"], y=stats[label], orientation="h",
                marker_color=BLUE_LIGHT, width=0.5, hoverinfo="skip")
    fig.add_scatter(x=stats["median"], y=stats[label], mode="markers",
                    marker=dict(color=BLUE, size=11, line=dict(color=CARD, width=2)),
                    customdata=stats[["p25", "p75", "n", "ci"]],
                    hovertemplate="Typical salary <b>%{x:$,.0f}</b> %{customdata[3]}<br>"
                                  "Most pay between %{customdata[0]:$,.0f} and %{customdata[1]:$,.0f}<br>"
                                  "Based on %{customdata[2]:,} postings<extra></extra>")
    fig.update_layout(height=height, xaxis=dict(tickformat="$~s", title="Yearly salary (USD)"),
                      yaxis=dict(gridcolor="rgba(0,0,0,0)"))
    return fig


def stats_table(stats: pd.DataFrame, label: str, name: str) -> pd.DataFrame:
    out = stats[[label, "n", "p25", "median", "p75"]].sort_values("median", ascending=False)
    out.columns = [name, "Salaried postings", "25th pct", "Median", "75th pct"]
    for c in ["25th pct", "Median", "75th pct"]:
        out[c] = out[c].map(lambda v: f"${v:,.0f}")
    return out


def finding(col, question: str, answer: str, detail: str, where: str):
    """One 'question → answer' card in the Key findings section."""
    col.markdown(f'<div class="finding"><div class="q">{question}</div><div class="a">{answer}</div>'
                 f'<div class="d">{detail}</div><div class="go">→ {where}</div></div>', unsafe_allow_html=True)


def key_findings():
    """The project's story: six research questions, each answered from the full dataset.

    Every number is computed here from the data and model results (nothing is
    typed in by hand), so the answers stay correct if the data is rebuilt.
    """
    sal = all_salaried["salary_mid"]
    fam = salary_stats(all_salaried[all_salaried["role_family"] != OTHER_FAMILY], "role_family")
    top3 = fam.nlargest(3, "median")
    sen = salary_stats(all_salaried, "seniority").set_index("seniority")["median"]
    entry, principal = sen["Entry / Junior"], sen["Principal / Staff"]
    effects = pd.DataFrame(meta["skill_effects"]).query("postings >= 300")["effect"] if meta else pd.Series([0.0])
    # Best-paid skill by raw median (skills with at least 100 salaried postings).
    skill_pay = (skills_long.dropna(subset=["salary_mid"]).groupby("skill")["salary_mid"]
                 .agg(["median", "size"]).query("size >= 100"))
    best_skill, best_pay = skill_pay["median"].idxmax(), skill_pay["median"].max()

    # Uncertainty from analysis/salary_statistics.py (company cluster bootstrap + regressions).
    med = stats_results.get("medians", {})
    overall_ci = f" (95% CI {usd(med['overall']['ci_low'])}–{usd(med['overall']['ci_high'])})" if med else ""
    fam_ci = med.get("role_family", {})
    # Do the top three's intervals overlap? If so, their order is not statistically certain.
    tops = [fam_ci.get(f) for f in top3["role_family"]]
    tied = all(tops) and max(t["ci_low"] for t in tops) <= min(t["ci_high"] for t in tops)
    ratio = med.get("ratios", {}).get("principal_vs_entry")
    regs = {r["model"]: r for r in stats_results.get("remote_regressions", [])}

    row1 = st.columns(3)
    finding(row1[0], "1 · How much do these jobs pay?", f"{usd(sal.median())} typical advertised salary",
            f"Half of advertised salaries fall between {usd(sal.quantile(.25))} and {usd(sal.quantile(.75))} "
            f"per year{overall_ci}, from {len(sal):,} postings with a yearly USD salary, mostly US jobs.",
            "Salaries tab")
    finding(row1[1], "2 · Which jobs pay the most?", ", ".join(top3["role_family"]),
            "Typical salaries of " + ", ".join(usd(v) for v in top3["median"]) +
            f", against {usd(sal.median())} across all jobs." +
            (" Their confidence intervals overlap, so the three are statistically tied." if tied else ""),
            "Salaries tab")
    finding(row1[2], "3 · How much does seniority matter?", f"{principal / entry:.1f}× from entry to staff level",
            f"Entry-level titles advertise {usd(entry)} typically; principal/staff titles {usd(principal)}" +
            (f" (95% CI for the ratio: {ratio['ci_low']:.1f}×–{ratio['ci_high']:.1f}×)." if ratio else "."),
            "Salaries tab")
    st.write("")
    row2 = st.columns(3)
    finding(row2[0], "4 · Do skills raise pay?", "Less than the job and employer do",
            f"Jobs asking for {best_skill} pay {usd(best_pay)} typically, but once title, company and city "
            f"are accounted for, a single skill shifts pay by at most {effects.abs().max():.0%}.", "Skills tab")
    if regs:
        raw, fe = regs["Raw difference"], regs["+ same company (fixed effects)"]
        finding(row2[1], "5 · Do remote jobs pay more?", f"Yes, about {fe['effect']:.0%} for comparable jobs",
                f"Remote postings advertise {raw['effect']:+.0%} more overall, but part of that is job mix. "
                f"Comparing similar jobs at the same company, the gap is {fe['effect']:+.1%} "
                f"(95% CI {fe['ci_low']:+.1%} to {fe['ci_high']:+.1%}).", "Salaries tab")
    if meta:
        finding(row2[2], "6 · Can machine learning predict a salary?",
                f"Yes, within {meta['test']['median_ape']:.0%} for a typical job",
                f"On {meta['n_test']:,} jobs it never saw, the model explains {meta['test']['r2']:.0%} of the "
                f"differences in pay (R² {meta['test']['r2']:.2f}). Job title and company matter most.",
                "Salary predictor tab")


TECH_FAMILIES = ["Software Engineering", "Data & AI", "Product & Design", "Security"]


def data_scope():
    """'What's in this data' box: which jobs are included and where the salary numbers come from.

    Answers the first question a reader has ("is this only tech jobs?") before any chart.
    All numbers are computed from the data.
    """
    known = jobs[jobs["role_family"] != OTHER_FAMILY]
    tech_share = known["role_family"].isin(TECH_FAMILIES).mean()
    biggest = [f for f in known["role_family"].value_counts().index if f not in TECH_FAMILIES][:5]
    sal_known = all_salaried[all_salaried["role_family"] != OTHER_FAMILY]
    us_share = (all_salaried["country"] == "United States").mean()
    senior_share = all_salaried["seniority"].isin(["Senior / Lead", "Principal / Staff",
                                                   "Director / Executive"]).mean()
    sal_tech = sal_known["role_family"].isin(TECH_FAMILIES).mean()
    listed = (jobs.assign(listed=jobs["salary_status"].ne("no_salary"))
              .groupby("country", observed=True)["listed"].mean())

    cols = st.columns(3)
    boxes = [
        ("Which jobs?", "Every type of role, not only tech",
         f"Tech jobs (software, data, product, security) are {tech_share:.0%} of postings. The rest include "
         f"{', '.join(b.lower() for b in biggest[:-1])} and {biggest[-1].lower()}, plus healthcare, education and more."),
        ("Which salaries?", f"{len(all_salaried):,} postings with a yearly USD salary",
         f"{us_share:.0%} of them are US jobs: pay-transparency laws mean {listed['United States']:.0%} of US "
         f"postings show a salary, versus {listed['Germany']:.0%} in Germany and {listed['France']:.0%} in France. "
         "Hourly or non-USD pay counts as a posting but not in salary charts."),
        ("Why do salaries look high?", "They describe these postings, not all workers",
         "Most are US jobs, where pay is higher than in most countries. Also, "
         f"{sal_tech:.0%} of salaried postings are tech jobs and {senior_share:.0%} are senior level or above. "
         "Use the figures to compare jobs with each other, not with national averages."),
    ]
    for col, (q, a, d) in zip(cols, boxes):
        col.markdown(f'<div class="scope"><div class="q">{q}</div><div class="a">{a}</div>'
                     f'<div class="d">{d}</div></div>', unsafe_allow_html=True)


def top_skill_by(df_rows: pd.Index, by: str) -> pd.Series:
    """Most frequently requested skill within each group (e.g. each role family)."""
    sk = skills_long[skills_long["row"].isin(df_rows)]
    counts = sk.groupby([by, "skill"], observed=True).size().reset_index(name="n")
    return counts.sort_values("n").groupby(by, observed=True).tail(1).set_index(by)["skill"]


# ══ Load everything ═══════════════════════════════════════════════════════════
jobs = load_jobs()
skills_long = load_skills_long()
model, meta = load_model()
summary = load_summary()
stats_results = load_json(STATS)       # confidence intervals + remote-pay regressions
label_check = load_json(LABEL_CHECK)   # job-type rules checked against employers' departments
all_salaried = jobs.dropna(subset=["salary_mid"])

# ══ Sidebar filters (apply to every tab except the ML predictor) ══════════════
with st.sidebar:
    st.markdown("### Filters")
    st.caption("Narrow every chart to the postings you care about.")
    families = sorted(jobs["role_family"].cat.categories)
    sel_family = st.multiselect("Role family", families, placeholder="All role families")
    sel_seniority = st.multiselect("Seniority (from title)", SENIORITY_ORDER, placeholder="All levels")
    top_countries = jobs["country"].value_counts().index[:40].tolist()
    sel_country = st.selectbox("Country", ["All countries"] + top_countries)
    sel_work = st.radio("Work arrangement", ["All", "Remote", "On-site"], horizontal=True)
    sel_type = st.multiselect("Employment type", sorted(jobs["employment_type"].cat.categories),
                              placeholder="All types")

# Combine the filters into one boolean mask over all postings.
mask = pd.Series(True, index=jobs.index)
if sel_family:
    mask &= jobs["role_family"].isin(sel_family)
if sel_seniority:
    mask &= jobs["seniority"].isin(sel_seniority)
if sel_country != "All countries":
    mask &= jobs["country"].eq(sel_country)
if sel_work != "All":
    mask &= jobs["remote"].eq(sel_work == "Remote")
if sel_type:
    mask &= jobs["employment_type"].isin(sel_type)
df = jobs[mask]
salaried = df.dropna(subset=["salary_mid"])

# ══ Header ════════════════════════════════════════════════════════════════════
links_html = "".join(f'<a href="{url}" target="_blank">{name}</a>' for name, url in LINKS.items())
# Key stats under the subtitle (all numbers come from the data files).
stats = [(f"{summary['postings']:,}", "Job postings"), (f"{summary['companies']:,}", "Companies"),
         (str(summary["countries"]), "Countries"), (f"{summary['usable_salaries']:,}", "Postings with a salary")]
if meta:
    stats.append((f"{meta['test']['r2']:.2f}", "Model R²"))
stats_html = "".join(f'<div class="stat"><div class="v">{v}</div><div class="l">{label}</div></div>'
                     for v, label in stats)
st.markdown(f"""
<div class="masthead">
  <div class="eyebrow">Job Market Analytics</div>
  <h1>What <span class="num">{summary['postings']:,}</span> job postings, across every type of role,
      reveal about pay, skills and hiring</h1>
  <div class="sub">An analysis of open roles at {summary['companies']:,} companies in {summary['countries']} countries
       ({pd.Timestamp(summary['posted_max']):%B %Y}), with a machine-learning model that estimates salaries.</div>
  <div class="stats">{stats_html}</div>
  <div class="author">{AUTHOR}</div>
  <div class="links">{links_html}</div>
</div>""", unsafe_allow_html=True)

tab_overview, tab_salary, tab_skills, tab_explore, tab_model, tab_about = st.tabs(
    ["Overview", "Salaries", "Skills", "Explore a role", "Salary predictor (ML)", "About the data"])

# ══ Overview ══════════════════════════════════════════════════════════════════
with tab_overview:
    st.markdown("### What's in this data")
    data_scope()
    st.write("")
    st.markdown("### Key findings")
    st.caption("Six questions this project set out to answer, using all postings (the sidebar filters "
               "don't change these). The tabs explore each one in detail.")
    key_findings()
    st.write("")
    st.markdown("### The market at a glance")
    if df.empty:
        st.warning("No postings match these filters.")
    else:
        if mask.all():
            st.caption("Showing all postings. Use the sidebar filters to narrow every chart below.")
        else:
            st.caption(f"Showing {len(df):,} postings that match your sidebar filters.")
        c1, c2, c3, c4, c5 = st.columns(5)
        kpi(c1, f"{len(df):,}", "Job postings")
        kpi(c2, f"{df['company'].nunique():,}", "Hiring companies")
        kpi(c3, f"{df['salary_status'].ne('no_salary').mean():.0%}", "Postings that show a salary")
        kpi(c4, f"${salaried['salary_mid'].median():,.0f}" if len(salaried) else "—",
            "Typical (median) salary", f"from {len(salaried):,} postings with a yearly USD salary")
        kpi(c5, f"{df['remote'].mean():.0%}", "Remote postings")
        st.write("")

        # World map: where postings are, and how often they state pay.
        geo = (df.assign(listed=df["salary_status"].ne("no_salary"))
               .groupby("country", observed=True)
               .agg(postings=("title", "size"), listed=("listed", "mean"), remote=("remote", "mean"))
               .reset_index())
        geo = geo[geo["postings"] > 0]
        geo["iso3"] = geo["country"].map(iso3_lookup())
        # Median advertised salary per country, with the number of postings it is based on,
        # so the tooltip says exactly what the figure is (and isn't).
        country_pay = salary_stats(salaried, "country").set_index("country")
        geo["median"] = geo["country"].map(country_pay["median"])
        geo["median_txt"] = [
            f"${m:,.0f} (median of {int(n):,} postings with a yearly USD salary)" if pd.notna(m)
            else "too few postings with a yearly USD salary"
            for m, n in zip(geo["median"], geo["country"].map(country_pay["n"]))]
        with card("Where are the jobs?",
                  "Darker countries have more job postings. Hover over a country to see its number of postings, "
                  "how many show a salary, how many are remote, and the typical advertised salary. Salaries "
                  "describe the jobs in this dataset (mostly tech and professional roles), not all workers "
                  "in a country."):
            ticks = [10, 100, 1_000, 10_000, 100_000]
            fig = go.Figure(go.Choropleth(
                locations=geo["iso3"], locationmode="ISO-3", z=np.log10(geo["postings"]),
                text=geo["country"],
                colorscale=[[0, "#dbe8fa"], [0.5, "#5598e7"], [1, NAVY]], marker_line_color="white",
                marker_line_width=0.5,
                colorbar=dict(title=dict(text="Postings", font=dict(size=12, color=INK_2)), thickness=10,
                              len=0.6, tickvals=np.log10(ticks), ticktext=[f"{t:,}" for t in ticks],
                              tickfont=dict(color=MUTED, size=11), outlinewidth=0),
                customdata=geo[["postings", "listed", "remote", "median_txt"]],
                hovertemplate="<b>%{text}</b><br>%{customdata[0]:,} postings<br>"
                              "%{customdata[1]:.0%} show a salary · %{customdata[2]:.0%} remote<br>"
                              "Typical advertised salary: %{customdata[3]}<extra></extra>"))
            fig.update_geos(showframe=False, showcoastlines=False, projection_type="natural earth",
                            landcolor="#f1f2f4", showland=True, showcountries=True, countrycolor="white",
                            bgcolor=CARD, lataxis_range=[-58, 85])
            fig.update_layout(height=430, margin=dict(l=0, r=0, t=0, b=0))
            show(fig, geo[["country", "postings", "listed", "remote", "median"]]
                 .sort_values("postings", ascending=False)
                 .rename(columns={"country": "Country", "postings": "Postings", "listed": "Share listing pay",
                                  "remote": "Share remote", "median": "Median USD salary"}))

        left, right = st.columns(2)
        with left:
            # Role mix, with pay / remote share / top skill in the tooltip.
            prof = (df.groupby("role_family", observed=True)
                    .agg(postings=("title", "size"), remote=("remote", "mean")).reset_index())
            prof = prof[prof["postings"] > 0]
            prof["share"] = prof["postings"] / prof["postings"].sum()
            # Titles that match no job type are left out of the chart (their share is stated instead).
            other_share = prof.loc[prof["role_family"] == OTHER_FAMILY, "share"].sum()
            prof = prof[prof["role_family"] != OTHER_FAMILY]
            prof["median"] = prof["role_family"].map(salaried.groupby("role_family", observed=True)["salary_mid"]
                                                     .median()).map(lambda v: usd(v) if pd.notna(v) else "n/a")
            prof["skill"] = prof["role_family"].map(top_skill_by(df.index, "role_family")).fillna("n/a")
            prof["_text"] = prof["share"].map(lambda v: f"{v:.0%}")
            prof = prof[["role_family", "postings", "share", "median", "remote", "skill", "_text"]]
            with card("Which types of jobs are in demand?",
                      "Each bar is a type of job; the % is its share of all postings. Hover to see typical pay, "
                      f"% remote, and the skill employers ask for most. The {other_share:.0%} of postings whose "
                      "title fits no job type (mostly non-English or very generic titles) are not shown."):
                fig = ranked_bar(prof, "postings", "role_family",
                                 "%{x:,} postings<br>Typical salary: %{customdata[2]}<br>"
                                 "Remote: %{customdata[3]:.0%}<br>Most-asked skill: %{customdata[4]}", 520)
                fig.update_layout(xaxis=dict(title="Number of job postings"))
                show(fig, prof.drop(columns="_text").rename(columns={
                    "role_family": "Role family", "postings": "Postings", "share": "Share",
                    "median": "Median salary", "remote": "Share remote", "skill": "Top skill"}))
        with right:
            # Companies: hiring volume vs. pay. Name is only shown on hover (too many to label).
            comp = salary_stats(salaried, "company")
            comp = comp.merge(df.groupby("company").size().rename("postings"), left_on="company",
                              right_index=True)
            comp = comp.nlargest(60, "n")
            comp["family"] = comp["company"].map(salaried.groupby("company")["role_family"]
                                                 .agg(lambda s: s.mode().iat[0]))
            label_top = comp.nlargest(4, "median")["company"]  # label only the four best payers
            with card("Which companies hire the most, and which pay the most?",
                      "Each bubble is a company. Further right = more job postings; higher up = better typical pay; "
                      "bigger bubble = more postings that show a salary. Hover for the company name."):
                fig = go.Figure(go.Scatter(
                    x=comp["postings"], y=comp["median"], mode="markers+text",
                    text=np.where(comp["company"].isin(label_top), comp["company"], ""),
                    textposition="top center", textfont=dict(size=11, color=INK_2),
                    marker=dict(size=np.sqrt(comp["n"]) * 1.4 + 6, color=BLUE, opacity=0.55,
                                line=dict(color=CARD, width=1.5)),
                    customdata=comp[["company", "n", "p25", "p75", "family"]],
                    hovertemplate="<b>%{customdata[0]}</b><br>%{x:,} postings · typical pay %{y:$,.0f}<br>"
                                  "Most pay between %{customdata[2]:$,.0f} and %{customdata[3]:$,.0f}<br>"
                                  "Mostly hiring: %{customdata[4]}<extra></extra>"))
                # Log scale spreads out companies with few vs. many postings; ticks are spelled out.
                ticks = [50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000]
                fig.update_layout(height=520, xaxis=dict(type="log", title="Number of job postings (log scale)",
                                                         tickvals=ticks, ticktext=[f"{t:,}" for t in ticks]),
                                  yaxis=dict(tickformat="$~s", title="Typical (median) salary"))
                show(fig, stats_table(comp, "company", "Company"))

# ══ Salaries ══════════════════════════════════════════════════════════════════
with tab_salary:
    if len(salaried) < MIN_GROUP:
        st.warning(f"Fewer than {MIN_GROUP} postings with an annual USD salary match these filters.")
    else:
        st.caption("How to read these charts: a posting's salary is the middle of its advertised yearly range "
                   "(e.g. \\$100k–\\$140k → \\$120k). “Typical” means the **median**: half of jobs pay more, half "
                   f"pay less. Groups with fewer than {MIN_GROUP} postings are left out because they are too small.")

        # Distribution with an interactive "where do I fall?" marker.
        median = salaried["salary_mid"].median()
        with card("How much do jobs pay?",
                  f"Each bar counts job postings whose salary falls in a $10k band ({len(salaried):,} postings). "
                  "The gray line marks the typical salary. Type a salary below to see how it ranks."):
            mine = st.number_input("Compare a salary (USD per year)", min_value=0, value=120_000, step=5_000)
            pct = (salaried["salary_mid"] < mine).mean()
            fig = go.Figure(go.Histogram(
                x=salaried["salary_mid"].clip(upper=500_000), xbins=dict(start=20_000, end=510_000, size=10_000),
                marker=dict(color=BLUE, line=dict(color=CARD, width=1)),
                hovertemplate="%{y:,} postings<extra></extra>"))
            fig.add_vline(x=median, line_color=INK_2, line_width=1.5)
            fig.add_annotation(x=median, y=1.0, yref="paper", yanchor="bottom", showarrow=False,
                               text=f"Typical salary {usd(median)}", font=dict(color=INK, size=12))
            if mine:
                fig.add_vline(x=min(mine, 500_000), line_color=ORANGE, line_width=2)
                fig.add_annotation(x=min(mine, 500_000), y=0.85, yref="paper", showarrow=False, xanchor="left",
                                   xshift=6, text=f"{usd(mine)} pays more than {pct:.0%} of jobs",
                                   font=dict(color=INK, size=12), bgcolor="rgba(255,255,255,0.85)")
            fig.update_layout(height=340, bargap=0.04, margin=dict(t=28),
                              xaxis=dict(tickformat="$~s", title="Yearly salary (USD)"),
                              yaxis=dict(title="Number of job postings"))
            quantiles = salaried["salary_mid"].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
            show(fig, pd.DataFrame({"Percentile": ["10th", "25th", "50th (median)", "75th", "90th"],
                                    "Annual salary": [f"${v:,.0f}" for v in quantiles]}))

        left, right = st.columns(2)
        with left:
            stats = salary_stats(salaried[salaried["role_family"] != OTHER_FAMILY], "role_family")
            # Confidence intervals were computed on all postings, so they are shown only when unfiltered.
            fam_ci = stats_results.get("medians", {}).get("role_family") if mask.all() else None
            with card("Typical pay by type of job",
                      "Dot = typical (median) salary. Light bar = the range where the middle half of salaries "
                      "fall, so a longer bar means pay varies more." +
                      (" Hover for the 95% confidence interval of each median." if fam_ci else "")):
                show(median_iqr_chart(stats, "role_family", 520, fam_ci),
                     stats_table(stats, "role_family", "Role family"))
        with right:
            stats = salary_stats(salaried, "seniority")
            stats["seniority"] = stats["seniority"].astype(str)
            stats["order"] = stats["seniority"].map(SENIORITY_ORDER.index)
            stats = stats.sort_values("order")
            stats["label"] = stats["seniority"].str.replace(" / ", "<br>")
            # % change versus the previous level, shown in the tooltip.
            stats["step"] = stats["median"].pct_change().map(lambda v: f"{v:+.0%} vs. previous level"
                                                             if pd.notna(v) else "starting level")
            with card("How pay grows with seniority",
                      "Typical salary at each career level. The level comes from words in the job title "
                      "(e.g. 'Junior', 'Senior', 'Staff', 'Director')."):
                fig = go.Figure(go.Bar(
                    x=stats["label"], y=stats["median"], marker_color=[BLUE_RAMP[i] for i in stats["order"]],
                    text=stats["median"].map(usd), textposition="outside", cliponaxis=False,
                    textfont=dict(color=INK_2), customdata=stats[["p25", "p75", "n", "step"]],
                    hovertemplate="%{customdata[3]}<br>Most pay between %{customdata[0]:$,.0f} and "
                                  "%{customdata[1]:$,.0f}<br>%{customdata[2]:,} postings<extra></extra>"))
                # 95% confidence-interval whiskers (bootstrap, all postings): only when unfiltered.
                sen_ci = stats_results.get("medians", {}).get("seniority") if mask.all() else None
                if sen_ci:
                    lo = stats["seniority"].map(lambda k: sen_ci[k]["ci_low"])
                    hi = stats["seniority"].map(lambda k: sen_ci[k]["ci_high"])
                    fig.update_traces(error_y=dict(type="data", symmetric=False, array=hi - stats["median"],
                                                   arrayminus=stats["median"] - lo, color=INK_2,
                                                   thickness=1.5, width=6),
                                      # value labels move inside the bars so they don't overlap the whiskers
                                      textposition="inside", insidetextanchor="start",
                                      # dark text on the light bars, white text on the dark bars
                                      textfont=dict(color=[INK if i < 2 else "#ffffff" for i in stats["order"]],
                                                    size=12))
                fig.update_layout(height=520, margin=dict(t=24),
                                  yaxis=dict(tickformat="$~s", title="Typical (median) salary"),
                                  xaxis=dict(gridcolor="rgba(0,0,0,0)", categoryorder="array",
                                             categoryarray=stats["label"].tolist(), tickangle=0))
                if sen_ci:
                    st.caption("Black whiskers = 95% confidence interval for each median.")
                show(fig, stats_table(stats, "seniority", "Seniority"))

        # Remote vs. on-site pay, per role family (dumbbell chart).
        g = salaried.groupby(["role_family", "remote"], observed=True)["salary_mid"].agg(["median", "size"])
        gap = g["median"].unstack().rename(columns={True: "remote", False: "onsite"})
        sizes = g["size"].unstack()
        gap = gap[(sizes >= MIN_GROUP).all(axis=1)].dropna().reset_index()
        gap = gap[gap["role_family"] != OTHER_FAMILY]
        if len(gap):
            gap["diff"] = gap["remote"] / gap["onsite"] - 1
            gap = gap.sort_values("diff")
            with card("Remote vs. on-site pay, by type of job",
                      "For each type of job: blue dot = typical remote salary, orange dot = typical on-site salary. "
                      "The gray line between them is the pay gap. A gap can also reflect a different mix of jobs "
                      "(e.g. remote healthcare postings are mostly senior clinicians), not only the work arrangement."):
                fig = go.Figure()
                for _, r in gap.iterrows():  # grey connector per role family
                    fig.add_scatter(x=[r["onsite"], r["remote"]], y=[r["role_family"]] * 2, mode="lines",
                                    line=dict(color=AXIS, width=3), hoverinfo="skip", showlegend=False)
                fig.add_scatter(x=gap["onsite"], y=gap["role_family"], mode="markers", name="On-site",
                                marker=dict(color=ORANGE, size=12, line=dict(color=CARD, width=2)),
                                customdata=gap[["diff"]],
                                hovertemplate="On-site: %{x:$,.0f}<extra></extra>")
                fig.add_scatter(x=gap["remote"], y=gap["role_family"], mode="markers", name="Remote",
                                marker=dict(color=BLUE, size=12, line=dict(color=CARD, width=2)),
                                customdata=gap[["diff"]],
                                hovertemplate="Remote: %{x:$,.0f}<br>%{customdata[0]:+.0%} compared with on-site"
                                              "<extra></extra>")
                fig.update_layout(height=max(320, 34 * len(gap)), showlegend=True,
                                  xaxis=dict(tickformat="$~s", title="Typical (median) salary"),
                                  yaxis=dict(gridcolor="rgba(0,0,0,0)"), margin=dict(t=36))
                show(fig, gap.assign(diff=gap["diff"].map(lambda v: f"{v:+.1%}"))
                     .rename(columns={"role_family": "Role family", "remote": "Remote median",
                                      "onsite": "On-site median", "diff": "Remote vs. on-site"}))

        # Fair comparison: how the remote gap changes as similar jobs are compared (regressions).
        regs = pd.DataFrame(stats_results.get("remote_regressions", []))
        if len(regs):
            with card("A fair comparison: how much of the remote gap is real?",
                      "Each row compares remote and on-site postings more strictly than the one above it. "
                      "Dot = estimated remote pay difference; line = 95% confidence interval (if it doesn't cross "
                      "0%, the difference is statistically significant). Uses all postings, not the filters."):
                regs = regs.iloc[::-1]  # first model at the top
                fig = go.Figure(go.Scatter(
                    x=regs["effect"], y=regs["model"], mode="markers+text",
                    marker=dict(color=BLUE, size=12, line=dict(color=CARD, width=2)),
                    error_x=dict(type="data", symmetric=False, array=regs["ci_high"] - regs["effect"],
                                 arrayminus=regs["effect"] - regs["ci_low"], color=INK_2, thickness=1.5, width=6),
                    text=regs["effect"].map(lambda v: f"{v:+.1%}"), textposition="top center",
                    textfont=dict(color=INK_2, size=12), customdata=regs[["ci_low", "ci_high", "p_value", "n"]],
                    hovertemplate="Remote pay difference %{x:+.1%}<br>95% CI %{customdata[0]:+.1%} to "
                                  "%{customdata[1]:+.1%}<br>p-value %{customdata[2]:.2g} · "
                                  "%{customdata[3]:,} postings<extra></extra>"))
                fig.add_vline(x=0, line_color=AXIS, line_width=1.5)
                fig.update_layout(height=320, margin=dict(t=24),
                                  xaxis=dict(tickformat="+.0%", title="Remote pay vs. on-site (comparable jobs)",
                                             range=[min(0, regs["ci_low"].min()) - 0.02, regs["ci_high"].max() + 0.03]),
                                  yaxis=dict(gridcolor="rgba(0,0,0,0)"))
                st.caption("Rows: (1) all postings · (2) same job type and seniority · (3) also same country and "
                           "employment type · (4) also the same company. Regression of log salary with "
                           "company-clustered standard errors.")
                show(fig, regs.iloc[::-1].assign(
                    effect=lambda d: d["effect"].map(lambda v: f"{v:+.1%}"),
                    ci=lambda d: [f"{a:+.1%} to {b:+.1%}" for a, b in zip(d["ci_low"], d["ci_high"])],
                    p_value=lambda d: d["p_value"].map(lambda v: f"{v:.2g}"))
                    [["model", "effect", "ci", "p_value", "n"]]
                    .rename(columns={"model": "Comparison", "effect": "Remote difference", "ci": "95% CI",
                                     "p_value": "p-value", "n": "Postings"}))

        left, right = st.columns(2)
        with left:
            stats = salary_stats(salaried.dropna(subset=["city"]), "city").nlargest(15, "n")
            with card("Typical pay in the 15 cities with the most jobs",
                      "Dot = typical salary · light bar = range of the middle half of salaries"):
                show(median_iqr_chart(stats, "city", 480), stats_table(stats, "city", "City"))
        with right:
            stats = salary_stats(salaried.dropna(subset=["industry"]), "industry").nlargest(15, "n")
            with card("Typical pay in the 15 industries with the most jobs",
                      "Dot = typical salary · light bar = range of the middle half of salaries"):
                show(median_iqr_chart(stats, "industry", 480), stats_table(stats, "industry", "Industry"))

# ══ Skills ════════════════════════════════════════════════════════════════════
with tab_skills:
    sk = skills_long[skills_long["row"].isin(df.index)]
    if sk.empty:
        st.warning("No skills found for these filters.")
    else:
        overall_median = salaried["salary_mid"].median()

        # Interactive skill explorer.
        with card("Skill explorer", "Choose a skill to see how often employers ask for it, what those jobs pay, "
                                    "and which other skills usually come with it."):
            counts = sk["skill"].value_counts()
            choices = counts.index[:120].tolist()
            skill = st.selectbox("Skill", choices, index=choices.index("python") if "python" in choices else 0)
            rows = sk.loc[sk["skill"] == skill, "row"]
            with_skill = jobs.loc[rows]
            pay = with_skill["salary_mid"].median()
            k1, k2, k3, k4 = st.columns(4)
            kpi(k1, f"{len(rows):,}", f"Job postings asking for {skill}", f"{len(rows) / len(df):.1%} of all postings")
            kpi(k2, f"${pay:,.0f}" if pd.notna(pay) else "—", "Typical salary of these jobs",
                f"from {with_skill['salary_mid'].notna().sum():,} postings with a salary")
            kpi(k3, f"{pay / overall_median - 1:+.0%}" if pd.notna(pay) else "—", "Compared with all jobs",
                f"all jobs: {usd(overall_median)}")
            kpi(k4, f"{with_skill['remote'].mean():.0%}", "Of these jobs are remote")
            st.write("")

            a, b = st.columns(2)
            with a:
                # Co-occurrence: of the postings that list the skill, what share also list each other skill?
                pairs = sk[sk["row"].isin(rows) & sk["skill"].ne(skill)]
                paired = (pairs["skill"].value_counts().head(12) / len(rows)).rename_axis("skill") \
                    .reset_index(name="share")
                paired["_text"] = paired["share"].map(lambda v: f"{v:.0%}")
                st.markdown(f"**Skills most often asked for together with {skill}**")
                fig = ranked_bar(paired, "share", "skill",
                                 f"%{{x:.0%}} of jobs asking for {skill} also ask for this", 400)
                fig.update_layout(xaxis=dict(tickformat=".0%", title=f"% of {skill} jobs"))
                show(fig)
            with b:
                roles = with_skill["role_family"].value_counts().drop(OTHER_FAMILY, errors="ignore").head(8)
                roles = (roles / len(rows)).rename_axis("family").reset_index(name="share")
                roles["_text"] = roles["share"].map(lambda v: f"{v:.0%}")
                st.markdown(f"**Types of jobs that ask for {skill}**")
                fig = ranked_bar(roles, "share", "family", f"%{{x:.0%}} of jobs asking for {skill}", 400)
                fig.update_layout(xaxis=dict(tickformat=".0%", title=f"% of {skill} jobs"))
                show(fig)

        left, right = st.columns(2)
        with left:
            demand = counts.head(20).rename_axis("skill").reset_index(name="postings")
            demand["share"] = demand["postings"] / len(df)
            skill_pay = sk.groupby("skill")["salary_mid"].median()
            demand["median"] = demand["skill"].map(skill_pay).map(lambda v: usd(v) if pd.notna(v) else "n/a")
            demand["_text"] = demand["share"].map(lambda v: f"{v:.1%}")
            with card("Most requested skills", "The % is the share of all job postings that mention the skill."):
                fig = ranked_bar(demand, "postings", "skill",
                                 "%{x:,} postings<br>Typical salary of these jobs: %{customdata[2]}", 560)
                fig.update_layout(xaxis=dict(title="Number of job postings"))
                show(fig, demand.drop(columns="_text").rename(columns={
                    "skill": "Skill", "postings": "Postings", "share": "Share", "median": "Median salary"}))
        with right:
            pay = (sk.dropna(subset=["salary_mid"]).groupby("skill")["salary_mid"]
                   .agg(median="median", n="size").reset_index())
            pay = pay[pay["n"] >= 100].nlargest(20, "median")
            pay["premium"] = pay["median"] / overall_median - 1
            pay["_text"] = pay["median"].map(usd)
            with card("Skills found in the best-paying jobs",
                      "Typical salary of job postings that list each skill (skills with at least 100 postings)."):
                fig = ranked_bar(pay, "median", "skill", "%{customdata[2]:+.0%} compared with all jobs<br>"
                                 "Based on %{customdata[1]:,} postings", 560)
                fig.update_layout(xaxis=dict(tickformat="$~s", title="Typical (median) salary"))
                show(fig, pay.drop(columns="_text").rename(columns={
                    "skill": "Skill", "median": "Median salary", "n": "Salaried postings", "premium": "Premium"}))

        if meta and meta.get("skill_effects"):
            effects = pd.DataFrame(meta["skill_effects"])
            effects = effects[effects["postings"] >= 300]
            effects = pd.concat([effects.nlargest(10, "effect"), effects.nsmallest(10, "effect")])
            effects = effects.drop_duplicates("skill").sort_values("effect")
            with card("How much does a single skill change pay? (machine-learning estimate)",
                      "Compares otherwise identical jobs (same title, company and city) with and without the skill. "
                      "Blue = skill comes with higher pay, red = lower. The effects are small: the job and the "
                      "employer matter far more than any one skill."):
                fig = go.Figure(go.Bar(
                    x=effects["effect"], y=effects["skill"], orientation="h",
                    marker_color=np.where(effects["effect"] >= 0, BLUE, RED),
                    text=effects["effect"].map(lambda v: f"{v:+.1%}"), textposition="outside",
                    textfont=dict(color=INK_2, size=12), cliponaxis=False, customdata=effects[["postings"]],
                    hovertemplate="Estimated from %{customdata[0]:,} postings<extra></extra>"))
                fig.add_vline(x=0, line_color=AXIS, line_width=1)
                reach = effects["effect"].abs().max() * 1.35
                fig.update_layout(height=560, yaxis=dict(gridcolor="rgba(0,0,0,0)"),
                                  xaxis=dict(tickformat="+.0%", title="Change in salary when the skill is listed",
                                             range=[-reach, reach]))
                show(fig, effects.sort_values("effect", ascending=False)
                     .assign(effect=lambda d: d["effect"].map(lambda v: f"{v:+.2%}"))
                     .rename(columns={"skill": "Skill", "effect": "Effect", "postings": "Postings"}))

# ══ Explore a role ════════════════════════════════════════════════════════════
with tab_explore:
    st.markdown("Type a job title to see its pay, skills and top-paying companies. Add a second title to "
                "compare them side by side (**blue** = first title, **orange** = second).")
    q1, q2 = st.columns(2)
    query_a = q1.text_input("Job title", "data engineer").strip().lower()
    query_b = q2.text_input("Compare with (optional)", "software engineer").strip().lower()
    queries = [(query_a, BLUE)] + ([(query_b, ORANGE)] if query_b else [])
    matches = {q: df[df["title_lc"].str.contains(q, regex=False)] for q, _ in queries if q}

    if not query_a or matches[query_a].empty:
        st.warning("No postings match that title. Try a broader phrase, e.g. 'analyst'.")
    else:
        # One KPI row per searched title.
        for q, color in queries:
            m = matches.get(q)
            if m is None or m.empty:
                continue
            sal = m["salary_mid"].dropna()
            st.markdown(f'<span style="color:{color};font-weight:700">●</span> **“{q}”**', unsafe_allow_html=True)
            k1, k2, k3, k4, k5 = st.columns(5)
            kpi(k1, f"{len(m):,}", "Job postings")
            kpi(k2, f"${sal.median():,.0f}" if len(sal) else "—", "Typical salary", f"from {len(sal):,} postings")
            kpi(k3, f"{usd(sal.quantile(.25))} – {usd(sal.quantile(.75))}" if len(sal) >= 4 else "—",
                "Most jobs pay between", "middle half of salaries")
            kpi(k4, f"{m['remote'].mean():.0%}", "Remote jobs")
            kpi(k5, f"{m['company'].nunique():,}", "Companies hiring")
            st.write("")

        left, right = st.columns(2)
        with left:
            with card("How the salaries compare",
                      "Each bar shows the % of that title's job postings paying within a $20k band. "
                      "Where the colors overlap, both titles pay similarly."):
                fig = go.Figure()
                for q, color in queries:
                    sal = matches.get(q, pd.DataFrame(columns=["salary_mid"]))["salary_mid"].dropna()
                    if len(sal):
                        fig.add_histogram(x=sal.clip(upper=500_000), name=f"“{q}”", histnorm="percent",
                                          xbins=dict(start=20_000, end=520_000, size=20_000),
                                          marker=dict(color=color, line=dict(color=CARD, width=1)), opacity=0.7,
                                          hovertemplate="%{y:.1f}% of postings<extra>%{fullData.name}</extra>")
                fig.update_layout(barmode="overlay", height=400, showlegend=True, margin=dict(t=36),
                                  xaxis=dict(tickformat="$~s", title="Yearly salary (USD)"),
                                  yaxis=dict(title="% of job postings", ticksuffix="%"))
                show(fig)
        with right:
            with card("Skills each title asks for",
                      "% of each title's job postings that mention the skill"):
                frames = []
                for q, _ in queries:
                    m = matches.get(q)
                    if m is not None and len(m):
                        s = skills_long[skills_long["row"].isin(m.index)]["skill"].value_counts() / len(m)
                        frames.append(s.rename(q))
                both = pd.concat(frames, axis=1).fillna(0)
                both = both.loc[both[query_a].nlargest(10).index].iloc[::-1]
                fig = go.Figure()
                for q, color in queries[::-1]:  # reversed so the first title's bars sit on top
                    if q in both:
                        fig.add_bar(x=both[q], y=both.index, orientation="h", name=f"“{q}”", marker_color=color,
                                    hovertemplate="%{x:.0%} of postings<extra>%{fullData.name}</extra>")
                fig.update_layout(barmode="group", height=400, showlegend=True, bargap=0.25, bargroupgap=0.08,
                                  xaxis=dict(tickformat=".0%", title="% of job postings"),
                                  yaxis=dict(gridcolor="rgba(0,0,0,0)"), margin=dict(t=36),
                                  legend=dict(traceorder="reversed"))
                show(fig)

        m = matches[query_a]
        top_pay = (m.dropna(subset=["salary_mid"]).groupby("company")["salary_mid"]
                   .agg(median="median", n="size").reset_index().query("n >= 3").nlargest(12, "median"))
        left, right = st.columns(2)
        with left:
            with card(f"Best-paying companies for “{query_a}”",
                      "Typical salary at each company (companies with at least 3 postings that show a salary)"):
                if top_pay.empty:
                    st.info("Not enough salaried postings per company for this title.")
                else:
                    top_pay["_text"] = top_pay["median"].map(usd)
                    fig = ranked_bar(top_pay, "median", "company", "Based on %{customdata[1]:,} postings", 420)
                    fig.update_layout(xaxis=dict(tickformat="$~s", title="Typical (median) salary"))
                    show(fig)
        with right:
            with card(f"Most recent “{query_a}” postings", "Click “open” to view the original job posting"):
                st.dataframe(m.sort_values("posted", ascending=False)
                             [["title", "company", "location", "salary_mid", "posted", "url"]].head(200),
                             width="stretch", hide_index=True, height=420,
                             column_config={"title": st.column_config.TextColumn("Title", width="medium"),
                                            "company": "Company", "location": "Location",
                                            "salary_mid": st.column_config.NumberColumn("Salary (mid)", format="$%d"),
                                            "posted": st.column_config.DateColumn("Posted"),
                                            "url": st.column_config.LinkColumn("Posting", display_text="open")})

# ══ Salary predictor ══════════════════════════════════════════════════════════
with tab_model:
    if model is None:
        st.error("Model files are missing. Run `python models/train.py`.")
    else:
        test, unseen = meta["test"], meta["unseen_company_test"]
        st.markdown(
            f"Describe a job and a machine-learning model, trained on {meta['n_deduplicated']:,} job postings "
            "that show a salary, estimates what it pays. You also get a range: in testing, the real salary "
            "landed inside this range 8 times out of 10.")

        with st.form("estimator"):
            a, b, c = st.columns(3)
            title = a.text_input("Job title", "Senior Data Engineer")
            companies = sorted(all_salaried["company"].value_counts().loc[lambda s: s >= 3].index)
            company = a.selectbox("Company", ["Not listed"] + companies,
                                  help="Companies with at least 3 salaried postings in the data.")
            department = a.text_input("Department (optional)", "")

            city_counts = all_salaried["city"].value_counts()
            cities = sorted(city_counts[city_counts >= 20].index)
            city = b.selectbox("City", ["Not listed"] + cities,
                               index=(cities.index("New York") + 1) if "New York" in cities else 0)
            country = b.selectbox("Country (used when the city is not listed)",
                                  ["United States"] + [x for x in top_countries if x != "United States"])
            industries = sorted(jobs["industry"].dropna().unique())
            industry = b.selectbox("Industry", ["Use company's industry"] + industries)

            employment = c.selectbox("Employment type", ["Full Time", "Part Time", "Contract", "Internship",
                                                         "Temporary"])
            remote = c.toggle("Remote", value=False)
            all_skills = sorted(skills_long["skill"].unique())
            skills = c.multiselect("Skills", all_skills,
                                   default=[s for s in ["python", "sql", "aws"] if s in all_skills])
            submitted = st.form_submit_button("Estimate salary", type="primary")

        if submitted and title.strip():
            # Fill in country/industry from lookups learned at training time.
            if city != "Not listed":
                country = meta["city_country"].get(city, country)
            if industry == "Use company's industry":
                industry = meta["company_industry"].get(company)
            row = pd.DataFrame([{
                "title": title.strip(), "company": None if company == "Not listed" else company,
                "industry": industry, "department": department.strip() or None, "country": country,
                "city": None if city == "Not listed" else city, "employment_type": employment,
                "remote": remote, "skills": skills}])
            pred_log = float(model.predict(model_frame(row))[0])
            low = np.exp(pred_log + meta["interval"]["log_low"])
            high = np.exp(pred_log + meta["interval"]["log_high"])
            fam, sen = role_family(title), seniority(title)
            peers = all_salaried[(all_salaried["role_family"] == fam) & (all_salaried["seniority"] == sen)]

            r1, r2, r3 = st.columns(3)
            kpi(r1, f"${np.exp(pred_log):,.0f}", "Estimated salary per year")
            kpi(r2, f"${low:,.0f} – ${high:,.0f}", "Likely range",
                f"the real salary fell in this range {test['interval_coverage']:.0%} of the time")
            kpi(r3, f"${peers['salary_mid'].median():,.0f}" if len(peers) else "—",
                "Typical salary for similar jobs", f"{fam} · {sen} ({len(peers):,} postings)")
            st.write("")

            # Where the prediction sits among similar real postings.
            if len(peers) >= MIN_GROUP:
                with card("Your estimate vs. similar real jobs",
                          f"Light-blue bars: salaries of {len(peers):,} real {fam} · {sen} job postings. "
                          "Orange line: your estimate. Shaded orange: the likely range."):
                    fig = go.Figure(go.Histogram(
                        x=peers["salary_mid"].clip(upper=500_000), xbins=dict(start=20_000, end=510_000, size=10_000),
                        marker=dict(color=BLUE_LIGHT, line=dict(color=CARD, width=1)),
                        hovertemplate="%{y:,} postings<extra></extra>"))
                    fig.add_vrect(x0=low, x1=high, fillcolor=ORANGE, opacity=0.12, line_width=0)
                    fig.add_vline(x=np.exp(pred_log), line_color=ORANGE, line_width=2.5)
                    fig.add_annotation(x=np.exp(pred_log), y=1, yref="paper", yanchor="bottom", showarrow=False,
                                       text=f"Estimate {usd(np.exp(pred_log))}", font=dict(color=INK, size=12))
                    fig.update_layout(height=280, bargap=0.04, margin=dict(t=28),
                                      xaxis=dict(tickformat="$~s", title="Yearly salary (USD)"),
                                      yaxis=dict(title="Number of job postings"))
                    show(fig)
            # Job types that usually pay by the hour have few yearly salaries to learn from,
            # so the estimate for them is less reliable. Say so instead of hiding it.
            family_n = int((all_salaried["role_family"] == fam).sum())
            if family_n < 1000:
                st.warning(f"Only {family_n:,} {fam} postings list a yearly salary (many jobs of this type pay "
                           "by the hour), so this estimate is less reliable than for office and tech roles.")
            if employment in ("Part Time", "Internship", "Temporary"):
                st.caption("Note: the estimate is a full-year salary figure, as advertised by employers.")
            if company == "Not listed":
                st.info("Tip: choosing a company makes the estimate more precise, because pay differs a lot "
                        "between employers.")
            else:
                same = all_salaried[all_salaried["company"] == company]
                same = same.assign(match=same["role_family"].eq(fam)).sort_values(
                    ["match", "salary_mid"], ascending=[False, False]).head(8)
                if len(same):
                    st.markdown(f"**Real job postings at {company} that show a salary** (similar jobs first)")
                    st.dataframe(same[["title", "location", "salary_min", "salary_max", "url"]],
                                 width="stretch", hide_index=True,
                                 column_config={"title": "Title", "location": "Location",
                                                "salary_min": st.column_config.NumberColumn("Min", format="$%d"),
                                                "salary_max": st.column_config.NumberColumn("Max", format="$%d"),
                                                "url": st.column_config.LinkColumn("Posting", display_text="open")})

        st.markdown("### How accurate is the model?")
        st.caption(f"Scores come from {meta['n_test']:,} job postings the model never saw during training "
                   "(the last tile uses a separate test where whole companies were left out). "
                   "**R²** is the share of the salary differences between jobs that the model explains "
                   "(1.0 = perfect, 0 = no better than guessing the average).")
        k1, k2, k3, k4, k5 = st.columns(5)
        kpi(k1, f"{test['r2']:.2f}", "R² (accuracy score)", f"explains {test['r2']:.0%} of salary differences")
        kpi(k2, f"${test['mae']:,.0f}", "Average error", "average distance from the real salary")
        kpi(k3, f"{test['median_ape']:.0%}", "Typical % error", "half of estimates are closer than this")
        kpi(k4, f"{test['interval_coverage']:.0%}", "Range hit rate", "real salary inside the likely range")
        kpi(k5, f"{unseen['r2']:.2f}", "R² for new companies", "companies absent from training")
        st.write("")

        left, right = st.columns(2)
        with left:
            comp = pd.DataFrame(meta["comparison"]).T.rename_axis("model").reset_index().sort_values("cv_r2")
            with card("Which model predicts best?",
                      "Four approaches were tested on the same data (higher R² = more accurate). The thin black line "
                      "shows how much the score varied across 5 test rounds. Dark blue = the model chosen."):
                fig = go.Figure(go.Bar(
                    x=comp["cv_r2"], y=comp["model"], orientation="h",
                    marker_color=[BLUE if m == meta["model"] else BLUE_LIGHT for m in comp["model"]],
                    error_x=dict(type="data", array=comp["cv_r2_std"], color=INK_2, thickness=1.5, width=4),
                    text=comp["cv_r2"].map(lambda v: f"R² {v:.3f}" if v > 0.1 else ""), textposition="inside",
                    insidetextanchor="start", textfont=dict(color=INK),
                    customdata=comp[["cv_mae", "test_r2", "test_mae"]],
                    hovertemplate="Average error: %{customdata[0]:$,.0f}<br>"
                                  "On unseen test jobs: R² %{customdata[1]:.3f}, error %{customdata[2]:$,.0f}"
                                  "<extra></extra>"))
                fig.update_layout(height=300, yaxis=dict(gridcolor="rgba(0,0,0,0)"),
                                  xaxis=dict(title="R² (higher = more accurate)",
                                             range=[min(0, comp["cv_r2"].min()), 1]))
                table = comp[["model", "cv_r2", "cv_r2_std", "cv_mae", "test_r2", "test_mae", "seconds"]].iloc[::-1]
                table.columns = ["Model", "CV R²", "CV R² SD", "CV MAE ($)", "Test R²", "Test MAE ($)", "Fit time (s)"]
                show(fig, table)

            imp = pd.Series(meta["permutation_importance"]).rename_axis("input").reset_index(name="drop")
            imp["input"] = imp["input"].str.replace("_", " ").str.capitalize()
            imp["after"] = test["r2"] - imp["drop"]
            imp["_text"] = imp["drop"].map(lambda v: f"−{v:.3f}")
            with card("What information matters most?",
                      "How much accuracy (R²) drops when one piece of information is scrambled. "
                      "A bigger drop means the model relies on it more."):
                fig = ranked_bar(imp, "drop", "input",
                                 f"Without it, R² falls from {test['r2']:.2f} to %{{customdata[1]:.2f}}", 340)
                fig.update_layout(xaxis=dict(title="Accuracy lost (drop in R²)"))
                show(fig, imp.drop(columns=["_text", "after"]).rename(columns={"input": "Input", "drop": "R² drop"}))
        with right:
            sample = pd.DataFrame(meta["test_sample"])
            sample["error"] = sample["predicted"] / sample["actual"] - 1
            with card("Estimated vs. real salaries",
                      f"Each dot is one of {len(sample):,} jobs the model never saw. Dots on the diagonal line "
                      "were estimated exactly; the closer the cloud hugs the line, the better."):
                fig = go.Figure()
                fig.add_scatter(x=sample["actual"], y=sample["predicted"], mode="markers",
                                marker=dict(color=BLUE, size=7, opacity=0.35),
                                customdata=sample[["role_family", "error"]],
                                hovertemplate="%{customdata[0]}<br>Real %{x:$,.0f} · estimated %{y:$,.0f}<br>"
                                              "Off by %{customdata[1]:+.0%}<extra></extra>")
                lo, hi = 25_000, 700_000
                fig.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color=INK_2, width=1.5),
                                hoverinfo="skip")
                ticks = [30_000, 50_000, 100_000, 200_000, 300_000, 500_000]
                axis = dict(type="log", range=[np.log10(lo), np.log10(hi)], tickvals=ticks,
                            ticktext=[usd(t) for t in ticks])
                fig.update_layout(height=470, xaxis=dict(**axis, title="Real salary"),
                                  yaxis=dict(**axis, title="Model's estimate"))
                show(fig)

            fam_acc = pd.DataFrame(meta["by_role_family"]).sort_values("mae")
            with card("Accuracy for each type of job", "Scores on jobs the model never saw, sorted by average error"):
                st.dataframe(fam_acc, width="stretch", hide_index=True,
                             column_config={"role_family": "Type of job",
                                            "n": st.column_config.NumberColumn("Jobs tested", format="%d"),
                                            "r2": st.column_config.NumberColumn("R²", format="%.2f"),
                                            "mae": st.column_config.NumberColumn("Average error", format="$%d"),
                                            "median_actual": st.column_config.NumberColumn("Typical salary",
                                                                                           format="$%d")})

# ══ About the data ════════════════════════════════════════════════════════════
with tab_about:
    counts = summary["salary_status_counts"]
    labels = {"usable": "Annual USD salary (used in salary charts)", "no_salary": "No salary listed",
              "not_annual_usd": "Salary in another currency or per hour/month",
              "implausible_amount": "Midpoint outside $20k–$1M",
              "range_too_wide": "Range wider than 4×",
              "invalid_range": "Invalid range (e.g. minimum above maximum)"}
    funnel = pd.DataFrame({"Salary information": [labels.get(k, k) for k in counts],
                           "Postings": list(counts.values())}).sort_values("Postings", ascending=False)
    funnel["Share"] = (funnel["Postings"] / funnel["Postings"].sum()).map(lambda v: f"{v:.1%}")
    funnel["Postings"] = funnel["Postings"].map(lambda v: f"{v:,}")

    # Accuracy of the job-type rules, shown once titles have been hand-labeled.
    label_check_md = (
        f"* **Job-type labels** were checked against the department each employer chose for the posting: on "
        f"{label_check['n_checked']:,} postings from {label_check['n_companies']:,} companies, the title-based job "
        f"type matched the department {label_check['agreement']:.0%} of the time (95% CI "
        f"{label_check['ci_low']:.1%}–{label_check['ci_high']:.1%}). Cross-functional types (operations, "
        "consulting, management) sit in every department, so they can't be checked this way.\n"
        if label_check else "")

    left, right = st.columns([1, 1], gap="large")
    with left:
        st.markdown(f"""
### Source
**{summary['source']}**: {summary['raw_postings']:,} job postings collected from public company career
pages (Greenhouse, Lever, Ashby, SmartRecruiters, Workable) across {summary['countries']} countries.

### How the data was prepared
* Every posting counts toward posting volumes, role mix and skill demand.
* Salary charts use postings that advertise pay **per year in USD**, so all figures are directly
  comparable without currency or hours-per-year conversions.
* Each salary is the **midpoint** of the advertised range. Midpoints outside \\$20k–\\$1M and ranges
  wider than 4× are set aside as outliers.
* **Role family** and **seniority** are derived from title keywords with transparent rules
  (`pipeline/features.py`).

### How the model works
* Predicts the log of the salary midpoint from title words, company, industry, department, city,
  country, employment type, remote flag, skills, and title-derived seniority and role family.
* Postings repeated across many cities with an identical range are collapsed
  ({meta['n_usable_salaries']:,} → {meta['n_deduplicated']:,}) so the test set only contains new jobs.
* Candidate models are compared by 5-fold cross-validation on 80% of the data; the winner is scored once on
  the remaining 20%, and again on companies it has never seen.

### How uncertainty is measured
* **Confidence intervals** for medians use a *company cluster bootstrap* (1,000 replicates): each replicate
  resamples whole companies, because postings from the same company are not independent.
* **Remote vs. on-site** is estimated with regressions of log salary that add controls step by step (job type,
  seniority, country, employment type, then company fixed effects), with company-clustered standard errors.
{label_check_md}
### Scope
* Salaries are the ranges employers advertise in their postings.
* Most salaried postings are from the US, where pay-transparency laws require ranges.
* The data is a snapshot of roles open in {pd.Timestamp(summary['posted_max']):%B %Y}.
* Skill–salary relationships show association, not cause and effect.
""" if meta else "Model metadata missing.")
    with right:
        st.markdown("### Salary information in postings")
        st.dataframe(funnel, width="stretch", hide_index=True)
        st.markdown("### Browse the data")
        st.caption("Postings matching the sidebar filters (first 2,000 rows).")
        st.dataframe(df[["title", "company", "location", "role_family", "seniority", "salary_mid", "posted", "url"]]
                     .head(2000),
                     width="stretch", hide_index=True, height=420,
                     column_config={"title": st.column_config.TextColumn("Title", width="medium"),
                                    "company": "Company", "location": "Location",
                                    "role_family": "Role family", "seniority": "Seniority",
                                    "salary_mid": st.column_config.NumberColumn("Salary (mid)", format="$%d"),
                                    "posted": st.column_config.DateColumn("Posted"),
                                    "url": st.column_config.LinkColumn("Posting", display_text="open")})

# ══ Footer ════════════════════════════════════════════════════════════════════
st.markdown(f'<div class="footer">Built by <b>{AUTHOR}</b> · Python · pandas · scikit-learn · Plotly · '
            f'Streamlit · <a href="{LINKS["GitHub"]}" target="_blank">Source code</a></div>',
            unsafe_allow_html=True)
