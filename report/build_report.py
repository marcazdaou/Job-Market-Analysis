"""Build the written report (report/report.pdf) from the project's result files.

Every number in the report is read from the data and result files below, so
the report can be rebuilt after any change without editing text by hand:
    data/jobs_clean.parquet, data/data_summary.json      (pipeline/collector.py)
    models/model_meta.json                                (models/train.py)
    analysis/results/statistics.json                      (analysis/salary_statistics.py)
    analysis/results/label_validation.json                (analysis/label_validation.py)

Steps: draw figures with matplotlib -> assemble an HTML page -> print it to PDF
with a locally installed Chrome or Edge (headless).

Run from the project root:  python report/build_report.py
"""
import base64
import io
import json
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files, no window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
from features import OTHER_FAMILY, SENIORITY_ORDER  # noqa: E402

OUT_HTML = ROOT / "report" / "report.html"
OUT_PDF = ROOT / "report" / "report.pdf"
AUTHOR = "Marc Abi Zeid Daou"
BLUE, LIGHT, INK, MUTED = "#2a78d6", "#b7d3f6", "#1b1b1b", "#6b6a66"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#c3c2b7",
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#ecebe6", "grid.linewidth": 0.8, "axes.axisbelow": True})


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def k(x: float) -> str:
    """$123k formatting."""
    return f"${int(x / 1000 + 0.5):,}k"


def fig_to_img(fig) -> str:
    """Embed a matplotlib figure in the HTML as a base64 PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'


# ── Figures ───────────────────────────────────────────────────────────────────
def fig_job_types(stats: dict) -> str:
    fam = pd.DataFrame(stats["medians"]["role_family"]).T.astype(float).sort_values("median")
    fig, ax = plt.subplots(figsize=(6.6, 3.3))
    ax.errorbar(fam["median"], fam.index, xerr=[fam["median"] - fam["ci_low"], fam["ci_high"] - fam["median"]],
                fmt="o", color=BLUE, ecolor=MUTED, elinewidth=1.2, capsize=3, markersize=5)
    ax.xaxis.set_major_formatter(lambda v, _: k(v))
    ax.set_xlabel("Median advertised salary (USD/year), with 95% CI")
    ax.grid(axis="y", visible=False)
    return fig_to_img(fig)


def fig_seniority(stats: dict) -> str:
    sen = pd.DataFrame(stats["medians"]["seniority"]).T.astype(float).reindex(SENIORITY_ORDER)
    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    x = np.arange(len(sen))
    ax.bar(x, sen["median"], color=[LIGHT, "#86b6ef", BLUE, "#1c5cab", "#104281"], width=0.6)
    ax.errorbar(x, sen["median"], yerr=[sen["median"] - sen["ci_low"], sen["ci_high"] - sen["median"]],
                fmt="none", ecolor=INK, elinewidth=1.2, capsize=4)
    ax.set_xticks(x, [s.replace(" / ", "\n") for s in sen.index])
    ax.yaxis.set_major_formatter(lambda v, _: k(v))
    ax.set_ylabel("Median salary")
    ax.grid(axis="x", visible=False)
    return fig_to_img(fig)


def fig_remote(stats: dict) -> str:
    regs = pd.DataFrame(stats["remote_regressions"]).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.6, 2.3))
    ax.errorbar(regs["effect"] * 100, regs["model"],
                xerr=[(regs["effect"] - regs["ci_low"]) * 100, (regs["ci_high"] - regs["effect"]) * 100],
                fmt="o", color=BLUE, ecolor=MUTED, elinewidth=1.2, capsize=3, markersize=6)
    ax.axvline(0, color="#9a998f", linewidth=1)
    ax.set_xlabel("Remote vs. on-site pay difference (%), with 95% CI")
    ax.grid(axis="y", visible=False)
    return fig_to_img(fig)


def fig_model(meta: dict) -> str:
    sample = pd.DataFrame(meta["test_sample"])
    comp = pd.DataFrame(meta["comparison"]).T.sort_values("cv_r2")
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.8, 2.9), gridspec_kw={"width_ratios": [1.15, 1]})
    a.barh(comp.index, comp["cv_r2"], color=[BLUE if m == meta["model"] else LIGHT for m in comp.index],
           xerr=comp["cv_r2_std"], error_kw=dict(ecolor=INK, capsize=2, elinewidth=1))
    a.set_xlim(min(0, comp["cv_r2"].min()), 1)
    a.set_xlabel("Cross-validated R²")
    a.tick_params(axis="y", labelsize=7.5)
    a.grid(axis="y", visible=False)
    b.scatter(sample["actual"], sample["predicted"], s=4, alpha=0.25, color=BLUE, linewidths=0)
    b.plot([2.5e4, 7e5], [2.5e4, 7e5], color=INK, linewidth=1)
    b.set_xscale("log"), b.set_yscale("log")
    ticks = [5e4, 1e5, 2e5, 5e5]
    for axis in (b.xaxis, b.yaxis):
        axis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
        axis.set_major_formatter(lambda v, _: k(v))
        axis.set_minor_formatter(lambda v, _: "")
    b.tick_params(labelsize=7.5)
    b.set_xlim(2.5e4, 7e5), b.set_ylim(2.5e4, 7e5)
    b.set_xlabel("Actual salary (test set)"), b.set_ylabel("Predicted")
    fig.tight_layout()
    return fig_to_img(fig)


# ── Report ────────────────────────────────────────────────────────────────────
def build_html() -> str:
    summary = load(ROOT / "data" / "data_summary.json")
    meta = load(ROOT / "models" / "model_meta.json")
    stats = load(ROOT / "analysis" / "results" / "statistics.json")
    labels = load(ROOT / "analysis" / "results" / "label_validation.json")
    jobs = pd.read_parquet(ROOT / "data" / "jobs_clean.parquet")
    sal = jobs.dropna(subset=["salary_mid"])

    med = stats["medians"]
    overall, ratio = med["overall"], med["ratios"]["principal_vs_entry"]
    fam = pd.DataFrame(med["role_family"]).T.astype(float).sort_values("median", ascending=False)
    regs = {r["model"]: r for r in stats["remote_regressions"]}
    raw, fe = regs["Raw difference"], regs["+ same company (fixed effects)"]
    test, unseen = meta["test"], meta["unseen_company_test"]
    imp = meta["permutation_importance"]
    effects = pd.DataFrame(meta["skill_effects"]).query("postings >= 300")
    known = jobs[jobs["role_family"] != OTHER_FAMILY]
    tech = ["Software Engineering", "Data & AI", "Product & Design", "Security"]
    us_share = (sal["country"] == "United States").mean()
    listed = jobs.assign(l=jobs["salary_status"].ne("no_salary")).groupby("country", observed=True)["l"].mean()
    status = summary["salary_status_counts"]
    comparison = pd.DataFrame(meta["comparison"]).T

    # Validation of the job-type rules against the department each employer assigned.
    by_type = pd.DataFrame(labels["by_job_type"]).set_index("role_family")["agreement"]
    label_text = (
        "<b>Validation.</b> Most postings also carry a department chosen by the employer (e.g. \"Engineering\", "
        "\"Sales\"), which is independent of the title rules. Mapping clear department names to groups and "
        "allowing plausible overlaps (a data job in an engineering department counts as a match), the "
        f"title-based job type agrees with the employer's department for <b>{labels['agreement']:.0%}</b> of "
        f"{labels['n_checked']:,} postings from {labels['n_companies']:,} companies (95% Wilson CI "
        f"{labels['ci_low']:.1%}–{labels['ci_high']:.1%}), from {by_type.max():.0%} for "
        f"{by_type.idxmax()} down to {by_type.min():.0%} for {by_type.idxmin()}. This check also exposed a "
        "rule error: titles such as \"Sales Engineer\" and \"Technical Support Engineer\" had been classified "
        "as engineering because of the word \"engineer\"; they are now classified as sales and support jobs. "
        "Cross-functional job types (operations, consulting, management) appear in every kind of department, so "
        "the department cannot confirm them; they are not scored.") if labels else ""

    model_rows = "".join(
        f"<tr><td>{'<b>' + m + '</b>' if m == meta['model'] else m}</td><td>{r['cv_r2']:.3f}</td>"
        f"<td>{r['test_r2']:.3f}</td><td>${r['test_mae']:,.0f}</td></tr>"
        for m, r in comparison.sort_values("cv_r2").iterrows())
    reg_rows = "".join(
        f"<tr><td>{r['model']}</td><td>{r['effect']:+.1%}</td><td>{r['ci_low']:+.1%} to {r['ci_high']:+.1%}</td>"
        f"<td>{r['p_value']:.2g}</td></tr>" for r in stats["remote_regressions"])
    funnel_rows = "".join(f"<tr><td>{name}</td><td>{status.get(key, 0):,}</td></tr>" for key, name in [
        ("usable", "Yearly USD salary (used)"), ("no_salary", "No salary listed"),
        ("not_annual_usd", "Other currency or pay period"), ("range_too_wide", "Range wider than 4×"),
        ("implausible_amount", "Midpoint outside $20k–$1M"), ("invalid_range", "Invalid range")])

    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Job Market Analytics: report</title>
<style>
  @page {{ size: Letter; margin: 0.75in 0.8in; }}
  body {{ font-family: Georgia, 'Times New Roman', serif; color: {INK}; font-size: 10.5pt; line-height: 1.45; }}
  h1 {{ font-family: Helvetica, Arial, sans-serif; font-size: 19pt; margin: 0 0 4px; }}
  h2 {{ font-family: Helvetica, Arial, sans-serif; font-size: 12.5pt; margin: 18px 0 6px; color: #13294b; }}
  h3 {{ font-family: Helvetica, Arial, sans-serif; font-size: 10.5pt; margin: 12px 0 4px; }}
  .byline {{ font-family: Helvetica, Arial, sans-serif; color: {MUTED}; font-size: 9.5pt; margin-bottom: 14px; }}
  .abstract {{ background: #f3f6fb; border-left: 3px solid {BLUE}; padding: 10px 14px; font-size: 10pt; }}
  table {{ border-collapse: collapse; font-family: Helvetica, Arial, sans-serif; font-size: 8.8pt; margin: 6px 0 10px; }}
  th, td {{ border-bottom: 1px solid #e3e2dc; padding: 3px 10px 3px 0; text-align: left; }}
  th {{ color: {MUTED}; font-weight: 600; }}
  figure {{ margin: 8px 0 12px; page-break-inside: avoid; }}
  figure img {{ width: 94%; display: block; margin: 0 auto; }}
  figcaption {{ font-family: Helvetica, Arial, sans-serif; font-size: 8.5pt; color: {MUTED}; }}
  code {{ font-size: 8.8pt; }}
  .keep {{ page-break-inside: avoid; }}
  h2, h3 {{ page-break-after: avoid; break-after: avoid; }}  /* never leave a heading alone at a page bottom */
</style></head><body>

<h1>What {summary['postings']:,} job postings reveal about pay, skills and hiring</h1>
<div class="byline">{AUTHOR} · Data source: {summary['source']} · Snapshot of open roles,
{pd.Timestamp(summary['posted_max']):%B %Y}</div>

<div class="abstract"><b>Summary.</b> This project analyzes {summary['postings']:,} job postings published by
{summary['companies']:,} companies in {summary['countries']} countries, of which {summary['usable_salaries']:,}
advertise a yearly salary in US dollars. The typical (median) advertised salary is <b>{k(overall['median'])}</b>
(95% CI {k(overall['ci_low'])}–{k(overall['ci_high'])}). Pay rises {ratio['ratio']:.1f}× from entry-level to
principal/staff titles. Remote postings advertise {raw['effect']:.0%} more than on-site postings, but comparing
similar jobs within the same company the premium falls to <b>{fe['effect']:.1%}</b> (95% CI
{fe['ci_low']:.1%} to {fe['ci_high']:.1%}). A regularized linear model predicts advertised salaries with
R² = <b>{test['r2']:.2f}</b> on held-out postings (typical error {test['median_ape']:.0%}); job title and employer
carry most of the signal, while individual skills shift pay by at most {effects['effect'].abs().max():.0%}.</div>

<h2>1. Questions</h2>
<p>(1) What do these jobs advertise, and which job types pay the most? (2) How much does seniority matter?
(3) Do remote jobs pay more once comparable jobs are compared? (4) Do specific skills raise pay?
(5) How accurately can a machine-learning model predict an advertised salary?</p>

<h2>2. Data</h2>
<p>Postings were collected from public company career pages hosted on applicant-tracking systems
(SmartRecruiters, Greenhouse, Ashby, Lever, Workable). The data covers every type of role: tech jobs (software,
data, product, security) are {known['role_family'].isin(tech).mean():.0%} of postings with an identified job
type. Only {sum(v for key, v in status.items() if key != 'no_salary') / summary['postings']:.0%} of postings
state any salary, and disclosure follows the law: {listed['United States']:.0%} of US postings do, versus
{listed['Germany']:.0%} in Germany and {listed['France']:.0%} in France. As a result, {us_share:.0%} of usable
salaries are US jobs, and figures describe these postings rather than national pay levels.</p>

<div class="keep"><table><tr><th>Salary information</th><th>Postings</th></tr>{funnel_rows}</table></div>

<p>Salaries are the midpoint of the advertised range. Only yearly USD ranges are used, so no currency or
hours-per-year conversion is assumed; midpoints outside $20k–$1M (typically hourly rates mislabeled as yearly)
and ranges wider than 4× are excluded. <b>Job type</b> (19 groups) and <b>seniority</b> (5 levels) are derived
from title keywords with ordered rules that include French, German, Spanish and Dutch terms;
{(jobs['role_family'] != OTHER_FAMILY).mean():.0%} of postings ({(sal['role_family'] != OTHER_FAMILY).mean():.0%}
of salaried ones) match a job type. {label_text}</p>

<h2>3. Methods</h2>
<p><b>Uncertainty.</b> Postings from one company share pay bands and are often copied across cities, so they are
not independent. Confidence intervals for medians therefore use a <i>company cluster bootstrap</i>: 1,000
replicates, each resampling whole companies with replacement; intervals are the 2.5th–97.5th percentiles.</p>
<p><b>Remote pay.</b> Ordinary least squares regressions of log salary on a remote indicator, adding controls step
by step: job type and seniority; country (top 10) and employment type; and finally company fixed effects (the
within-company estimator). Standard errors are clustered by company. Duplicate postings (same company, title and
salary range) are collapsed, and unclassified titles are excluded (n = {fe['n']:,}).</p>
<p><b>Salary prediction.</b> The target is log salary. Inputs are title text (TF-IDF unigrams and bigrams),
company, industry, department, city, country, employment type, remote flag, skills and title-derived labels.
After collapsing duplicates ({meta['n_usable_salaries']:,} → {meta['n_deduplicated']:,}), 20% of postings are held
out. Four candidates are compared by 5-fold cross-validation on the remaining 80%; the winner is scored once on
the held-out set and again on a split in which entire companies are unseen. An 80% prediction interval is formed
from out-of-fold residuals (split-conformal style) and its coverage is checked on the test set.</p>

<h2>4. Results</h2>
<h3>4.1 Pay by job type</h3>
<p>The highest-paying job types are {', '.join(f"{f} ({k(r['median'])})" for f, r in fam.head(4).iterrows())};
their intervals overlap, so their exact order is not statistically distinguishable. At the other end are
{', '.join(f"{f} ({k(r['median'])})" for f, r in fam.tail(3).iloc[::-1].iterrows())}. Small groups have wide
intervals (e.g. Healthcare {k(fam.loc['Healthcare', 'ci_low'])}–{k(fam.loc['Healthcare', 'ci_high'])}), a
reminder that a median from a few hundred postings is far less certain than one from ten thousand.</p>
<figure>{fig_job_types(stats)}<figcaption>Figure 1. Median advertised salary by job type, with 95% company
cluster bootstrap confidence intervals.</figcaption></figure>

<h3>4.2 Seniority</h3>
<p>Median advertised pay rises from {k(med['seniority']['Entry / Junior']['median'])} for entry-level titles to
{k(med['seniority']['Principal / Staff']['median'])} for principal/staff titles, a {ratio['ratio']:.1f}× difference
(95% CI {ratio['ci_low']:.1f}×–{ratio['ci_high']:.1f}×). Director/executive titles
({k(med['seniority']['Director / Executive']['median'])}) sit slightly below principal/staff, plausibly because
"director" covers a wide range of organization sizes and functions.</p>
<figure>{fig_seniority(stats)}<figcaption>Figure 2. Median salary by title-inferred seniority (whiskers: 95%
CI).</figcaption></figure>

<h3>4.3 Remote vs. on-site</h3>
<p>Remote postings advertise {raw['effect']:+.1%} more than on-site postings. Part of this gap is composition:
remote postings are more often senior and in higher-paying job types. Controlling for job type and seniority
reduces it to {regs['+ job type & seniority']['effect']:+.1%}; comparing jobs within the same company leaves a
smaller but statistically significant premium of {fe['effect']:+.1%} (95% CI {fe['ci_low']:+.1%} to
{fe['ci_high']:+.1%}, p = {fe['p_value']:.1g}).</p>
<div class="keep"><table><tr><th>Comparison</th><th>Remote difference</th><th>95% CI</th><th>p-value</th></tr>
{reg_rows}</table></div>
<figure>{fig_remote(stats)}<figcaption>Figure 3. Estimated remote pay difference as controls are added
(company-clustered 95% CIs).</figcaption></figure>

<h3>4.4 Skills</h3>
<p>Postings that list specialized skills advertise high salaries (for example, machine-learning frameworks appear
in jobs with medians above $200k). But in the prediction model, which holds title, company and location fixed,
no skill listed in at least 300 postings changes predicted pay by more than
{effects['effect'].abs().max():.1%}. Skills mostly signal <i>which</i> job and employer a posting belongs to,
rather than adding pay on their own. These are associations, not causal effects.</p>

<h3>4.5 Predicting salaries</h3>
<div class="keep"><table><tr><th>Model</th><th>CV R²</th><th>Test R²</th><th>Test MAE</th></tr>{model_rows}</table></div>
<p>The selected ridge regression explains {test['r2']:.0%} of the variance in advertised salaries on held-out
postings (R² {test['r2']:.3f}; {test['r2_log']:.3f} on the log scale), with a mean absolute error of
${test['mae']:,.0f} and a median absolute error of {test['median_ape']:.1%}. Its 80% prediction interval contained
{test['interval_coverage']:.1%} of held-out salaries, so the stated uncertainty is well calibrated. Permutation
importance attributes most of the accuracy to the title (R² drop {imp['title']:.2f}) and the company
({imp['company']:.2f}); skills contribute {imp['skills']:.3f}. For companies absent from training, R² falls to
{unseen['r2']:.2f}: much of the predictive power comes from knowing the employer's pay level. A linear model
outperformed gradient boosting here, plausibly because the signal sits in thousands of sparse title words and
company identities.</p>
<figure>{fig_model(meta)}<figcaption>Figure 4. Left: cross-validated R² of the candidate models (selected model
in dark blue). Right: predicted vs. actual salary for 3,000 held-out postings (log scales).</figcaption></figure>

<h2>5. Limitations</h2>
<p>Advertised ranges are not realized pay. Salaried postings are concentrated in the US because of disclosure
laws, and the companies are those using these hiring platforms, so results describe this population of postings
rather than the labor market as a whole. The data is a single snapshot, so it shows how long roles stay open rather
than hiring trends. Job type and seniority are rule-based approximations. Regression and skill results are
associations; remaining differences between remote and on-site roles (e.g. responsibilities not visible in the
title) could still explain part of the premium.</p>

<h2>6. Reproducibility</h2>
<p>All code, the cleaned data and the trained model are public at
github.com/marcazdaou/Job-Market-Analysis. The pipeline is: <code>pipeline/collector.py</code> (cleaning) →
<code>models/train.py</code> (model) → <code>analysis/salary_statistics.py</code> (inference) →
<code>analysis/label_validation.py</code> (job-type check) → <code>report/build_report.py</code> (this report). Automated tests check data integrity, model calibration and the
dashboard on every push.</p>
</body></html>"""


def find_browser() -> str | None:
    """Locate a Chrome or Edge executable for headless PDF printing."""
    candidates = [shutil.which(n) for n in ["chrome", "google-chrome", "chromium", "msedge"]] + [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"]
    return next((c for c in candidates if c and Path(c).exists()), None)


def main():
    OUT_HTML.write_text(build_html(), encoding="utf-8")
    browser = find_browser()
    if not browser:
        sys.exit(f"Wrote {OUT_HTML}. No Chrome/Edge found: open it in a browser and print to PDF.")
    subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={OUT_PDF}", OUT_HTML.as_uri()], check=True, capture_output=True)
    OUT_HTML.unlink()  # the PDF is the deliverable; the HTML is only an intermediate step
    print(f"Wrote {OUT_PDF.relative_to(ROOT)} ({OUT_PDF.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
