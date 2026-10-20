"""Step 2 of 3: train, compare, and evaluate the salary prediction model.

Target   : log of the advertised annual USD salary midpoint (salaries are
           right-skewed, so a log target makes errors proportional).
Features : title text, company, industry, department, country, city,
           employment type, remote flag, inferred seniority/role family, skills.
Protocol : 1. de-duplicate identical postings so copies cannot leak across splits
           2. hold out 20% as an untouched test set
           3. compare candidates with 5-fold cross-validation on the other 80%
           4. score the winner once on the test set (plus an unseen-company test)
           5. refit the winner on all data and save it for the dashboard

Outputs: models/salary_model.pkl, models/model_meta.json
Run from the project root:  python models/train.py
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, KFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, TargetEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
from features import model_frame  # noqa: E402  (path must be set first)

DATA = ROOT / "data" / "jobs_clean.parquet"
MODEL = ROOT / "models" / "salary_model.pkl"
META = ROOT / "models" / "model_meta.json"
SEED = 42
INTERVAL = (0.10, 0.90)   # 80% prediction interval


# ── Candidate models ──────────────────────────────────────────────────────────
def ridge_full():
    """Linear model on sparse text + one-hot features (the expected winner)."""
    features = ColumnTransformer([
        # Unigrams + bigrams so "data engineer" and "staff engineer" are distinct signals.
        ("title", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), "title"),
        ("department", TfidfVectorizer(min_df=2), "department"),
        # Skills are already clean tokens; a binary 0/1 flag per skill.
        ("skills", TfidfVectorizer(token_pattern=r"[^ ]+", binary=True, use_idf=False, norm=None), "skills"),
        # Rare categories (seen once) are pooled so they cannot be memorized.
        ("categorical", OneHotEncoder(handle_unknown="ignore", min_frequency=2),
         ["company", "industry", "country", "city", "employment_type", "seniority", "role_family", "remote"]),
    ])
    return Pipeline([("features", features), ("ridge", Ridge(alpha=1.0))])


def ridge_title_location():
    """The project's previous feature set: title, location, seniority, type, remote."""
    features = ColumnTransformer([
        ("title", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), "title"),
        ("categorical", OneHotEncoder(handle_unknown="ignore", min_frequency=2),
         ["country", "city", "employment_type", "seniority", "remote"]),
    ])
    return Pipeline([("features", features), ("ridge", Ridge(alpha=1.0))])


def gradient_boosting():
    """Non-linear tree ensemble; high-cardinality columns are target-encoded."""
    features = ColumnTransformer([
        ("target", TargetEncoder(target_type="continuous", random_state=SEED),
         ["company", "title", "city", "industry", "department"]),
        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
         ["country", "employment_type", "seniority", "role_family", "remote"]),
    ])
    model = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.06, max_leaf_nodes=63,
                                          min_samples_leaf=20, l2_regularization=1.0,
                                          random_state=SEED)
    return Pipeline([("features", features), ("gbm", model)])


CANDIDATES = {
    "Median baseline": lambda: DummyRegressor(strategy="median"),
    "Ridge: title + location only": ridge_title_location,
    "Gradient boosting (target-encoded)": gradient_boosting,
    "Ridge: full feature set": ridge_full,
}


# ── Metrics ───────────────────────────────────────────────────────────────────
def score(y_log, pred_log):
    """Metrics on the dollar scale (what a reader cares about) and the log scale."""
    actual, predicted = np.exp(y_log), np.exp(pred_log)
    return {
        "r2": round(float(r2_score(actual, predicted)), 4),
        "r2_log": round(float(r2_score(y_log, pred_log)), 4),
        "mae": round(float(mean_absolute_error(actual, predicted)), 0),
        "median_ape": round(float(np.median(np.abs(predicted / actual - 1))), 4),
    }


# ── Training ──────────────────────────────────────────────────────────────────
def load_training_data():
    jobs = pd.read_parquet(DATA)
    jobs = jobs[jobs["salary_mid"].notna()]
    n_usable = len(jobs)
    # The same job is often posted in several cities with an identical salary.
    # Keeping copies would let the model "see" test rows during training.
    jobs = jobs.drop_duplicates(["company", "title", "salary_min", "salary_max"]).reset_index(drop=True)
    return jobs, n_usable


def train():
    jobs, n_usable = load_training_data()
    X = model_frame(jobs)
    y = np.log(jobs["salary_mid"].to_numpy())
    print(f"Usable salaries: {n_usable:,} -> {len(X):,} after de-duplication", flush=True)

    idx_train, idx_test = train_test_split(np.arange(len(X)), test_size=0.20, random_state=SEED)
    X_train, X_test, y_train, y_test = X.iloc[idx_train], X.iloc[idx_test], y[idx_train], y[idx_test]
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)

    # 1) Cross-validated comparison on the training split only.
    comparison, oof_predictions = {}, {}
    for name, factory in CANDIDATES.items():
        started = time.time()
        oof = cross_val_predict(factory(), X_train, y_train, cv=cv)
        oof_predictions[name] = oof
        fold_r2 = [r2_score(np.exp(y_train[v]), np.exp(oof[v])) for _, v in cv.split(X_train)]
        comparison[name] = {**{f"cv_{k}": v for k, v in score(y_train, oof).items()},
                            "cv_r2_std": round(float(np.std(fold_r2)), 4),
                            "seconds": round(time.time() - started, 1)}
        print(f"{name:<38} CV R2={comparison[name]['cv_r2']:.3f}  "
              f"CV MAE=${comparison[name]['cv_mae']:,.0f}", flush=True)

    winner = min(comparison, key=lambda n: comparison[n]["cv_mae"])
    print(f"Selected: {winner}", flush=True)

    # 2) Prediction interval from out-of-fold residuals (split-conformal style):
    #    the 10th/90th percentiles of log errors become a multiplicative range.
    residuals = y_train - oof_predictions[winner]
    lo_q, hi_q = np.quantile(residuals, INTERVAL)

    # 3) One-time evaluation on the untouched test set.
    model = CANDIDATES[winner]().fit(X_train, y_train)
    test_pred = model.predict(X_test)
    test_metrics = score(y_test, test_pred)
    inside = (y_test >= test_pred + lo_q) & (y_test <= test_pred + hi_q)
    test_metrics["interval_coverage"] = round(float(inside.mean()), 4)
    for name, factory in CANDIDATES.items():  # test scores for the table, never used for selection
        pred = test_pred if name == winner else factory().fit(X_train, y_train).predict(X_test)
        comparison[name].update({f"test_{k}": v for k, v in score(y_test, pred).items()})
    print(f"Test R2={test_metrics['r2']:.3f}  MAE=${test_metrics['mae']:,.0f}  "
          f"80% interval coverage={test_metrics['interval_coverage']:.1%}", flush=True)

    # Accuracy per role family on the test set.
    test_frame = X_test.assign(actual=np.exp(y_test), predicted=np.exp(test_pred))
    by_family = (test_frame.groupby("role_family")
                 .apply(lambda g: pd.Series({
                     "n": len(g),
                     "r2": r2_score(g["actual"], g["predicted"]) if len(g) > 1 else np.nan,
                     "mae": mean_absolute_error(g["actual"], g["predicted"]),
                     "median_actual": g["actual"].median()}), include_groups=False)
                 .query("n >= 50").round(3).sort_values("mae").reset_index())

    # 4) Harder test: companies the model has never seen at all.
    g_train, g_test = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
                           .split(X, groups=X["company"]))
    unseen = CANDIDATES[winner]().fit(X.iloc[g_train], y[g_train])
    unseen_metrics = score(y[g_test], unseen.predict(X.iloc[g_test]))
    print(f"Unseen-company R2={unseen_metrics['r2']:.3f}", flush=True)

    importance = permutation_importance_by_column(model, X_test, y_test)
    print("Permutation importance:", importance, flush=True)

    # 5) Refit on every row for the deployed model.
    final = CANDIDATES[winner]().fit(X, y)
    skill_effects = skill_coefficients(final, X) if winner.startswith("Ridge") else []
    strip_for_pickle(final)

    # Sample of test predictions for the predicted-vs-actual chart (keeps the JSON small).
    sample = test_frame.sample(n=min(3000, len(test_frame)), random_state=SEED)

    # Lookup tables that let the estimator form auto-fill sensible defaults.
    company_industry = (jobs.dropna(subset=["industry"]).groupby("company")["industry"]
                        .agg(lambda s: s.mode().iat[0]).astype(str).to_dict())
    city_country = (jobs.dropna(subset=["city", "country"]).groupby("city")["country"]
                    .agg(lambda s: s.mode().iat[0]).astype(str).to_dict())

    meta = {
        "model": winner,
        "target": "log(advertised annual USD salary midpoint)",
        "n_usable_salaries": n_usable,
        "n_deduplicated": len(X),
        "n_train": len(idx_train),
        "n_test": len(idx_test),
        "test": test_metrics,
        "unseen_company_test": unseen_metrics,
        "interval": {"coverage_target": INTERVAL[1] - INTERVAL[0],
                     "log_low": round(float(lo_q), 4), "log_high": round(float(hi_q), 4)},
        "comparison": comparison,
        "by_role_family": by_family.to_dict(orient="records"),
        "permutation_importance": importance,
        "skill_effects": skill_effects,
        "test_sample": sample[["actual", "predicted", "role_family"]].round(0).to_dict(orient="list"),
        "company_industry": company_industry,
        "city_country": city_country,
    }

    MODEL.write_bytes(pickle.dumps({"model": final, "name": winner}))
    META.write_text(json.dumps(meta, indent=1, default=float), encoding="utf-8")
    print(f"Saved {MODEL.name} ({MODEL.stat().st_size / 1e6:.1f} MB) and {META.name}")


def permutation_importance_by_column(model, X_test, y_test, repeats=3):
    """How much test R² (dollar scale) drops when one input column is shuffled.

    Shuffling breaks that column's link to salary while keeping its
    distribution, so a large drop means the model relies on that input.
    """
    # Seniority and role family are derived from the title, so they are
    # shuffled together with it; otherwise the title would look unimportant.
    groups = {"title": ["title", "seniority", "role_family"], "company": ["company"],
              "city": ["city"], "country": ["country"], "industry": ["industry"],
              "department": ["department"], "skills": ["skills"],
              "employment_type": ["employment_type"], "remote": ["remote"]}
    rng = np.random.default_rng(SEED)
    base = r2_score(np.exp(y_test), np.exp(model.predict(X_test)))
    drops = {}
    for label, cols in groups.items():
        scores = []
        for _ in range(repeats):
            order = rng.permutation(len(X_test))
            shuffled = X_test.copy()
            shuffled[cols] = X_test[cols].to_numpy()[order]
            scores.append(base - r2_score(np.exp(y_test), np.exp(model.predict(shuffled))))
        drops[label] = round(float(np.mean(scores)), 4)
    return dict(sorted(drops.items(), key=lambda kv: -kv[1]))


def skill_coefficients(pipeline, X):
    """Ridge coefficient for each skill flag, as an approximate % salary effect.

    Skills are binary (0/1) features and the target is log salary, so
    exp(coef) - 1 is the % change in predicted pay when a skill is listed,
    holding title, company, location and everything else fixed.
    """
    names = pipeline.named_steps["features"].get_feature_names_out()
    is_skill = np.char.startswith(names.astype(str), "skills__")
    effects = pd.DataFrame({"skill": np.char.replace(names[is_skill].astype(str), "skills__", ""),
                            "effect": np.expm1(pipeline.named_steps["ridge"].coef_[is_skill])})
    # Number of training postings listing each skill, so rare skills can be hidden.
    counts = X["skills"].str.split().explode().value_counts()
    effects["postings"] = effects["skill"].map(counts).fillna(0).astype(int)
    return effects.round(4).to_dict(orient="records")


def strip_for_pickle(pipeline):
    """Drop TfidfVectorizer.stop_words_ (the pruned vocabulary): unused at
    prediction time but it can make the pickle many megabytes larger."""
    if not hasattr(pipeline, "named_steps"):
        return
    for _, transformer, _ in pipeline.named_steps["features"].transformers_:
        if isinstance(transformer, TfidfVectorizer) and hasattr(transformer, "stop_words_"):
            del transformer.stop_words_


if __name__ == "__main__":
    train()
