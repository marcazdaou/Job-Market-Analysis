"""Shared, deterministic rules used by the pipeline, the model, and the dashboard.

Keeping these in one module guarantees that a title is classified the same way
when the data is cleaned, when the model is trained, and when a visitor asks
the dashboard for a salary estimate.
"""
import re

import numpy as np
import pandas as pd

# ── Salary quality rules ──────────────────────────────────────────────────────
# Only annual USD salaries are comparable, so every other currency/period is
# excluded from salary statistics (those postings still count as postings).
SALARY_FLOOR = 20_000        # below this, "annual" figures are almost always hourly rates
SALARY_CEILING = 1_000_000   # above this, figures are usually typos or currency errors
MAX_RANGE_RATIO = 4.0        # a $50k–$400k range carries too little information

# ── Seniority (inferred from title keywords) ─────────────────────────────────
SENIORITY_ORDER = ["Entry / Junior", "Mid / Unspecified", "Senior / Lead",
                   "Principal / Staff", "Director / Executive"]
_SENIORITY_RULES = [
    ("Director / Executive", r"\b(director|vp|vice president|chief|head of|c[eotf]o|president|partner)\b"),
    # "Staff Engineer" is a level, but "Staff Nurse"/"Staff Accountant" are not.
    ("Principal / Staff", r"\b(principal|staff(?! (nurse|accountant|assistant|writer|pharmacist|rn))|"
                          r"distinguished|fellow)\b"),
    ("Senior / Lead", r"\b(senior|sr|lead|iii|iv|supervisor)\b"),
    ("Entry / Junior", r"\b(junior|jr|entry|intern|internship|graduate|trainee|apprentice|assistant|associate|i)\b"),
]

# ── Role family (inferred from title keywords) ───────────────────────────────
# Order matters: the first matching rule wins, so specific families
# ("machine learning engineer") are checked before broad ones ("engineer"),
# and the generic "Management" catch-all comes last.
# Keywords include common French, German, Spanish and Dutch job words, because
# about a third of the postings are from non-English-speaking countries.
ROLE_FAMILIES = [
    ("Data & AI", r"\b(data (scientist|science|engineer\w*|analyst|analytics|architect|platform)|machine learning|"
                  r"ml|ai (engineer|researcher|scientist)|analytics|business intelligence|bi (analyst|developer|engineer)|"
                  r"business analyst|deep learning|nlp|computer vision|llm|statistician|quantitative|"
                  r"data\w* (ingenieur|ingénieur)|analyste? de données)\b"),
    ("Product & Design", r"\b(product manager|product owner|product lead|head of product|product director|"
                         r"product management|product design|designer\w*|ux|ui|user experience|user research\w*|"
                         r"creative director|graphiste)\b"),
    ("Security", r"\b(security|cyber\w*|infosec|penetration|appsec|sécurité|sicherheit\w*|beveilig\w*)\b"),
    ("Software Engineering", r"\b(software|developer|programmer|backend|back end|frontend|front end|full ?stack|"
                             r"ios|android|mobile engineer|web engineer|devops|sre|site reliability|platform engineer|"
                             r"infrastructure engineer|cloud|qa engineer|test engineer|sdet|engineering manager|"
                             r"solutions? architect\w*|architekt\w*|member of technical staff|it systems|"
                             r"systems administrator|dotnet|java|sap|salesforce developer|tech lead|technical lead|tpm|"
                             r"développeur\w*|developpeur\w*|entwickler\w*|desarrollador\w*|informati\w*|logiciel\w*|"
                             r"it)\b"),
    ("Research & Science", r"\b(scientist|research\w*|laboratory|lab|labor\w*|laboratorio|chemist\w*|biolog\w*|"
                           r"physicist|clinical research|postdoc\w*|chercheur|wissenschaftl\w*)\b"),
    ("Hardware & Engineering", r"\b(engineer|engineering|mechanical|electrical|civil|hardware|firmware|"
                               r"manufacturing|structural|chemical|avionics|propulsion|surveyor|bim|estimat\w*|"
                               r"ingénieur\w*|ingenieur\w*|ingeniero\w*|techniker\w*|konstrukt\w*)\b"),
    ("Sales & Business Dev.", r"\b(sales|account (executive|manager|management|partner|director|lead)|"
                              r"business development|bdr|sdr|partnerships?|solutions consultant|pre ?sales|"
                              r"territory|trader|underwriter|advisor|commercial\w*|comercial\w*|conseiller\w*|"
                              r"agent\w*|agente|representative|vertrieb\w*|versicherung\w*|dealer|"
                              r"außendienst|key account)\b"),
    ("Marketing & Comms", r"\b(marketing|marketeer|growth|brand|content|seo|sem|communications?|copywriter|"
                          r"social media|public relations|pr manager|demand generation|events?|campaign\w*)\b"),
    ("Customer Success & Support", r"\b(customer success|customer support|customer service|support specialist|"
                                   r"support engineer|client services|implementation|onboarding|technical support|"
                                   r"help ?desk|service delivery|client engagement|kundenservice|service client)\b"),
    ("Finance & Legal", r"\b(finance|financial|accountant|accounting|controller|fp&a|tax|audit\w*|treasury|payroll|"
                        r"bookkeep\w*|counsel|attorney|lawyer|legal|paralegal|compliance|risk|cost manager|"
                        r"comptab\w*|buchhalt\w*|contrôleur|juriste|bank\w*|credit|customs)\b"),
    ("People & HR", r"\b(recruit\w*|recrutement|talent|human resources|hr|people|hrbp|benefits|compensation|"
                    r"personal\w*referent|ressources humaines)\b"),
    ("Healthcare", r"\b(nurse|rn|lpn|physician|doctor|therapist|clinical|clinician|medical|pharmac\w*|dental|"
                   r"caregiver|care (assistant|worker|specialist)|personal care|health|patient|veterinar\w*|"
                   r"psychiatr\w*|psycholog\w*|bcba|behavior analyst|rbt|ophthalm\w*|opthalm\w*|speech|"
                   r"infirmi\w*|soignant\w*|pflege\w*|enfermer\w*|médecin|arzt|ärzt\w*|kranken\w*)\b"),
    ("Education", r"\b(teacher|tutor|instructor|educat\w*|school|professor|teaching|curriculum|faculty|trainer|"
                  r"language|enseignant\w*|lehrer\w*|formateur\w*|profesor\w*)\b"),
    ("Operations & Admin", r"\b(operations|ops|logistics|logisti\w*|supply|procurement|buyer|program manager|"
                           r"project manager\w*|program management|project management|projektleiter\w*|"
                           r"chef de projet|coordinat\w*|coördinator|administrat\w*|office|executive assistant|"
                           r"chief of staff|analyst|analista|planner|planning|quality|qualité|qualität\w*|"
                           r"production|produktion\w*|kaufmann|kauffrau|kaufleute|assistant\w*|assistente|"
                           r"delivery|lager\w*|entrepôt|magasinier\w*)\b"),
    ("Skilled Trades & Field", r"\b(technician|electrician|mechanic|inspector|plumber|hvac|welder|installer|"
                               r"maintenance|construction|carpenter|driver|operator|warehouse|forklift|"
                               r"technicien\w*|técnico|tecnico|mécanicien\w*|mecanicien\w*|kfz|meister|elektrik\w*|"
                               r"monteur\w*|chauffeur\w*|fahrer\w*|arbeiter\w*|ouvrier\w*|配送|ドライバー)\b"),
    ("Retail & Hospitality", r"\b(store|retail|cashier|sales associate|barista|cook|chef|server|crew|restaurant|"
                             r"hotel|housekeep\w*|host|hôte\w*|bartender|shift|vendeu\w*|verkäufer\w*|verkauf|"
                             r"vente|caisse|caissier\w*|rayon|merchandis\w*|salon|deli|kitchen|cuisin\w*|commis|"
                             r"bar|waiting|guest|attendant|greeter|valet|femme de chambre|polyvalent\w*|"
                             r"aushilfe|minijob\w*|employé\w*|employe|boulang\w*|bäcker\w*|filial\w*|markt)\b"),
    ("Consulting & Strategy", r"\b(consultant\w*|consulting|advisory|strateg\w*|berater\w*|conseil|"
                              r"transformation)\b"),
    ("Students & Apprentices", r"\b(intern|internship|student\w*|werkstudent\w*|praktik\w*|schülerpraktikum|"
                               r"ausbildung|azubi|apprenti\w*|apprentice\w*|alternance|alternant\w*|stage|stagiaire|"
                               r"trainee\w*|graduate|duales studium|abiturient\w*|becario|prácticas)\b"),
    # Generic leadership titles with no clearer function ("General Manager", "Director, Region").
    ("Management", r"\b(manager\w*|management|director\w*|head|leader\w*|lead|supervisor|teamleiter\w*|"
                   r"leiter\w*|responsable|directeur\w*|gerente|chef|executive|officer|president|partner)\b"),
]
OTHER_FAMILY = "Other"

_COMPILED_FAMILIES = [(name, re.compile(rx)) for name, rx in ROLE_FAMILIES]
_COMPILED_SENIORITY = [(name, re.compile(rx)) for name, rx in _SENIORITY_RULES]


def _normalize_title(title) -> str:
    """Lower-case a title and turn punctuation into spaces so word boundaries work.

    Letters of every alphabet are kept (\\w is Unicode-aware), so words such as
    "ingénieur" or "verkäufer" survive intact and can be matched.
    """
    text = "" if title is None or (isinstance(title, float) and np.isnan(title)) else str(title)
    return re.sub(r"[^\w&+ ]+|_", " ", text.lower())


def role_family(title) -> str:
    """Map a free-text job title to one of ~15 broad role families."""
    text = _normalize_title(title)
    for name, pattern in _COMPILED_FAMILIES:
        if pattern.search(text):
            return name
    return OTHER_FAMILY


def seniority(title) -> str:
    """Infer a seniority level from title keywords (not verified experience)."""
    text = _normalize_title(title)
    for name, pattern in _COMPILED_SENIORITY:
        if pattern.search(text):
            return name
    return "Mid / Unspecified"


def clean_annual_usd(salary_min, salary_max, currency, period):
    """Vectorized salary cleaning.

    Returns (midpoint, status) Series. The midpoint is NaN unless the posting
    has a plausible annual USD range; status explains why a salary was excluded.
    """
    lo = pd.to_numeric(salary_min, errors="coerce")
    hi = pd.to_numeric(salary_max, errors="coerce")
    mid = (lo + hi) / 2

    status = pd.Series("usable", index=lo.index, dtype="object")
    # Each rule only applies to rows that are still "usable", so the first
    # failing rule is the one recorded (this makes the funnel add up exactly).
    rules = [
        ("no_salary", lo.isna() | hi.isna()),
        ("not_annual_usd", (currency != "USD") | (period != "year")),
        ("invalid_range", (lo <= 0) | (hi < lo)),
        ("implausible_amount", ~mid.between(SALARY_FLOOR, SALARY_CEILING)),
        ("range_too_wide", hi / lo > MAX_RANGE_RATIO),
    ]
    for label, mask in rules:
        status[(status == "usable") & mask.fillna(False)] = label
    return mid.where(status == "usable"), status


def model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Build the exact input table the salary model expects.

    Used both at training time and when the dashboard makes a prediction, so
    the two can never drift apart. Missing values become the token "Unknown".
    """
    out = pd.DataFrame(index=df.index)
    out["title"] = df["title"].fillna("").astype(str)
    for col in ["company", "industry", "department", "country", "city", "employment_type"]:
        out[col] = df[col].astype("object").fillna("Unknown").astype(str)
    out["seniority"] = out["title"].map(seniority)
    out["role_family"] = out["title"].map(role_family)
    out["remote"] = df["remote"].fillna(False).astype(bool).map({True: "Remote", False: "On-site"})
    # Skills become one space-separated string; the model tokenizes on spaces.
    out["skills"] = df["skills"].map(lambda v: " ".join(v) if v is not None and len(v) else "")
    return out
