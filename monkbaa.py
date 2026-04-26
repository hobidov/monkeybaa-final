import os
from dotenv import load_dotenv
import math
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

try:
    import plotly.express as px
except Exception:
    px = None

PALETTE = {
    "orange": "#FF6A2A",
    "coral": "#F04E4E",
    "blue": "#4DB7E5",
    "cream": "#FFF7E8",
    "ink": "#2F2A24",
    "purple": "#7C3AED",
    "teal": "#14B8A6",
    "gold": "#D4A017",
}

STAGE_COLORS = {
    ("Social", "Spark"): PALETTE["orange"],
    ("Social", "Growth"): PALETTE["coral"],
    ("Social", "Horizon"): PALETTE["gold"],
    ("Cultural", "Spark"): PALETTE["blue"],
    ("Cultural", "Growth"): PALETTE["purple"],
    ("Cultural", "Horizon"): PALETTE["teal"],
}

BOT_INTRO = "Ask me about Social Spark, Social Horizon, Cultural Growth, Cultural Horizon, outcomes, or recommendations."
PROMPT_SUGGESTIONS = [
    "What is the strongest outcome area?",
    "Which outcome is weakest?",
    "How are Social and Cultural Horizon performing?",
    "What should Monkey Baa improve next?",
    "Summarize the core findings.",
]

MAPPING_RULES = [
    {"sourceColumn": "It gave me a sense of joy, beauty and wonder slider", "outcome": "Young people experience joy and wonder.", "stage": "Spark", "category": "Social", "normalizer": "zero_to_one"},
    {"sourceColumn": "It meant something to me personally slider", "outcome": "Young people experience a spark of inspiration.", "stage": "Spark", "category": "Social", "normalizer": "zero_to_one"},
    {"sourceColumn": "It is one of the best examples of its type that I have experienced slider", "outcome": "Young people experience a spark of inspiration.", "stage": "Spark", "category": "Social", "normalizer": "zero_to_one"},
    {"sourceColumn": "It inspired my own creativity slider", "outcome": "Young people build confidence and self-esteem.", "stage": "Growth", "category": "Social", "normalizer": "zero_to_one"},
    {"sourceColumn": "It opened my mind to new possibilities slider", "outcome": "Young people demonstrate enhanced empathy and emotional intelligence.", "stage": "Growth", "category": "Social", "normalizer": "zero_to_one"},
    {"sourceColumn": "It helped me feel part of the community slider", "outcome": "Young people experience greater social inclusion and community connection.", "stage": "Growth", "category": "Social", "normalizer": "zero_to_one"},
    {"sourceColumn": "How likely are you to attend an event/activity by Monkey Baa again? dropdown", "outcome": "Communities experience a lasting increase in social capital and youth engagement.", "stage": "Horizon", "category": "Social", "normalizer": "likelihood"},
    {"sourceColumn": "How likely is it that you would recommend this show to a friend or colleague? dropdown", "outcome": "Communities experience a lasting increase in social capital and youth engagement.", "stage": "Horizon", "category": "Social", "normalizer": "likelihood"},
    {"sourceColumn": "How would you rate your experience overall? dropdown", "outcome": "Young people benefit from improved well-being and create lifelong positive memories.", "stage": "Horizon", "category": "Social", "normalizer": "overall_experience"},
    {"sourceColumn": "The performance was entertaining slider", "outcome": "Young people develop curiosity and engagement with theatre.", "stage": "Spark", "category": "Cultural", "normalizer": "zero_to_one"},
    {"sourceColumn": "The performance was emotionally impactful slider", "outcome": "Young people see themselves in stories and feel validated.", "stage": "Spark", "category": "Cultural", "normalizer": "zero_to_one"},
    {"sourceColumn": "It opened my mind to new possibilities slider", "outcome": "Young people build increased cultural literacy and openness.", "stage": "Growth", "category": "Cultural", "normalizer": "zero_to_one"},
    {"sourceColumn": "How likely is it that you would recommend this show to a friend or colleague? dropdown", "outcome": "Young people develop a growing appreciation for theatre and the arts.", "stage": "Growth", "category": "Cultural", "normalizer": "likelihood"},
    {"sourceColumn": "How likely are you to attend an event/activity by Monkey Baa again? dropdown", "outcome": "Young people and communities become repeat attendees and new audiences are formed.", "stage": "Growth", "category": "Cultural", "normalizer": "likelihood"},
    {"sourceColumn": "How likely is it that you would recommend this show to a friend or colleague? dropdown", "outcome": "Monkey Baa influences the broader arts sector.", "stage": "Horizon", "category": "Cultural", "normalizer": "likelihood"},
    {"sourceColumn": "How likely are you to attend an event/activity by Monkey Baa again? dropdown", "outcome": "A generation of lifelong arts engagers is cultivated.", "stage": "Horizon", "category": "Cultural", "normalizer": "likelihood"},
    {"sourceColumn": "How would you rate your experience overall? dropdown", "outcome": "Australian storytelling is enriched and diversified.", "stage": "Horizon", "category": "Cultural", "normalizer": "overall_experience"},
]

DEFAULT_KPIS = [
    {"label": "Social Spark", "value": 0, "category": "Social", "stage": "Spark", "color": PALETTE["orange"]},
    {"label": "Social Growth", "value": 0, "category": "Social", "stage": "Growth", "color": PALETTE["coral"]},
    {"label": "Social Horizon", "value": 0, "category": "Social", "stage": "Horizon", "color": PALETTE["gold"]},
    {"label": "Cultural Spark", "value": 0, "category": "Cultural", "stage": "Spark", "color": PALETTE["blue"]},
    {"label": "Cultural Growth", "value": 0, "category": "Cultural", "stage": "Growth", "color": PALETTE["purple"]},
    {"label": "Cultural Horizon", "value": 0, "category": "Cultural", "stage": "Horizon", "color": PALETTE["teal"]},
]

def safe(value: Any) -> str:
    return "" if value is None or pd.isna(value) else str(value).strip()

def lower(value: Any) -> str:
    return safe(value).lower()

def avg(values: List[float]) -> int:
    return int(round(sum(values) / len(values))) if values else 0

def clamp(n: float, low: int = 0, high: int = 100) -> int:
    return max(low, min(high, int(round(n))))

def normalize_zero_to_one(value: Any) -> Optional[int]:
    if isinstance(value, (int, float)) and not pd.isna(value):
        if 0 <= value <= 1:
            return round(value * 100)
        if 1 < value <= 100:
            return round(value)
        return None
    numeric = pd.to_numeric(safe(value), errors="coerce")
    if pd.isna(numeric):
        return None
    return normalize_zero_to_one(float(numeric))

def normalize_likelihood(value: Any) -> Optional[int]:
    text = lower(value)
    if not text:
        return None
    direct = pd.to_numeric(text, errors="coerce")
    if not pd.isna(direct):
        return clamp((float(direct) / 10) * 100)
    if "extremely likely" in text:
        return 100
    if "very likely" in text:
        return 90
    if text == "likely":
        return 75
    if "neutral" in text:
        return 50
    if "unlikely" in text:
        return 25
    match = re.search(r"\b(10|[1-9])\b", text)
    return clamp((int(match.group(1)) / 10) * 100) if match else None

def normalize_overall_experience(value: Any) -> Optional[int]:
    text = lower(value)
    if not text:
        return None
    if "excellent" in text:
        return 100
    if "good" in text:
        return 75
    if "neutral" in text:
        return 50
    if "poor" in text:
        return 25
    return None

def decode_text(value: str) -> str:
    return (
        value.replace("?üòä", "Happy")
        .replace("�", "Happy")
        .replace("?üßê", "Curious")
        .replace("?üòÆ", "Surprised")
        .replace("?üò®", "Scared")
        .replace("?üòê", "Bored")
        .replace("?üòï", "Confused")
        .replace("‚Äô", "'")
        .strip()
    )

def split_multi_select(value: str) -> List[str]:
    return [
        part.replace("'", "").strip()
        for part in decode_text(value).replace("[", "").replace("]", "").replace("', '", ",").split(",")
        if part.replace("'", "").strip()
    ]

def infer_audience(row: Dict[str, Any]) -> str:
    return safe(row.get("What title best describes you? dropdown")) or safe(row.get("Which category does the respondent belong to? shorttext")) or "Respondent"

def infer_show(row: Dict[str, Any]) -> str:
    return safe(row.get("What Monkey Baa show did you recently attend? dropdown")) or "Unknown show"

def infer_location(row: Dict[str, Any]) -> str:
    return safe(row.get("Where did you see the show? dropdown")) or safe(row.get("Location")) or "Unknown"

def get_required_columns() -> List[str]:
    return sorted(set(rule["sourceColumn"] for rule in MAPPING_RULES))

def run_normalizer(kind: str, value: Any) -> Optional[int]:
    if kind == "zero_to_one":
        return normalize_zero_to_one(value)
    if kind == "likelihood":
        return normalize_likelihood(value)
    if kind == "overall_experience":
        return normalize_overall_experience(value)
    return None

def map_survey_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    mapped: List[Dict[str, Any]] = []
    for row in rows:
        audience = infer_audience(row)
        show = infer_show(row)
        location = infer_location(row)
        for rule in MAPPING_RULES:
            score = run_normalizer(rule["normalizer"], row.get(rule["sourceColumn"]))
            if score is None:
                continue
            mapped.append(
                {
                    "audience": audience,
                    "show": show,
                    "location": location,
                    "question": rule["sourceColumn"],
                    "score": score,
                    "outcome": rule["outcome"],
                    "stage": rule["stage"],
                    "category": rule["category"],
                    "sourceColumn": rule["sourceColumn"],
                }
            )
    return mapped

def build_kpi(label: str, rows: List[Dict[str, Any]], category: str, stage: str, color: str) -> Dict[str, Any]:
    return {"label": label, "value": avg([row["score"] for row in rows]), "category": category, "stage": stage, "color": color}

def find_kpi(kpis: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
    for kpi in kpis:
        if kpi["label"] == label:
            return kpi
    for kpi in DEFAULT_KPIS:
        if kpi["label"] == label:
            return kpi
    return DEFAULT_KPIS[0]

def count_by_label(rows: List[Dict[str, Any]], getter) -> List[Dict[str, Any]]:
    counts: Dict[str, int] = {}
    for row in rows:
        key = getter(row) or "Unknown"
        counts[key] = counts.get(key, 0) + 1
    return [{"label": label, "count": count} for label, count in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)]

def count_multi_select(rows: List[Dict[str, Any]], column: str) -> List[Dict[str, Any]]:
    counts: Dict[str, int] = {}
    for row in rows:
        raw = safe(row.get(column))
        if not raw:
            continue
        for item in split_multi_select(raw):
            counts[item] = counts.get(item, 0) + 1
    return [{"label": label, "count": count} for label, count in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)]

def outcome_stats(mapped_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, str], List[int]] = {}
    for row in mapped_rows:
        key = (row["category"], row["stage"], row["outcome"])
        groups.setdefault(key, []).append(int(row["score"]))
    items = []
    for (category, stage, outcome), values in groups.items():
        items.append({"category": category, "stage": stage, "outcome": outcome, "value": avg(values), "label": f"{outcome[:52]}{'…' if len(outcome) > 52 else ''}"})
    return sorted(items, key=lambda x: x["value"], reverse=True)

def compute_analytics(filtered_rows: List[Dict[str, Any]], source_row_count: int) -> Dict[str, Any]:
    def by_category_stage(category: str, stage: str) -> List[Dict[str, Any]]:
        return [row for row in filtered_rows if row["category"] == category and row["stage"] == stage]
    kpis = [
        build_kpi("Social Spark", by_category_stage("Social", "Spark"), "Social", "Spark", PALETTE["orange"]),
        build_kpi("Social Growth", by_category_stage("Social", "Growth"), "Social", "Growth", PALETTE["coral"]),
        build_kpi("Social Horizon", by_category_stage("Social", "Horizon"), "Social", "Horizon", PALETTE["gold"]),
        build_kpi("Cultural Spark", by_category_stage("Cultural", "Spark"), "Cultural", "Spark", PALETTE["blue"]),
        build_kpi("Cultural Growth", by_category_stage("Cultural", "Growth"), "Cultural", "Growth", PALETTE["purple"]),
        build_kpi("Cultural Horizon", by_category_stage("Cultural", "Horizon"), "Cultural", "Horizon", PALETTE["teal"]),
    ]
    outcomes = outcome_stats(filtered_rows)
    overall = avg([k["value"] for k in kpis])
    strongest = sorted(kpis, key=lambda x: x["value"], reverse=True)[0] if kpis else DEFAULT_KPIS[0]
    weakest = sorted(kpis, key=lambda x: x["value"])[0] if kpis else DEFAULT_KPIS[0]
    completion = round((len(filtered_rows) / (source_row_count * len(MAPPING_RULES))) * 100) if source_row_count else 0
    return {"rows": filtered_rows, "overall": overall, "strongest": strongest, "weakest": weakest, "kpis": kpis, "outcomeStats": outcomes, "dataQuality": {"mappedRows": len(filtered_rows), "sourceRows": source_row_count, "completion": completion}}

def build_bot_reply(prompt: str, analytics: Dict[str, Any]) -> str:

    try:
        context = {
            "overall_score": analytics["overall"],
            "strongest_area": analytics["strongest"]["label"],
            "strongest_value": analytics["strongest"]["value"],
            "weakest_area": analytics["weakest"]["label"],
            "weakest_value": analytics["weakest"]["value"],
        }

        system_prompt = f"""
You are an expert data analyst for Monkey Baa Theatre.

Rules:
- Use ONLY provided data
- Be short and clear
- Always give a recommendation

DATA:
{context}
"""

        import requests
        import os

        api_key = os.getenv("TOGETHER_API_KEY")

        response = requests.post(
            "https://api.together.xyz/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "meta-llama/Meta-Llama-3-8B-Instruct-Lite",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 300,
                "temperature": 0.7
            }
        )
        result = response.json()

# DEBUG SAFETY (VERY IMPORTANT)
        if "choices" not in result:
           return f"API Error: {result}"
        return response.json()["choices"][0]["message"]["content"]

    except Exception as e:
        return f"Error: {str(e)}"
    
def generate_ai_summary_together(analytics: Dict[str, Any]) -> str:
    import requests
    import os

    api_key = os.getenv("TOGETHER_API_KEY")

    context = {
        "overall": analytics["overall"],
        "strongest": analytics["strongest"]["label"],
        "strongest_value": analytics["strongest"]["value"],
        "weakest": analytics["weakest"]["label"],
        "weakest_value": analytics["weakest"]["value"],
        "kpis": {k["label"]: k["value"] for k in analytics["kpis"]}
    }

    prompt = f"""
You are an expert arts impact analyst.

Write:
- 1 short executive summary paragraph
- 3 bullet insights
- 1 recommendation

Use ONLY this data:
{context}
"""

    response = requests.post(
        "https://api.together.xyz/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": "meta-llama/Meta-Llama-3-8B-Instruct-Lite",
            "messages": [
                {"role": "system", "content": "You are a professional analyst."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 300
        }
    )

    result = response.json()

    if "choices" not in result:
        return f"AI Summary Error: {result}"

    return result["choices"][0]["message"]["content"]

def create_report_text(analytics: Dict[str, Any], summary_name: Optional[str] = None) -> str:
    ai_summary = generate_ai_summary_together(analytics)
    strongest = analytics["strongest"]
    weakest = analytics["weakest"]
    social_spark = find_kpi(analytics["kpis"], "Social Spark")
    social_growth = find_kpi(analytics["kpis"], "Social Growth")
    social_horizon = find_kpi(analytics["kpis"], "Social Horizon")
    cultural_spark = find_kpi(analytics["kpis"], "Cultural Spark")
    cultural_growth = find_kpi(analytics["kpis"], "Cultural Growth")
    cultural_horizon = find_kpi(analytics["kpis"], "Cultural Horizon")
    return f"""
EXECUTIVE SUMMARY (AI GENERATED)

{ai_summary}

Survey responses from {summary_name or 'the uploaded survey'} indicate stronger immediate outcomes in {lower(strongest['label'])} and weaker performance in {lower(weakest['label'])}. Overall impact is {analytics['overall']}%, showing that both social and cultural impact streams now include spark, growth, and horizon proxy measures."""

def load_uploaded_file(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file), ["Sheet1"]
    sheets = pd.read_excel(uploaded_file, sheet_name=None)
    if not sheets:
        raise ValueError("No sheets found")
    first_name = list(sheets.keys())[0]
    return sheets[first_name], list(sheets.keys())

def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized.columns = [safe(col) for col in normalized.columns]
    for col in normalized.columns:
        if normalized[col].dtype == "object":
            normalized[col] = normalized[col].map(lambda v: v.strip() if isinstance(v, str) else v)
    normalized = normalized.dropna(how="all")
    normalized = normalized.loc[normalized.apply(lambda row: any(safe(v) != "" for v in row.values.tolist()), axis=1)]
    return normalized

def kpi_card(label: str, value: int, category: str, stage: str, color: str):
    st.markdown(f"""
        <div class="kpi-card" style="background:white;border:1px solid #f1f1f1;border-radius:24px;padding:20px;box-shadow:0 2px 10px rgba(0,0,0,0.06);min-height:190px;">
          <div style="font-size:13px;color:#6b7280;margin-bottom:6px;">{category} · {stage}</div>
          <div style="font-size:17px;font-weight:700;line-height:1.4;margin-bottom:8px;">{label}</div>
          <div style="font-size:40px;font-weight:900;color:{color};margin-top:6px;">{value}%</div>
        </div>
    """, unsafe_allow_html=True)

def bar_rows(items: List[Dict[str, Any]], value_key: str, color: str, pct_suffix: str = ""):
    if not items:
        st.info("No data available.")
        return
    max_value = max(float(item[value_key]) for item in items)
    max_value = max(max_value, 1.0)
    for item in items:
        value = float(item[value_key])
        pct = (value / max_value) * 100
        st.markdown(f"""
            <div style="margin-bottom:12px;">
              <div style="display:flex;justify-content:space-between;gap:12px;font-size:14px;margin-bottom:6px;">
                <div style="flex:1;min-width:0;">{item['label']}</div>
                <div style="font-weight:700;">{int(round(value))}{pct_suffix}</div>
              </div>
              <div style="height:14px;background:#f1f5f9;border-radius:999px;overflow:hidden;">
                <div style="width:{pct:.1f}%;height:14px;background:{color};border-radius:999px;"></div>
              </div>
            </div>
        """, unsafe_allow_html=True)

st.set_page_config(page_title="Monkey Baa Impact Dashboard", layout="wide")
st.markdown("""
<style>
.stApp { background: #FFF7E8; }

/* GLOBAL PADDING */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

/* BANNERS */
.banner-core { 
    background:#EEF6F1;
    border:1px solid #d8efe2;
    border-radius:20px;
    padding:16px 18px;
    margin-bottom:20px;
    line-height:1.5;
}

.banner-support { 
    background:#EEF6FB;
    border:1px solid #d7ebf7;
    border-radius:20px;
    padding:16px 18px;
    margin-bottom:20px;
    line-height:1.5;
}

/* CARDS */
.side-card { 
    background:white;
    border:1px solid #f1f1f1;
    border-radius:22px;
    padding:18px;
    box-shadow:0 2px 10px rgba(0,0,0,0.06);
    margin-bottom:18px;
}

/* KPI SPACING */
.kpi-card {
    margin-bottom:16px;
}

/* COLUMN GAP FIX */
div[data-testid="column"] {
    padding-left:10px;
    padding-right:10px;
}

/* CHAT */
.chat-bubble {
    margin-bottom:10px;
}
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": BOT_INTRO}]

st.title("Monkey Baa Impact Analyzer")
uploaded_file = st.file_uploader("Upload a CSV or Excel survey export", type=["csv", "xlsx", "xls"])
if uploaded_file is None:
    st.info("Upload a CSV or Excel survey export and analyze it using the Monkey Baa outcome framework.")
    st.stop()

with st.spinner("Processing your file..."):
    df, sheets = load_uploaded_file(uploaded_file)
    df = normalize_dataframe(df)
    if df.empty:
        st.error("No rows found in the uploaded file.")
        st.stop()
    missing_columns = [col for col in get_required_columns() if col not in df.columns]
    if missing_columns:
        st.error("Missing required columns: " + ", ".join(missing_columns))
        st.stop()
    source_rows = df.to_dict(orient="records")
    all_mapped_rows = map_survey_rows(source_rows)

if not all_mapped_rows:
    st.error("The file loaded, but no rows could be mapped into the framework.")
    st.stop()

shows = ["All shows"] + sorted({row["show"] for row in all_mapped_rows})
default_show = shows[0]
base_rows = all_mapped_rows

c1, c2, c3 = st.columns(3)
with c1:
    selected_show = st.selectbox("Show", shows, index=0)
base_rows = all_mapped_rows if selected_show == "All shows" else [row for row in all_mapped_rows if row["show"] == selected_show]
audiences = ["All"] + sorted({row["audience"] for row in base_rows})
locations = ["All"] + sorted({row["location"] for row in base_rows})
with c2:
    filter_audience = st.selectbox("Audience", audiences)
with c3:
    filter_location = st.selectbox("Location", locations)

filtered_rows = [row for row in base_rows if (filter_audience == "All" or row["audience"] == filter_audience) and (filter_location == "All" or row["location"] == filter_location)]
filtered_source_rows = [row for row in source_rows if (selected_show == "All shows" or infer_show(row) == selected_show) and (filter_audience == "All" or infer_audience(row) == filter_audience) and (filter_location == "All" or infer_location(row) == filter_location)]

analytics = compute_analytics(filtered_rows, len(source_rows))
report_text = create_report_text(analytics, uploaded_file.name if selected_show == "All shows" else f"{uploaded_file.name} - {selected_show}")

top_left, top_mid, top_right = st.columns([1, 2, 1])
with top_left:
    st.download_button("Download report", data=report_text.encode("utf-8"), file_name="monkey_baa-impact-report.txt", mime="text/plain", use_container_width=True)
with top_mid:
    st.markdown(f"""
    <div style="text-align:center;">
      <div style="font-size:14px;font-weight:600;color:#6b7280;">Monkey Baa Outcome Dashboard</div>
      <div style="font-size:32px;font-weight:900;">Framework-Aligned Impact Analysis</div>
      <div style="margin-top:6px;font-size:14px;color:#6b7280;">{uploaded_file.name} · {len(source_rows)} rows · {len(df.columns)} columns · {len(sheets)} sheet(s){'' if selected_show == 'All shows' else ' · Filter: ' + selected_show}</div>
    </div>
    """, unsafe_allow_html=True)
with top_right:
    st.metric("Overall", f"{analytics['overall']}%")

main_col, side_col = st.columns([3.2, 1.25], gap="large")

with main_col:
    st.markdown('<div class="banner-core"><div style="font-size:13px;font-weight:700;text-transform:uppercase;">Core Impact Charts</div><div style="margin-top:4px;font-size:14px;color:#4b5563;">These are framework-critical metrics directly mapped to Monkey Baa’s Theory of Change (Spark → Growth → Horizon). Use these for decision-making and reporting.</div></div>', unsafe_allow_html=True)

    kpi_cols = st.columns(3)
    for idx, kpi in enumerate(analytics["kpis"]):
        with kpi_cols[idx % 3]:
            kpi_card(kpi["label"], kpi["value"], kpi["category"], kpi["stage"], kpi["color"])

    stage_pairs = [("Social", "Spark"), ("Social", "Growth"), ("Social", "Horizon"), ("Cultural", "Spark"), ("Cultural", "Growth"), ("Cultural", "Horizon")]
    for i in range(0, len(stage_pairs), 2):
        cols = st.columns(2)
        for col, (category, stage) in zip(cols, stage_pairs[i:i+2]):
            subset = [item for item in analytics["outcomeStats"] if item["category"] == category and item["stage"] == stage]
            with col:
                st.markdown('<div class="side-card">', unsafe_allow_html=True)
                st.subheader(f"Outcome Breakdown — {category} {stage}")
                if subset:
                    bar_rows(subset, "value", STAGE_COLORS[(category, stage)], "%")
                else:
                    st.info("No data available.")
                st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="banner-support" style="margin-top:16px;"><div style="font-size:13px;font-weight:700;text-transform:uppercase;">Supporting Impact Charts</div><div style="margin-top:4px;font-size:14px;color:#4b5563;">Contextual metrics that support interpretation of core outcomes, including response distribution and engagement patterns.</div></div>', unsafe_allow_html=True)

    show_counts = count_by_label(filtered_source_rows, infer_show)
    behaviour_counts = count_multi_select(filtered_source_rows, "After the show, did the young person... multiplechoice")
    emotion_counts = count_multi_select(filtered_source_rows, "What feeling/s did the young person experience during the performance? multiplechoice")
    audience_counts = count_by_label(filtered_source_rows, infer_audience)
    location_counts = count_by_label(filtered_source_rows, infer_location)

    s1, s2 = st.columns(2)
    with s1:
        st.markdown('<div class="side-card">', unsafe_allow_html=True)
        st.subheader("Show Comparison")
        bar_rows(show_counts, "count", PALETTE["blue"])
        st.markdown("</div>", unsafe_allow_html=True)
    with s2:
        st.markdown('<div class="side-card">', unsafe_allow_html=True)
        st.subheader("Behaviour Distribution")
        bar_rows(behaviour_counts, "count", PALETTE["coral"])
        st.markdown("</div>", unsafe_allow_html=True)

    s3, s4 = st.columns(2)
    with s3:
        st.markdown('<div class="side-card">', unsafe_allow_html=True)
        st.subheader("Emotion Distribution")
        if px and emotion_counts:
            fig = px.pie(pd.DataFrame(emotion_counts), values="count", names="label", hole=0)
            fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            bar_rows(emotion_counts, "count", PALETTE["gold"])
        st.markdown("</div>", unsafe_allow_html=True)
    with s4:
        st.markdown('<div class="side-card">', unsafe_allow_html=True)
        st.subheader("Audience Segmentation")
        bar_rows(audience_counts, "count", PALETTE["purple"])
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="side-card" style="margin-top:16px;">', unsafe_allow_html=True)
    st.subheader("Location Distribution")
    bar_rows(location_counts, "count", PALETTE["teal"])
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="side-card" style="margin-top:16px;">', unsafe_allow_html=True)
    st.subheader("Mapped Survey Data")
    tf1, tf2, tf3 = st.columns(3)
    table_audience_options = ["All"] + sorted({row["audience"] for row in filtered_rows})
    with tf1:
        table_audience = st.selectbox("Audience filter", table_audience_options)
    with tf2:
        table_category = st.selectbox("Category filter", ["All", "Social", "Cultural"])
    with tf3:
        table_stage = st.selectbox("Stage filter", ["All", "Spark", "Growth", "Horizon"])

    filtered_table_rows = [row for row in filtered_rows if (table_audience == "All" or row["audience"] == table_audience) and (table_category == "All" or row["category"] == table_category) and (table_stage == "All" or row["stage"] == table_stage)]
    page_size = 10
    page_count = max(1, math.ceil(len(filtered_table_rows) / page_size))
    page = st.number_input("Page", min_value=1, max_value=page_count, value=1, step=1)
    start = (page - 1) * page_size
    table_df = pd.DataFrame(filtered_table_rows[start:start + page_size])
    if not table_df.empty:
        st.dataframe(table_df[["audience", "show", "location", "outcome", "category", "stage", "score"]], use_container_width=True, hide_index=True)
    else:
        st.info("No mapped data available.")
    st.caption(f"Page {page} of {page_count}")
    st.markdown("</div>", unsafe_allow_html=True)

with side_col:
    st.markdown('<div class="side-card">', unsafe_allow_html=True)
    st.subheader("Gen AI Insight Bot")
    for message in st.session_state.messages:
        if message["role"] == "assistant":
            st.markdown(f'<div style="white-space:pre-line;border:1px solid #f6d3c7;background:white;border-radius:18px;padding:12px;margin-bottom:10px;">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="white-space:pre-line;background:#FF6A2A;color:white;border-radius:18px;padding:12px;margin-bottom:10px;margin-left:24px;">{message["content"]}</div>', unsafe_allow_html=True)
    bot_input = st.text_area("Ask the bot for insights", height=120)
    if st.button("Get insight", use_container_width=True) and bot_input.strip():
        st.session_state.messages.append({"role": "user", "content": bot_input.strip()})
        st.session_state.messages.append({"role": "assistant", "content": build_bot_reply(bot_input.strip(), analytics)})
        st.rerun()

    st.caption("Suggested questions:")
    for question in PROMPT_SUGGESTIONS:
        if st.button(question, key=question, use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": question})
            st.session_state.messages.append({"role": "assistant", "content": build_bot_reply(question, analytics)})
            st.rerun()

    st.markdown('<div class="side-card" style="margin-top:14px;">', unsafe_allow_html=True)
    st.markdown("**Framework Quality Summary**")
    st.write(f"**Source rows:** {analytics['dataQuality']['sourceRows']}")
    st.write(f"**Mapped records:** {analytics['dataQuality']['mappedRows']}")
    st.write(f"**Approx. mapping completion:** {analytics['dataQuality']['completion']}%")
    st.write(f"**Show filter:** {selected_show}")
    st.write(f"**Location filter:** {filter_location}")
    st.write(f"**Strongest area:** {analytics['strongest']['label']}")
    st.write(f"**Weakest area:** {analytics['weakest']['label']}")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True) 
