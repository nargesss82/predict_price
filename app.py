# -*- coding: utf-8 -*-
"""
app.py
======
داشبورد Streamlit برای تحلیل و پیش‌بینی قیمت طلا.
مصنوعات آموزش‌دیده (مدل‌ها، اسکیلر، متریک‌ها) را از پوشهٔ محلی
`gold_price_project/` می‌خواند.

اجرا:
    streamlit run app.py
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import joblib

# =============================================================================
# تنظیمات صفحه
# =============================================================================
st.set_page_config(
    page_title="پیش‌بینی و تحلیل احساسات قیمت طلا",
    page_icon="🥇",
    layout="wide",
)

CUSTOM_CSS = """
<style>
    .main .block-container {padding-top: 2rem; max-width: 1200px;}
    .metric-card {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        border-radius: 14px;
        padding: 20px 24px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 14px rgba(0,0,0,0.15);
    }
    .metric-card .value {font-size: 2.1rem; font-weight: 700; color: #FFD700;}
    .metric-card .label {font-size: 0.95rem; color: #d1d5db; margin-bottom: 6px;}
    div[data-testid="stMetricValue"] {font-size: 1.6rem;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =============================================================================
# ثابت‌ها و مسیر مصنوعات نوت‌بوک
# =============================================================================
RAW_NUMERIC_COLS = ["Close/Last", "Volume", "Open", "High", "Low"]
TARGET_COL = "Close/Last"

_BASE = Path(__file__).resolve().parent / "gold_price_project"
PATHS = {
    "base": str(_BASE),
    "models_dir": str(_BASE / "models"),
    "data_dir": str(_BASE / "data"),
    "optuna_dir": str(_BASE / "optuna"),
    "raw_csv": str(_BASE / "data" / "gold_prices.csv"),
    "features_csv": str(_BASE / "data" / "features.csv"),
    "scaler_path": str(_BASE / "models" / "minmax_scaler.pkl"),
    "lstm_model_path": str(_BASE / "models" / "lstm_model.keras"),
    "gru_model_path": str(_BASE / "models" / "gru_model.keras"),
    "xgb_model_path": str(_BASE / "models" / "xgb_model.json"),
    "lstm_history_path": str(_BASE / "models" / "lstm_history.json"),
    "gru_history_path": str(_BASE / "models" / "gru_history.json"),
    "metrics_path": str(_BASE / "models" / "metrics.json"),
    "test_predictions_path": str(_BASE / "models" / "test_predictions.json"),
    "feature_columns_path": str(_BASE / "models" / "feature_columns.json"),
    "train_meta_path": str(_BASE / "models" / "train_meta.json"),
}


def load_raw_csv(path):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="coerce")
    if df["Date"].isna().any():
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df.sort_values("Date").reset_index(drop=True)


def load_features_csv(path):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df.sort_values("Date").reset_index(drop=True)


def missing_value_report(df, cols=None):
    cols = cols or df.columns.tolist()
    n = len(df)
    rows = []
    for c in cols:
        na = int(df[c].isna().sum())
        rows.append({"ستون": c, "تعداد گمشده": na, "درصد گمشده": round(100 * na / n, 3) if n else 0.0})
    return pd.DataFrame(rows)


def compute_simulated_sentiment(df, lookback=30):
    d = df.tail(lookback).copy()
    if len(d) < 5:
        return {"score": 0.0, "label": "خنثی", "label_en": "Neutral"}

    last_return = d["daily_return"].iloc[-1] if "daily_return" in d.columns else 0.0
    last_return_norm = np.clip(last_return / 5.0, -3, 3)
    vol_mean, vol_std = d["Volume"].mean(), d["Volume"].std()
    volume_z = (d["Volume"].iloc[-1] - vol_mean) / vol_std if vol_std and not np.isnan(vol_std) and vol_std != 0 else 0.0
    volume_z = np.clip(volume_z, -3, 3)
    ma = d[TARGET_COL].mean()
    last_price = d[TARGET_COL].iloc[-1]
    price_position = (last_price - ma) / ma if ma != 0 else 0.0
    price_position_norm = np.clip(price_position * 20, -3, 3)
    score = float(np.tanh(last_return_norm * 0.4 + volume_z * 0.3 + price_position_norm * 0.3))

    if score > 0.15:
        label, label_en = "مثبت (صعودی)", "Positive (Bullish)"
    elif score < -0.15:
        label, label_en = "منفی (نزولی)", "Negative (Bearish)"
    else:
        label, label_en = "خنثی", "Neutral"

    return {
        "score": round(score, 4),
        "label": label,
        "label_en": label_en,
        "components": {
            "last_return_norm": round(float(last_return_norm), 4),
            "volume_z_score": round(float(volume_z), 4),
            "price_position_vs_MA": round(float(price_position_norm), 4),
        },
    }


def combine_prediction(model_prediction, sentiment_score, average_price_change):
    return model_prediction + (0.25 * sentiment_score * average_price_change)


def get_average_price_swing(df, lookback=30, target_col=TARGET_COL):
    recent = df.tail(lookback)
    return recent["daily_return"].abs().mean() / 100.0 * recent[target_col].iloc[-1]


# =============================================================================
# بررسی وجود مدل‌ها
# =============================================================================
def artifacts_exist():
    required = [
        PATHS["lstm_model_path"], PATHS["gru_model_path"], PATHS["xgb_model_path"],
        PATHS["scaler_path"], PATHS["metrics_path"], PATHS["train_meta_path"],
        PATHS["feature_columns_path"], PATHS["test_predictions_path"],
        PATHS["features_csv"], PATHS["raw_csv"],
    ]
    return all(Path(p).exists() for p in required)


# =============================================================================
# توابع کش‌شده برای بارگذاری داده و مدل‌ها
# =============================================================================
@st.cache_data(show_spinner=False)
def load_notebook_data(raw_csv_path, features_csv_path):
    """دادهٔ خام و features.csv آماده‌شده توسط نوت‌بوک را می‌خواند."""
    df_raw = load_raw_csv(raw_csv_path)
    features_df = load_features_csv(features_csv_path)
    features_df["MA_7"] = features_df["rolling_mean_7"]
    features_df["MA_30"] = features_df[TARGET_COL].rolling(window=30).mean()
    return df_raw, features_df


@st.cache_resource(show_spinner=False)
def load_models_and_meta():
    import xgboost as xgb
    from tensorflow.keras.models import load_model

    lstm_model = load_model(PATHS["lstm_model_path"])
    gru_model = load_model(PATHS["gru_model_path"])
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(PATHS["xgb_model_path"])
    scaler = joblib.load(PATHS["scaler_path"])

    with open(PATHS["metrics_path"], "r", encoding="utf-8") as f:
        metrics = json.load(f)
    with open(PATHS["train_meta_path"], "r", encoding="utf-8") as f:
        train_meta = json.load(f)
    with open(PATHS["feature_columns_path"], "r", encoding="utf-8") as f:
        feature_cols = json.load(f)
    with open(PATHS["test_predictions_path"], "r", encoding="utf-8") as f:
        test_predictions = json.load(f)

    lstm_history, gru_history = None, None
    if Path(PATHS["lstm_history_path"]).exists():
        with open(PATHS["lstm_history_path"], "r") as f:
            lstm_history = json.load(f)
    if Path(PATHS["gru_history_path"]).exists():
        with open(PATHS["gru_history_path"], "r") as f:
            gru_history = json.load(f)

    return {
        "lstm_model": lstm_model,
        "gru_model": gru_model,
        "xgb_model": xgb_model,
        "scaler": scaler,
        "metrics": metrics,
        "train_meta": train_meta,
        "feature_cols": feature_cols,
        "test_predictions": test_predictions,
        "lstm_history": lstm_history,
        "gru_history": gru_history,
    }


def inverse_scale_target(scaler, scaled_values, nn_feature_cols, target_idx):
    scaled_values = np.asarray(scaled_values).ravel()
    dummy = np.zeros((len(scaled_values), len(nn_feature_cols)))
    dummy[:, target_idx] = scaled_values
    return scaler.inverse_transform(dummy)[:, target_idx]


def scale_target_value(scaler, raw_value, nn_feature_cols, target_idx):
    dummy = np.zeros((1, len(nn_feature_cols)))
    dummy[:, target_idx] = raw_value
    return scaler.transform(dummy)[0, target_idx]


# =============================================================================
# جمع‌آوری خودکار اخبار اقتصادی/طلا از وب (بدون کلید API)
# =============================================================================
NEWS_KEYWORDS = [
    "gold", "bullion", "federal reserve", "fed ", "interest rate",
    "inflation", "fomc", "central bank", "dollar", "treasury yield",
]

# منابع خبری برای اسکرپینگ:
#  - Investing.com: فید RSS رسمی و پایدار برای اخبار کالاها (commodities)
#  - Kitco News: صفحه‌ی HTML اصلی اخبار (بدون RSS عمومی پایدار)، مستقیماً پارس می‌شود
INVESTING_RSS_URL = "https://www.investing.com/rss/news_11.rss"
KITCO_NEWS_URL = "https://www.kitco.com/news"


def _headline_matches_keywords(text):
    text_lower = text.lower()
    return any(kw in text_lower for kw in NEWS_KEYWORDS)


def fetch_investing_com_news(max_items=10):
    import feedparser

    try:
        feed = feedparser.parse(INVESTING_RSS_URL)
        items = []
        for entry in feed.entries[:40]:
            title = entry.get("title", "").strip()
            if not title:
                continue
            if _headline_matches_keywords(title):
                items.append({
                    "title": title,
                    "source": "Investing.com",
                    "link": entry.get("link", ""),
                    "published": entry.get("published", ""),
                })
            if len(items) >= max_items:
                break
        return items
    except Exception:
        return []


def fetch_kitco_news(max_items=10):
    import requests
    from bs4 import BeautifulSoup

    try:
        resp = requests.get(
            KITCO_NEWS_URL,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
            timeout=10,
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        items = []
        seen_titles = set()

        def _add_if_match(text, href):
            text = text.strip()
            if not text or len(text) < 20 or len(text) > 220:
                return
            if text in seen_titles:
                return
            if _headline_matches_keywords(text):
                seen_titles.add(text)
                full_href = href
                if full_href and not full_href.startswith("http"):
                    full_href = "https://www.kitco.com" + full_href
                items.append({
                    "title": text,
                    "source": "Kitco News",
                    "link": full_href or "",
                    "published": "",
                })

        # استراتژی ۱: تگ‌های عنوان استاندارد
        for heading in soup.find_all(["h1", "h2", "h3"]):
            link_tag = heading.find("a") or heading.find_parent("a")
            href = link_tag.get("href", "") if link_tag else ""
            _add_if_match(heading.get_text(strip=True), href)
            if len(items) >= max_items:
                break

        # استراتژی ۲ (fallback): اگر استراتژی ۱ چیزی پیدا نکرد، همه‌ی لینک‌ها را بگرد
        if not items:
            for a_tag in soup.find_all("a"):
                _add_if_match(a_tag.get_text(strip=True), a_tag.get("href", ""))
                if len(items) >= max_items:
                    break

        return items[:max_items]
    except Exception:
        return []


def fetch_latest_news(max_total=5):
    investing_items = fetch_investing_com_news(max_items=max_total * 2)
    kitco_items = fetch_kitco_news(max_items=max_total * 2)

    combined = investing_items + kitco_items
    # حذف عناوین تکراری (ممکن است دو منبع خبر مشابه را با عنوان مشابه پوشش دهند)
    unique = []
    seen = set()
    for item in combined:
        key = item["title"].lower()[:60]
        if key not in seen:
            seen.add(key)
            unique.append(item)

    return unique[:max_total]


# =============================================================================
# تحلیل احساسات و تفسیر معنایی اخبار با LM Studio (مدل زبانی بزرگ محلی)
# =============================================================================
def analyze_news_with_llm(news_items, lm_studio_url, model_name):
    import requests

    if not news_items:
        return None

    headlines_block = "\n".join(f"- {item['title']} ({item['source']})" for item in news_items)

    prompt = (
        "You are a financial news analyst specializing in the gold market. "
        "Analyze the following recent headlines and assess their likely net impact on the gold price.\n\n"
        f"Headlines:\n{headlines_block}\n\n"
        "Respond with ONLY a single JSON object, no other text, no markdown fences, in exactly this form:\n"
        '{"score": <float between -1.0 and 1.0, where +1 means strongly bullish for gold and -1 means strongly bearish>, '
        '"label": "<Positive|Negative|Neutral>", '
        '"interpretation": "<one or two sentence explanation in English of WHY these headlines push gold up or down>"}'
    )

    try:
        resp = requests.post(
            f"{lm_studio_url.rstrip('/')}/v1/chat/completions",
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "max_tokens": 300,
            },
            timeout=30,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]

        # مدل‌های چت گاهی JSON را داخل ```json ... ``` می‌گذارند؛ این را پاک می‌کنیم
        content_clean = content.strip()
        if content_clean.startswith("```"):
            content_clean = content_clean.strip("`")
            content_clean = content_clean.replace("json\n", "").replace("json", "", 1).strip()

        # اگر مدل متن اضافه قبل/بعد از JSON گذاشته باشد، فقط بخش {...} را استخراج کن
        start_idx = content_clean.find("{")
        end_idx = content_clean.rfind("}")
        if start_idx == -1 or end_idx == -1:
            return None
        json_str = content_clean[start_idx:end_idx + 1]

        parsed = json.loads(json_str)
        score = float(parsed.get("score", 0.0))
        score = max(-1.0, min(1.0, score))
        interpretation = parsed.get("interpretation", "").strip()
        label_en = parsed.get("label", "Neutral")

        label_map_fa = {"Positive": "مثبت (صعودی)", "Negative": "منفی (نزولی)", "Neutral": "خنثی"}

        return {
            "score": round(score, 4),
            "label": label_map_fa.get(label_en, "خنثی"),
            "label_en": label_en,
            "interpretation": interpretation,
            "source": f"وب‌اسکرپینگ ({len(news_items)} خبر) + LM Studio ({model_name})",
            "headlines": [item["title"] for item in news_items],
        }
    except Exception:
        return None


def get_live_news_sentiment(lm_studio_url, model_name):
    news_items = fetch_latest_news(max_total=5)
    if not news_items:
        return None
    return analyze_news_with_llm(news_items, lm_studio_url, model_name)


def analyze_manual_news_text(news_text, lm_studio_url, model_name):
    manual_item = [{
        "title": news_text.strip(),
        "source": "متن وارد‌شده توسط کاربر",
        "link": "",
        "published": "",
    }]
    result = analyze_news_with_llm(manual_item, lm_studio_url, model_name)
    if result is not None:
        result["source"] = "تحلیل دستی (متن وارد‌شده توسط کاربر) + LM Studio"
    return result


# =============================================================================
# Sidebar
# =============================================================================
with st.sidebar:
    st.header(" تنظیمات")
    st.subheader(" تحلیل احساسات اخبار")

    lm_studio_url = st.text_input(
        "آدرس LM Studio API",
        value="http://127.0.0.1:1234",
        help="آدرس سروری که LM Studio روی آن در حال اجراست.",
    )
    lm_model_name = st.text_input(
        "نام مدل در LM Studio",
        value="qwen2.5-7b-instruct-1m",
        help="باید دقیقاً همان شناسه‌ی مدلی باشد که در LM Studio بارگذاری شده است.",
    )


# =============================================================================
# بررسی وجود مدل‌ها
# =============================================================================
if not artifacts_exist():
    st.error(
        " مدل‌ها یا فایل‌های دادهٔ آموزش در مسیر پروژه پیدا نشد.\n\n"
        "لطفاً ابتدا نوت‌بوک `train_local_vscode.ipynb` را اجرا کنید تا "
        "مدل‌ها، اسکیلر، متریک‌ها و `features.csv` ذخیره شوند.\n\n"
        f"مسیر مورد انتظار: `{PATHS['models_dir']}`"
    )
    st.stop()


# =============================================================================
# بارگذاری داده و مدل‌ها
# =============================================================================
with st.spinner("در حال بارگذاری داده و مدل‌ها ..."):
    df_raw, features_df = load_notebook_data(PATHS["raw_csv"], PATHS["features_csv"])
    artifacts = load_models_and_meta()

train_meta = artifacts["train_meta"]
FEATURE_COLS = artifacts["feature_cols"]
NN_FEATURE_COLS = train_meta["nn_feature_cols"]
TARGET_IDX = train_meta["target_idx_in_scaled"]
SEQ_LEN = train_meta["seq_len"]
metrics = artifacts["metrics"]
best_model_name = metrics["best_model"]


# =============================================================================
# عنوان اصلی
# =============================================================================
st.title(" سامانه تحلیل و پیش‌بینی قیمت طلا")
st.caption(
    f"داده‌ها از {features_df['Date'].min().date()} تا {features_df['Date'].max().date()} · "
    f"بهترین مدل بر اساس RMSE: **{best_model_name}**"
)

tab1, tab2, tab3 = st.tabs([" تحلیل اکتشافی", " مقایسه مدل‌ها", " پیش‌بینی زنده"])


# =============================================================================
# تب ۱: تحلیل اکتشافی
# =============================================================================
with tab1:
    plot_df = features_df

    st.subheader("قیمت و میانگین‌های متحرک")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df[TARGET_COL], name="قیمت واقعی", line=dict(width=1, color="#374151")))
    fig.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["MA_7"], name="میانگین ۷ روزه", line=dict(width=1.5, color="#3b82f6")))
    fig.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["MA_30"], name="میانگین ۳۰ روزه", line=dict(width=2, color="#f59e0b")))
    fig.update_layout(template="plotly_white", height=430, hovermode="x unified", legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("آمار توصیفی")
        desc = features_df[RAW_NUMERIC_COLS].describe().T
        st.dataframe(desc.style.format("{:.2f}"), use_container_width=True)

        st.subheader("گزارش مقادیر گمشده (داده خام)")
        st.dataframe(missing_value_report(df_raw, RAW_NUMERIC_COLS), use_container_width=True, hide_index=True)

    with col2:
        st.subheader("ماتریس همبستگی")
        corr = features_df[RAW_NUMERIC_COLS].corr()
        fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        fig_corr.update_layout(height=380, template="plotly_white", margin=dict(t=10, b=10))
        st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("تجمیع ۳۰ روزه (میانگین قیمت در برابر مجموع حجم)")
    df_ts = features_df.set_index("Date")
    resampled = df_ts.resample("30D").agg({TARGET_COL: "mean", "Volume": "sum"}).dropna()
    fig_dual = make_subplots(specs=[[{"secondary_y": True}]])
    fig_dual.add_trace(go.Scatter(x=resampled.index, y=resampled[TARGET_COL], name="میانگین قیمت", line=dict(color="#b45309", width=2)), secondary_y=False)
    fig_dual.add_trace(go.Bar(x=resampled.index, y=resampled["Volume"], name="مجموع حجم", opacity=0.35, marker_color="#6b7280"), secondary_y=True)
    fig_dual.update_layout(template="plotly_white", height=400, legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig_dual, use_container_width=True)

    st.subheader("تحلیل روند (رگرسیون خطی)")
    from sklearn.linear_model import LinearRegression
    x_numeric = np.arange(len(features_df)).reshape(-1, 1)
    lr = LinearRegression().fit(x_numeric, features_df[TARGET_COL].values)
    trend_line = lr.predict(x_numeric)
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(x=features_df["Date"], y=features_df[TARGET_COL], name="قیمت واقعی", line=dict(width=1, color="#374151")))
    fig_trend.add_trace(go.Scatter(x=features_df["Date"], y=trend_line, name="خط روند", line=dict(color="red", dash="dash", width=2)))
    fig_trend.update_layout(template="plotly_white", height=400)
    st.plotly_chart(fig_trend, use_container_width=True)
    st.caption(f"شیب روند: {lr.coef_[0]:.4f} واحد قیمت به ازای هر روز")

    with st.expander("تجزیه سری زمانی (Seasonal Decompose، دوره ۳۰ روزه)"):
        from statsmodels.tsa.seasonal import seasonal_decompose
        decomposition = seasonal_decompose(features_df.set_index("Date")[TARGET_COL], model="additive", period=30, extrapolate_trend="freq")
        fig_decomp = make_subplots(rows=4, cols=1, shared_xaxes=True,
                                    subplot_titles=("سری اصلی", "روند", "فصلی", "باقی‌مانده"))
        fig_decomp.add_trace(go.Scatter(x=decomposition.observed.index, y=decomposition.observed, showlegend=False), row=1, col=1)
        fig_decomp.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend, showlegend=False), row=2, col=1)
        fig_decomp.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, showlegend=False), row=3, col=1)
        fig_decomp.add_trace(go.Scatter(x=decomposition.resid.index, y=decomposition.resid, showlegend=False), row=4, col=1)
        fig_decomp.update_layout(height=750, template="plotly_white")
        st.plotly_chart(fig_decomp, use_container_width=True)


# =============================================================================
# تب ۲: مقایسه مدل‌ها
# =============================================================================
with tab2:
    st.subheader("جدول مقایسه‌ای معیارهای ارزیابی")

    tp = artifacts["test_predictions"]
    y_actual = np.asarray(tp["y_actual"], dtype=float)

    def _accuracy_precision(y_true, y_pred, mape):
        """دقت ≈ ۱۰۰−MAPE ؛ صحت ≈ R² به‌صورت درصد."""
        y_pred = np.asarray(y_pred, dtype=float)
        accuracy = 100.0 - float(mape)
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot else 0.0
        precision = max(0.0, r2) * 100.0
        return accuracy, precision

    rows = []
    for name in ["LSTM", "GRU", "XGBoost"]:
        acc, prec = _accuracy_precision(y_actual, tp[name], metrics[name]["MAPE"])
        rows.append({
            "مدل": name,
            "MAE": metrics[name]["MAE"],
            "RMSE": metrics[name]["RMSE"],
            "MAPE (%)": metrics[name]["MAPE"],
            "دقت (%)": acc,
            "صحت (%)": prec,
        })

    comparison_df = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)

    def highlight_best(row):
        if row["مدل"] == best_model_name:
            return ["background-color: #fef3c7; color: #111827; font-weight: 700"] * len(row)
        return ["color: #111827"] * len(row)

    st.dataframe(
        comparison_df.style.apply(highlight_best, axis=1).format({
            "MAE": "{:.3f}",
            "RMSE": "{:.3f}",
            "MAPE (%)": "{:.3f}",
            "دقت (%)": "{:.3f}",
            "صحت (%)": "{:.3f}",
        }),
        use_container_width=True,
        hide_index=True,
    )
    st.success(f" بهترین مدل بر اساس RMSE: **{best_model_name}**")

    st.subheader(f"پیش‌بینی در برابر واقعیت — {best_model_name}")
    dates = pd.to_datetime(tp["dates"])
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(x=dates, y=tp["y_actual"], name="واقعی", line=dict(width=2, color="#111827")))
    fig_pred.add_trace(go.Scatter(x=dates, y=tp[best_model_name], name=f"پیش‌بینی ({best_model_name})", line=dict(width=2, dash="dot", color="#dc2626")))
    fig_pred.update_layout(template="plotly_white", height=450, hovermode="x unified")
    st.plotly_chart(fig_pred, use_container_width=True)

    with st.expander(" تاریخچه Loss آموزش (LSTM و GRU)"):
        c1, c2 = st.columns(2)
        with c1:
            if artifacts["lstm_history"]:
                fig_l = go.Figure()
                fig_l.add_trace(go.Scatter(y=artifacts["lstm_history"]["loss"], name="Train Loss"))
                fig_l.add_trace(go.Scatter(y=artifacts["lstm_history"]["val_loss"], name="Validation Loss"))
                fig_l.update_layout(title="LSTM", template="plotly_white", height=350)
                st.plotly_chart(fig_l, use_container_width=True)
        with c2:
            if artifacts["gru_history"]:
                fig_g = go.Figure()
                fig_g.add_trace(go.Scatter(y=artifacts["gru_history"]["loss"], name="Train Loss"))
                fig_g.add_trace(go.Scatter(y=artifacts["gru_history"]["val_loss"], name="Validation Loss"))
                fig_g.update_layout(title="GRU", template="plotly_white", height=350)
                st.plotly_chart(fig_g, use_container_width=True)


# =============================================================================
# تب ۳: پیش‌بینی زنده
# =============================================================================
with tab3:
    progress = st.progress(0, text="در حال آماده‌سازی داده برای پیش‌بینی ...")

    scaler = artifacts["scaler"]
    last_window_raw = features_df[NN_FEATURE_COLS].values[-SEQ_LEN:]
    last_window_scaled = scaler.transform(last_window_raw)
    progress.progress(30, text="در حال اجرای مدل منتخب ...")

    def predict_next_nn(model, window_scaled):
        x = window_scaled.reshape(1, SEQ_LEN, len(NN_FEATURE_COLS))
        pred_scaled = model.predict(x, verbose=0).ravel()[0]
        return inverse_scale_target(scaler, [pred_scaled], NN_FEATURE_COLS, TARGET_IDX)[0]

    def predict_next_xgb(model, feat_row):
        return float(model.predict(feat_row.reshape(1, -1))[0])

    if best_model_name == "LSTM":
        model_prediction = predict_next_nn(artifacts["lstm_model"], last_window_scaled)
    elif best_model_name == "GRU":
        model_prediction = predict_next_nn(artifacts["gru_model"], last_window_scaled)
    else:
        last_feat_row = features_df[FEATURE_COLS].values[-1]
        model_prediction = predict_next_xgb(artifacts["xgb_model"], last_feat_row)

    progress.progress(60, text="در حال آماده‌سازی تحلیل احساسات بازار ...")
    progress.progress(100, text="آماده شد ")
    time.sleep(0.15)
    progress.empty()

    # --- دکمه‌ی به‌روزرسانی اخبار و تحلیل احساسات (خودکار، از وب) ---
    st.subheader(" تحلیل احساسات بازار (خودکار)")
    col_btn, col_status = st.columns([1, 3])
    with col_btn:
        refresh_clicked = st.button(" به‌روزرسانی اخبار و تحلیل", use_container_width=True)

    if refresh_clicked or "sentiment_result" not in st.session_state:
        with col_status:
            with st.spinner("در حال جمع‌آوری اخبار از وب و تحلیل با مدل زبانی محلی ..."):
                live_sentiment = get_live_news_sentiment(lm_studio_url, lm_model_name)
        if live_sentiment is not None:
            st.session_state["sentiment_result"] = live_sentiment
            st.session_state["sentiment_is_manual"] = False
        else:
            fallback = compute_simulated_sentiment(features_df)
            fallback["source"] = "شبیه‌ساز داخلی (بر اساس نوسان قیمت و حجم) — اخبار یا LM Studio در دسترس نبودند"
            fallback["interpretation"] = ""
            st.session_state["sentiment_result"] = fallback
            with col_status:
                st.warning(
                    " جمع‌آوری اخبار از وب یا اتصال به LM Studio ناموفق بود؛ از شبیه‌ساز داخلی استفاده شد. "
                    "مطمئن شوید LM Studio روشن است و مدل بارگذاری شده، و اتصال اینترنت برقرار است."
                )

    # --- بخش دستی: تحلیل یک خبر دلخواه که خودِ کاربر وارد می‌کند ---
    st.divider()
    st.subheader(" تحلیل دستی یک خبر")
    st.caption(
        "متن یک خبر یا تیتر را اینجا وارد کن تا مدل زبانی محلی امتیاز و تفسیر آن را بدهد، "
        "و ببینی همین یک خبر چطور روی پیش‌بینی نهایی قیمت اثر می‌گذارد."
    )
    manual_news_text = st.text_area(
        "متن خبر",
        placeholder="مثلاً: Federal Reserve signals possible rate cuts amid cooling inflation data...",
        height=100,
    )
    manual_analyze_clicked = st.button(" تحلیل این خبر")

    if manual_analyze_clicked:
        if not manual_news_text.strip():
            st.warning("لطفاً ابتدا متن خبر را وارد کن.")
        else:
            with st.spinner("در حال تحلیل خبر با مدل زبانی محلی ..."):
                manual_result = analyze_manual_news_text(manual_news_text, lm_studio_url, lm_model_name)
            if manual_result is None:
                st.error(
                    " تحلیل ناموفق بود. مطمئن شوید LM Studio روشن است، مدل "
                    f"`{lm_model_name}` در آن بارگذاری شده، و آدرس `{lm_studio_url}` درست است."
                )
            else:
                st.session_state["sentiment_result"] = manual_result
                st.session_state["sentiment_is_manual"] = True

    if st.session_state.get("sentiment_is_manual") and manual_analyze_clicked:
        st.success("نتیجه‌ی تحلیل دستی به‌عنوان منبع احساسات فعلی برای پیش‌بینی نهایی استفاده می‌شود.")

    sentiment = st.session_state["sentiment_result"]

    # --- ترکیب پیش‌بینی ---
    # از میانگین قدر مطلق بازده‌های روزانه (نه میانگین جبری که نزدیک صفر است)
    # استفاده می‌شود تا تأثیر امتیاز احساسات در پیش‌بینی نهایی واقعاً قابل‌مشاهده باشد.
    average_price_change = get_average_price_swing(features_df, lookback=30)
    final_prediction = combine_prediction(model_prediction, sentiment["score"], average_price_change)

    last_price = features_df[TARGET_COL].iloc[-1]
    last_date = features_df["Date"].iloc[-1]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"""<div class="metric-card"><div class="label">آخرین قیمت ثبت‌شده ({last_date.date()})</div>
            <div class="value">{last_price:,.2f}</div></div>""",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""<div class="metric-card"><div class="label">پیش‌بینی مدل ({best_model_name})</div>
            <div class="value">{model_prediction:,.2f}</div></div>""",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""<div class="metric-card"><div class="label">پیش‌بینی نهایی ترکیبی (روز بعد)</div>
            <div class="value">{final_prediction:,.2f}</div></div>""",
            unsafe_allow_html=True,
        )

    st.write("")
    st.caption(f"منبع تحلیل احساسات: {sentiment.get('source', 'نامشخص')} · امتیاز: {sentiment['score']:+.2f} ({sentiment['label']})")

    if sentiment.get("interpretation"):
        st.info(f" **تفسیر مدل زبانی از اخبار:** {sentiment['interpretation']}")

    if sentiment.get("headlines"):
        with st.expander(f" {len(sentiment['headlines'])} خبری که تحلیل شدند"):
            for h in sentiment["headlines"]:
                st.markdown(f"- {h}")

