import io
import math
from datetime import datetime
from html import escape

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ==========================================================
# 1. CẤU HÌNH CHUNG
# ==========================================================
st.set_page_config(
    page_title="Premier League AI Predictor | Nhóm 5",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_FILE = "nhom5_bongda_train.xlsx"
REQUIRED_COLS = ["HomeTeam", "AwayTeam", "FTHG", "FTAG"]
MODEL_NAMES = [
    "K-Nearest Neighbors",
    "Logistic Regression",
    "Random Forest",
    "Gradient Boosting",
]
ENSEMBLE = "Ensemble (trung bình các mô hình)"
RESULT_LABEL = {"H": "Chủ nhà thắng", "D": "Hòa", "A": "Khách thắng"}
RESULT_COLOR = {"H": "#2f6bff", "D": "#8b93a7", "A": "#e90052"}
ACCENT = {
    "H": ("#2f6bff", "#173a9e"),
    "D": ("#7c849a", "#444b5e"),
    "A": ("#e90052", "#8a0a3a"),
}

st.session_state.setdefault("history", [])

# ==========================================================
# 2. CSS GIAO DIỆN (hoạt động cả nền sáng lẫn nền tối)
# ==========================================================
st.markdown(
    """
<style>
:root {
    --pl-purple: #3d195b;
    --pl-purple-2: #6b2fa0;
    --pl-green: #00c46a;
    --pl-pink: #e90052;
    --card-bg: rgba(127,127,127,.08);
    --card-border: rgba(127,127,127,.22);
}
.block-container { padding-top: 2rem; max-width: 1280px; }

/* ---------- HERO ---------- */
.hero {
    position: relative; overflow: hidden;
    background: linear-gradient(120deg, #2b0d45 0%, #4b1d78 55%, #7b2fb5 100%);
    border-radius: 22px; padding: 34px 38px; margin-bottom: 22px;
    box-shadow: 0 14px 40px rgba(61,25,91,.35);
}
.hero::after {
    content: ""; position: absolute; right: -60px; top: -60px;
    width: 280px; height: 280px; border-radius: 50%;
    background: radial-gradient(circle, rgba(0,255,133,.35), transparent 70%);
}
.hero-badge {
    display: inline-block; background: rgba(255,255,255,.14); color: #e9d8ff;
    padding: 5px 14px; border-radius: 999px; font-size: 12.5px;
    font-weight: 700; letter-spacing: .6px; margin-bottom: 12px;
}
.hero-title { color: #fff; font-size: 38px; font-weight: 900; line-height: 1.15; margin: 0; }
.hero-title span { color: #00ff85; }
.hero-sub { color: #d9c8ee; font-size: 16px; margin-top: 8px; max-width: 760px; }

/* ---------- TIÊU ĐỀ MỤC ---------- */
.section-title {
    font-size: 21px; font-weight: 800; margin: 6px 0 12px 0;
    padding-left: 12px; border-left: 5px solid var(--pl-green);
}

/* ---------- THẺ ĐỘI BÓNG ---------- */
.team-card { border-radius: 16px; padding: 16px 20px; color: #fff; margin-top: 10px;
    box-shadow: 0 8px 22px rgba(0,0,0,.16); }
.team-home { background: linear-gradient(135deg, #1b44c8, #3d8bff); }
.team-away { background: linear-gradient(135deg, #b8003a, #ff4d79); }
.tc-label { font-size: 12px; font-weight: 800; letter-spacing: 1px; opacity: .85; }
.tc-name { font-size: 26px; font-weight: 900; margin: 2px 0 10px 0; }
.tc-stats { display: flex; flex-wrap: wrap; gap: 8px; }
.tc-stats span { background: rgba(255,255,255,.18); border-radius: 8px;
    padding: 3px 10px; font-size: 13px; }
.vs-badge { text-align: center; font-size: 30px; font-weight: 900; margin-top: 34px;
    background: linear-gradient(90deg, #ff8c00, #e90052);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; }

/* ---------- NÚT DỰ ĐOÁN ---------- */
.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {
    width: 100%; height: 58px; font-size: 19px; font-weight: 800;
    border-radius: 14px; border: none; color: #fff;
    background: linear-gradient(90deg, #3d195b, #7b2fb5 60%, #e90052);
    box-shadow: 0 8px 22px rgba(123,47,181,.45);
    transition: transform .2s ease, box-shadow .2s ease;
}
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover {
    transform: translateY(-2px) scale(1.01);
    box-shadow: 0 12px 28px rgba(123,47,181,.6); color: #fff;
}

/* ---------- THẺ KẾT QUẢ ---------- */
.result-card {
    border-radius: 20px; padding: 24px 28px; color: #fff;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    box-shadow: 0 12px 32px rgba(0,0,0,.22);
}
.rc-label { font-size: 12.5px; font-weight: 800; letter-spacing: 1.2px; opacity: .85; }
.rc-title { font-size: 32px; font-weight: 900; margin: 6px 0; line-height: 1.2; }
.rc-sub { font-size: 14.5px; opacity: .92; }

/* ---------- PHONG ĐỘ ---------- */
.form { display: inline-block; width: 32px; height: 32px; line-height: 32px;
    text-align: center; border-radius: 8px; font-weight: 800; color: #fff; margin-right: 6px; }
.form-W { background: #00a86b; } .form-D { background: #8b93a7; } .form-L { background: #e23b4e; }

/* ---------- METRIC & TAB ---------- */
div[data-testid="stMetric"] {
    background: var(--card-bg); border: 1px solid var(--card-border);
    border-left: 5px solid var(--pl-purple-2); padding: 12px 16px; border-radius: 12px;
}
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] { border-radius: 10px 10px 0 0; padding: 10px 18px; font-weight: 700; }
div[data-testid="stDataFrame"] { width: 100%; }
.footer { text-align: center; opacity: .6; font-size: 13px; margin-top: 30px; }
</style>
""",
    unsafe_allow_html=True,
)


# ==========================================================
# 3. HÀM TIỆN ÍCH
# ==========================================================
def show_df(data, **kwargs):
    """st.dataframe tương thích nhiều phiên bản Streamlit."""
    try:
        st.dataframe(data, width="stretch", **kwargs)
    except Exception:
        st.dataframe(data, use_container_width=True, **kwargs)


def style_fig(fig, height=320):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=45, b=10))
    return fig


@st.cache_data(show_spinner="📥 Đang tải dữ liệu...")
def load_data(content, filename):
    if content is None:
        df = pd.read_excel(DEFAULT_FILE)
    elif filename.lower().endswith(".csv"):
        df = pd.read_csv(io.BytesIO(content))
    else:
        df = pd.read_excel(io.BytesIO(content))

    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"File dữ liệu thiếu các cột bắt buộc: {', '.join(missing)}")

    df = df.dropna(subset=REQUIRED_COLS).copy()
    df["HomeTeam"] = df["HomeTeam"].astype(str).str.strip()
    df["AwayTeam"] = df["AwayTeam"].astype(str).str.strip()
    df["FTHG"] = pd.to_numeric(df["FTHG"], errors="coerce")
    df["FTAG"] = pd.to_numeric(df["FTAG"], errors="coerce")
    df = df.dropna(subset=["FTHG", "FTAG"])
    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
        df = df.sort_values("Date", kind="stable")

    df["Result"] = np.select([df["FTHG"] > df["FTAG"], df["FTHG"] == df["FTAG"]], ["H", "D"], "A")
    return df.reset_index(drop=True)


def compute_team_stats(df):
    """Thống kê sức mạnh từng đội (tấn công, phòng ngự, điểm/trận, sân nhà/khách)."""
    d = df.assign(
        HPts=np.select([df["Result"] == "H", df["Result"] == "D"], [3, 1], 0),
        APts=np.select([df["Result"] == "A", df["Result"] == "D"], [3, 1], 0),
    )
    lg_h, lg_a = d["FTHG"].mean(), d["FTAG"].mean()
    home = d.groupby("HomeTeam").agg(
        h_games=("FTHG", "size"), h_gf=("FTHG", "mean"), h_ga=("FTAG", "mean"), h_pts=("HPts", "mean")
    )
    away = d.groupby("AwayTeam").agg(
        a_games=("FTAG", "size"), a_gf=("FTAG", "mean"), a_ga=("FTHG", "mean"), a_pts=("APts", "mean")
    )
    s = home.join(away, how="outer")
    s[["h_games", "a_games"]] = s[["h_games", "a_games"]].fillna(0)
    s["h_gf"] = s["h_gf"].fillna(lg_h)
    s["h_ga"] = s["h_ga"].fillna(lg_a)
    s["h_pts"] = s["h_pts"].fillna(d["HPts"].mean())
    s["a_gf"] = s["a_gf"].fillna(lg_a)
    s["a_ga"] = s["a_ga"].fillna(lg_h)
    s["a_pts"] = s["a_pts"].fillna(d["APts"].mean())
    games = (s["h_games"] + s["a_games"]).clip(lower=1)
    s["games"] = s["h_games"] + s["a_games"]
    s["ppg"] = (s["h_pts"] * s["h_games"] + s["a_pts"] * s["a_games"]) / games
    s["gf"] = (s["h_gf"] * s["h_games"] + s["a_gf"] * s["a_games"]) / games
    s["ga"] = (s["h_ga"] * s["h_games"] + s["a_ga"] * s["a_games"]) / games
    s.index.name = "Team"
    return s


def feature_frame(matches, stats):
    """Biến mỗi trận đấu thành vector đặc trưng dựa trên sức mạnh hai đội."""
    fill = stats.mean()
    h = stats.reindex(matches["HomeTeam"].values).fillna(fill).reset_index(drop=True)
    a = stats.reindex(matches["AwayTeam"].values).fillna(fill).reset_index(drop=True)
    return pd.DataFrame(
        {
            "Điểm/trận (chủ nhà)": h["ppg"],
            "Điểm/trận (khách)": a["ppg"],
            "Điểm/trận sân nhà (chủ)": h["h_pts"],
            "Điểm/trận sân khách (khách)": a["a_pts"],
            "Bàn thắng sân nhà (chủ)": h["h_gf"],
            "Bàn thua sân nhà (chủ)": h["h_ga"],
            "Bàn thắng sân khách (khách)": a["a_gf"],
            "Bàn thua sân khách (khách)": a["a_ga"],
            "Chênh lệch điểm/trận": h["ppg"] - a["ppg"],
            "Chênh lệch hiệu số": (h["gf"] - h["ga"]) - (a["gf"] - a["ga"]),
        }
    )


def make_model(name, k):
    if name == "K-Nearest Neighbors":
        clf = KNeighborsClassifier(n_neighbors=k, weights="distance")
    elif name == "Logistic Regression":
        clf = LogisticRegression(max_iter=1000)
    elif name == "Random Forest":
        clf = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=5, random_state=42)
    else:
        clf = GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=3, random_state=42)
    return make_pipeline(StandardScaler(), clf)


@st.cache_resource(show_spinner="🧠 Đang huấn luyện các mô hình AI...")
def train_all(df, test_size, k):
    try:
        tr, te = train_test_split(df, test_size=test_size, random_state=42, stratify=df["Result"])
    except ValueError:
        tr, te = train_test_split(df, test_size=test_size, random_state=42)

    # Đánh giá: thống kê chỉ tính trên tập huấn luyện để không "nhìn trộm" tập kiểm tra
    stats_tr = compute_team_stats(tr)
    Xtr, Xte = feature_frame(tr, stats_tr), feature_frame(te, stats_tr)
    ytr, yte = tr["Result"], te["Result"]
    labels = ["H", "D", "A"]

    rows, details = [], {}
    for name in MODEL_NAMES:
        m = make_model(name, max(1, min(k, len(tr) - 1)))
        m.fit(Xtr, ytr)
        pred = m.predict(Xte)
        rep = classification_report(yte, pred, labels=labels, output_dict=True, zero_division=0)
        rep.pop("accuracy", None)
        rep_df = pd.DataFrame(rep).T.loc[labels + ["macro avg", "weighted avg"]]
        rep_df.index = ["Chủ nhà thắng", "Hòa", "Khách thắng", "Trung bình macro", "Trung bình có trọng số"]
        rep_df = rep_df.rename(
            columns={"precision": "Precision", "recall": "Recall", "f1-score": "F1", "support": "Số trận"}
        )
        details[name] = {"cm": confusion_matrix(yte, pred, labels=labels), "report": rep_df}
        rows.append(
            {
                "Mô hình": name,
                "Accuracy": accuracy_score(yte, pred),
                "F1 (macro)": f1_score(yte, pred, average="macro", zero_division=0),
            }
        )

    # Mô hình dùng để dự đoán thật: huấn luyện lại trên toàn bộ dữ liệu
    stats_full = compute_team_stats(df)
    Xf = feature_frame(df, stats_full)
    models = {n: make_model(n, max(1, min(k, len(df) - 1))).fit(Xf, df["Result"]) for n in MODEL_NAMES}

    importance = pd.DataFrame(
        {"Đặc trưng": Xf.columns, "Độ quan trọng": models["Random Forest"][-1].feature_importances_}
    ).sort_values("Độ quan trọng")

    return {
        "summary": pd.DataFrame(rows),
        "details": details,
        "models": models,
        "stats": stats_full,
        "baseline": float((yte == "H").mean()),
        "n_train": len(tr),
        "n_test": len(te),
        "importance": importance,
    }


def proba_dict(model, X):
    out = {"H": 0.0, "D": 0.0, "A": 0.0}
    for c, v in zip(model.classes_, model.predict_proba(X)[0]):
        out[c] = float(v)
    return out


def predict_match(home, away, model_choice, bundle):
    X = feature_frame(pd.DataFrame({"HomeTeam": [home], "AwayTeam": [away]}), bundle["stats"])
    if model_choice == ENSEMBLE:
        plist = [proba_dict(m, X) for m in bundle["models"].values()]
        probs = {k: float(np.mean([p[k] for p in plist])) for k in "HDA"}
    else:
        probs = proba_dict(bundle["models"][model_choice], X)
    return max(probs, key=probs.get), probs


def poisson_pmf(k, lam):
    return math.exp(-lam) * lam**k / math.factorial(k)


def scoreline_matrix(home, away, stats, lg_h, lg_a, max_goals=6):
    h, a = stats.loc[home], stats.loc[away]
    lam_h = max(0.2, h["h_gf"] * a["a_ga"] / lg_h)
    lam_a = max(0.2, a["a_gf"] * h["h_ga"] / lg_a)
    ph = np.array([poisson_pmf(i, lam_h) for i in range(max_goals + 1)])
    pa = np.array([poisson_pmf(i, lam_a) for i in range(max_goals + 1)])
    m = np.outer(ph, pa)
    return lam_h, lam_a, m / m.sum()


def league_table(data):
    home = pd.DataFrame(
        {
            "Đội": data["HomeTeam"], "BT": data["FTHG"], "BB": data["FTAG"],
            "Thắng": (data["Result"] == "H").astype(int),
            "Hòa": (data["Result"] == "D").astype(int),
            "Thua": (data["Result"] == "A").astype(int),
        }
    )
    away = pd.DataFrame(
        {
            "Đội": data["AwayTeam"], "BT": data["FTAG"], "BB": data["FTHG"],
            "Thắng": (data["Result"] == "A").astype(int),
            "Hòa": (data["Result"] == "D").astype(int),
            "Thua": (data["Result"] == "H").astype(int),
        }
    )
    t = pd.concat([home, away]).groupby("Đội").agg(
        Trận=("Thắng", "size"), Thắng=("Thắng", "sum"), Hòa=("Hòa", "sum"),
        Thua=("Thua", "sum"), BT=("BT", "sum"), BB=("BB", "sum"),
    )
    t["HS"] = t["BT"] - t["BB"]
    t["Điểm"] = t["Thắng"] * 3 + t["Hòa"]
    t = t.sort_values(["Điểm", "HS", "BT"], ascending=False).reset_index()
    t.insert(0, "Hạng", range(1, len(t) + 1))
    return t


def team_matches(df, team):
    m = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)].copy()
    home_map = {"H": "W", "D": "D", "A": "L"}
    away_map = {"H": "L", "D": "D", "A": "W"}
    m["Outcome"] = np.where(m["HomeTeam"] == team, m["Result"].map(home_map), m["Result"].map(away_map))
    return m


def form_html(df, team, n=5):
    letters = {"W": "T", "D": "H", "L": "B"}
    out = team_matches(df, team).tail(n)["Outcome"].tolist()
    if not out:
        return "<i>Chưa có dữ liệu</i>"
    return "".join(f'<span class="form form-{o}">{letters[o]}</span>' for o in out)


def team_card(team, side, stats, ranks):
    s = stats.loc[team]
    label = "🏠 ĐỘI CHỦ NHÀ" if side == "home" else "✈️ ĐỘI KHÁCH"
    return (
        f'<div class="team-card team-{side}"><div class="tc-label">{label}</div>'
        f'<div class="tc-name">{escape(team)}</div><div class="tc-stats">'
        f'<span>Hạng <b>#{ranks.get(team, "-")}</b></span>'
        f'<span>Điểm/trận <b>{s["ppg"]:.2f}</b></span>'
        f'<span>Ghi <b>{s["gf"]:.2f}</b></span>'
        f'<span>Thủng <b>{s["ga"]:.2f}</b></span></div></div>'
    )


def norm(series_all, v):
    lo, hi = series_all.min(), series_all.max()
    return 0.5 if hi == lo else float((v - lo) / (hi - lo))


def swap_teams():
    st.session_state["home_team"], st.session_state["away_team"] = (
        st.session_state["away_team"],
        st.session_state["home_team"],
    )


def clear_history():
    st.session_state["history"] = []


# ==========================================================
# 4. HERO + SIDEBAR (CÀI ĐẶT)
# ==========================================================
st.markdown(
    """
<div class="hero">
    <div class="hero-badge">🎓 ĐỒ ÁN · TRÍ TUỆ NHÂN TẠO & ỨNG DỤNG · NHÓM 5</div>
    <div class="hero-title">Premier League <span>AI Predictor</span></div>
    <div class="hero-sub">Dự đoán kết quả Ngoại hạng Anh bằng Học máy: so sánh nhiều thuật toán,
    phân tích sức mạnh đội bóng, tỷ số kỳ vọng, đối đầu và phong độ.</div>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## ⚽ AI Bóng Đá")
    st.caption("Nhóm 5 · Trí tuệ nhân tạo & Ứng dụng")
    st.divider()
    st.markdown("#### 📂 Dữ liệu")
    uploaded = st.file_uploader(
        "Tải dữ liệu của bạn (.xlsx / .csv)",
        type=["xlsx", "csv"],
        help="Cần các cột: HomeTeam, AwayTeam, FTHG, FTAG. Cột Date, Season là tùy chọn. "
        "Để trống sẽ dùng file mặc định của nhóm.",
    )
    st.markdown("#### 🧪 Huấn luyện")
    test_size = st.slider("Tỷ lệ tập kiểm tra", 10, 40, 20, step=5, format="%d%%") / 100
    k_neighbors = st.slider("Số láng giềng K (cho KNN)", 1, 25, 7, step=2)
    celebrate = st.toggle("🎉 Hiệu ứng ăn mừng", value=True)

try:
    df = load_data(uploaded.getvalue() if uploaded else None, uploaded.name if uploaded else "")
except FileNotFoundError:
    st.error(f"Không tìm thấy file `{DEFAULT_FILE}`. Hãy đặt file cùng thư mục với app.py hoặc tải file lên ở thanh bên.")
    st.stop()
except Exception as e:
    st.error(f"Không đọc được dữ liệu: {e}")
    st.stop()

teams = sorted(set(df["HomeTeam"]) | set(df["AwayTeam"]))
if len(df) < 30 or len(teams) < 2:
    st.error("Dữ liệu quá ít (cần tối thiểu 30 trận và 2 đội) để huấn luyện mô hình.")
    st.stop()

bundle = train_all(df, test_size, k_neighbors)
stats = bundle["stats"]
lg_h, lg_a = df["FTHG"].mean(), df["FTAG"].mean()
table_all = league_table(df)
ranks = table_all.set_index("Đội")["Hạng"].to_dict()

with st.sidebar:
    st.divider()
    st.markdown("#### 🤖 Mô hình dự đoán")
    model_choice = st.selectbox("Chọn mô hình", [ENSEMBLE] + MODEL_NAMES, label_visibility="collapsed")
    best = bundle["summary"].sort_values("Accuracy", ascending=False).iloc[0]
    st.caption(f"Chính xác nhất trên tập kiểm tra: **{best['Mô hình']}** ({best['Accuracy']:.1%})")
    with st.expander("ℹ️ Thông tin đồ án"):
        st.markdown(
            "- **Môn học:** Trí tuệ nhân tạo và Ứng dụng\n"
            "- **Nhóm:** Nhóm 5 · ĐH GTVT TP.HCM\n"
            "- **Thuật toán:** KNN, Logistic Regression, Random Forest, Gradient Boosting\n"
            "- **Bài toán:** Phân loại 3 lớp (Thắng / Hòa / Thua)"
        )

# ==========================================================
# 5. CÁC TAB CHÍNH
# ==========================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["🔮 Dự đoán", "🏆 Bảng xếp hạng", "📊 Thống kê", "🧠 Mô hình AI", "🗂️ Dữ liệu"]
)

# ----------------------------------------------------------
# TAB 1: DỰ ĐOÁN
# ----------------------------------------------------------
with tab1:
    if st.session_state.get("home_team") not in teams:
        st.session_state["home_team"] = teams[0]
    if st.session_state.get("away_team") not in teams:
        st.session_state["away_team"] = teams[1]

    st.markdown('<div class="section-title">🏟️ Thiết lập trận đấu</div>', unsafe_allow_html=True)
    c1, cvs, c2 = st.columns([5, 1.2, 5])
    with c1:
        home_team = st.selectbox("Đội chủ nhà", teams, key="home_team")
        st.markdown(team_card(home_team, "home", stats, ranks), unsafe_allow_html=True)
    with cvs:
        st.markdown('<div class="vs-badge">VS</div>', unsafe_allow_html=True)
        st.button("⇄ Đổi sân", on_click=swap_teams, key="swap", help="Hoán đổi đội chủ nhà và đội khách")
    with c2:
        away_team = st.selectbox("Đội khách", teams, key="away_team")
        st.markdown(team_card(away_team, "away", stats, ranks), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("🔮 PHÂN TÍCH & DỰ ĐOÁN", type="primary", key="predict")

    if predict_clicked:
        if home_team == away_team:
            st.error("🚨 Đội chủ nhà và đội khách không thể trùng nhau. Vui lòng chọn lại!")
        else:
            pred, probs = predict_match(home_team, away_team, model_choice, bundle)
            st.session_state["last"] = {
                "home": home_team, "away": away_team, "model": model_choice, "pred": pred, "probs": probs,
            }
            st.session_state["history"].insert(
                0,
                {
                    "Thời gian": datetime.now().strftime("%H:%M:%S"),
                    "Chủ nhà": home_team, "Khách": away_team, "Mô hình": model_choice,
                    "Dự đoán": RESULT_LABEL[pred],
                    "P(Chủ thắng) %": round(probs["H"] * 100, 1),
                    "P(Hòa) %": round(probs["D"] * 100, 1),
                    "P(Khách thắng) %": round(probs["A"] * 100, 1),
                },
            )
            if celebrate:
                if pred == "H":
                    st.balloons()
                elif pred == "A":
                    st.snow()

    last = st.session_state.get("last")
    if home_team == away_team:
        st.warning("Hãy chọn hai đội khác nhau để bắt đầu dự đoán.")
    elif last and last["home"] == home_team and last["away"] == away_team and last["model"] == model_choice:
        pred, probs = last["pred"], last["probs"]
        a1, a2 = ACCENT[pred]
        if pred == "H":
            title, icon = f"{home_team} thắng", "🏆"
        elif pred == "A":
            title, icon = f"{away_team} thắng", "🔥"
        else:
            title, icon = "Hai đội hòa nhau", "⚖️"
        conf = probs[pred]
        level = "Cao" if conf >= 0.6 else ("Trung bình" if conf >= 0.45 else "Thấp, trận khó đoán")

        st.markdown('<div class="section-title">🎯 Kết quả dự đoán</div>', unsafe_allow_html=True)
        r1, r2 = st.columns([1.15, 1])
        with r1:
            st.markdown(
                f'<div class="result-card" style="--accent:{a1};--accent2:{a2}">'
                f'<div class="rc-label">AI DỰ ĐOÁN</div>'
                f'<div class="rc-title">{icon} {escape(title)}</div>'
                f'<div class="rc-sub">Xác suất {conf:.0%} · Độ tin cậy: {level}<br>Mô hình: {escape(model_choice)}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            m1.metric(f"{home_team} thắng", f"{probs['H']:.0%}")
            m2.metric("Hòa", f"{probs['D']:.0%}")
            m3.metric(f"{away_team} thắng", f"{probs['A']:.0%}")
        with r2:
            fig = go.Figure(
                go.Pie(
                    labels=[f"{home_team} thắng", "Hòa", f"{away_team} thắng"],
                    values=[probs["H"], probs["D"], probs["A"]],
                    hole=0.62, sort=False, textinfo="percent",
                    marker=dict(colors=[RESULT_COLOR["H"], RESULT_COLOR["D"], RESULT_COLOR["A"]]),
                )
            )
            fig.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.05))
            st.plotly_chart(style_fig(fig, 300))

        s1, s2, s3, s4 = st.tabs(
            ["⚽ Tỷ số dự kiến", "🤝 Đối đầu & phong độ", "🕸️ Radar sức mạnh", "🧩 So sánh mô hình"]
        )

        # ---- Tỷ số dự kiến (Poisson) ----
        with s1:
            lam_h, lam_a, M = scoreline_matrix(home_team, away_team, stats, lg_h, lg_a)
            pH, pD, pA = float(np.tril(M, -1).sum()), float(np.trace(M)), float(np.triu(M, 1).sum())
            st.caption(
                "Mô hình Poisson ước lượng số bàn kỳ vọng của mỗi đội từ sức tấn công và phòng ngự "
                "(sân nhà/sân khách), từ đó suy ra xác suất từng tỷ số."
            )
            k1, k2, k3 = st.columns(3)
            k1.metric(f"Bàn kỳ vọng · {home_team}", f"{lam_h:.2f}")
            k2.metric(f"Bàn kỳ vọng · {away_team}", f"{lam_a:.2f}")
            k3.metric("Tổng bàn kỳ vọng", f"{lam_h + lam_a:.2f}")

            left, right = st.columns([1, 1.3])
            with left:
                flat = [(i, j, M[i, j]) for i in range(M.shape[0]) for j in range(M.shape[1])]
                top = sorted(flat, key=lambda x: x[2], reverse=True)[:6]
                top_df = pd.DataFrame(
                    {"Tỷ số": [f"{i} - {j}" for i, j, _ in top], "Xác suất": [p * 100 for _, _, p in top]}
                )
                st.markdown("**Top tỷ số có khả năng cao nhất**")
                show_df(
                    top_df,
                    hide_index=True,
                    column_config={
                        "Xác suất": st.column_config.ProgressColumn(
                            "Xác suất", min_value=0, max_value=float(top_df["Xác suất"].max()) * 1.1, format="%.1f%%"
                        )
                    },
                )
                st.caption(f"Poisson: chủ thắng {pH:.0%} · hòa {pD:.0%} · khách thắng {pA:.0%}")
            with right:
                fig = go.Figure(
                    go.Heatmap(
                        z=M * 100, x=list(range(M.shape[1])), y=list(range(M.shape[0])),
                        colorscale="Purples", texttemplate="%{z:.1f}", showscale=False,
                    )
                )
                fig.update_layout(
                    title="Ma trận xác suất tỷ số (%)",
                    xaxis_title=f"Bàn thắng {away_team}", yaxis_title=f"Bàn thắng {home_team}",
                )
                st.plotly_chart(style_fig(fig, 360))

        # ---- Đối đầu & phong độ ----
        with s2:
            mask1 = (df["HomeTeam"] == home_team) & (df["AwayTeam"] == away_team)
            mask2 = (df["HomeTeam"] == away_team) & (df["AwayTeam"] == home_team)
            h2h = df[mask1 | mask2]
            st.markdown("**Lịch sử đối đầu trong dữ liệu**")
            if h2h.empty:
                st.info("Hai đội chưa gặp nhau trong bộ dữ liệu này.")
            else:
                hw = int(((mask1 & (df["Result"] == "H")) | (mask2 & (df["Result"] == "A"))).sum())
                aw = int(((mask1 & (df["Result"] == "A")) | (mask2 & (df["Result"] == "H"))).sum())
                dr = int((h2h["Result"] == "D").sum())
                d1, d2, d3, d4 = st.columns(4)
                d1.metric("Số trận", len(h2h))
                d2.metric(f"{home_team} thắng", hw)
                d3.metric("Hòa", dr)
                d4.metric(f"{away_team} thắng", aw)
                disp = h2h.tail(10).iloc[::-1].copy()
                disp["Tỷ số"] = disp["FTHG"].astype(str) + " - " + disp["FTAG"].astype(str)
                cols = ["HomeTeam", "Tỷ số", "AwayTeam"]
                if "Date" in disp.columns:
                    disp["Ngày"] = disp["Date"].dt.strftime("%d/%m/%Y")
                    cols = ["Ngày"] + cols
                show_df(disp[cols].rename(columns={"HomeTeam": "Chủ nhà", "AwayTeam": "Khách"}), hide_index=True)

            st.markdown("**Phong độ 5 trận gần nhất** (T: thắng · H: hòa · B: bại)")
            f1c, f2c = st.columns(2)
            f1c.markdown(f"**{escape(home_team)}**", unsafe_allow_html=True)
            f1c.markdown(form_html(df, home_team), unsafe_allow_html=True)
            f2c.markdown(f"**{escape(away_team)}**", unsafe_allow_html=True)
            f2c.markdown(form_html(df, away_team), unsafe_allow_html=True)
            if "Date" not in df.columns:
                st.caption("Dữ liệu không có cột Date nên 'gần nhất' được tính theo thứ tự dòng trong file.")

        # ---- Radar ----
        with s3:
            cats = ["Điểm/trận", "Tấn công", "Phòng ngự", "Phong độ sân nhà/khách", "Hiệu số bàn thắng"]

            def radar_values(team, venue):
                r = stats.loc[team]
                venue_col = "h_pts" if venue == "home" else "a_pts"
                vals = [
                    norm(stats["ppg"], r["ppg"]),
                    norm(stats["gf"], r["gf"]),
                    norm(-stats["ga"], -r["ga"]),
                    norm(stats[venue_col], r[venue_col]),
                    norm(stats["gf"] - stats["ga"], r["gf"] - r["ga"]),
                ]
                return vals + vals[:1]

            fig = go.Figure()
            fig.add_trace(
                go.Scatterpolar(
                    r=radar_values(home_team, "home"), theta=cats + cats[:1], fill="toself", name=home_team,
                    line=dict(color="#2f6bff"), fillcolor="rgba(47,107,255,0.30)",
                )
            )
            fig.add_trace(
                go.Scatterpolar(
                    r=radar_values(away_team, "away"), theta=cats + cats[:1], fill="toself", name=away_team,
                    line=dict(color="#e90052"), fillcolor="rgba(233,0,82,0.30)",
                )
            )
            fig.update_layout(polar=dict(radialaxis=dict(range=[0, 1], showticklabels=False)))
            st.plotly_chart(style_fig(fig, 420))
            st.caption("Mỗi trục được chuẩn hóa 0–1 so với toàn bộ các đội trong dữ liệu (càng xa tâm càng mạnh).")

        # ---- So sánh mô hình ----
        with s4:
            X1 = feature_frame(pd.DataFrame({"HomeTeam": [home_team], "AwayTeam": [away_team]}), stats)
            rows = []
            for name, mdl in bundle["models"].items():
                p = proba_dict(mdl, X1)
                rows.append((name, p))
            ens = {k: float(np.mean([p[k] for _, p in rows])) for k in "HDA"}
            rows.append(("Ensemble (trung bình)", ens))
            _, _, Mp = scoreline_matrix(home_team, away_team, stats, lg_h, lg_a)
            rows.append(
                ("Poisson (bàn thắng kỳ vọng)",
                 {"H": float(np.tril(Mp, -1).sum()), "D": float(np.trace(Mp)), "A": float(np.triu(Mp, 1).sum())})
            )
            comp = pd.DataFrame(
                [
                    {
                        "Mô hình": n,
                        "Dự đoán": RESULT_LABEL[max(p, key=p.get)],
                        "Chủ thắng": p["H"] * 100,
                        "Hòa": p["D"] * 100,
                        "Khách thắng": p["A"] * 100,
                    }
                    for n, p in rows
                ]
            )
            pct = st.column_config.NumberColumn(format="%.0f%%")
            show_df(
                comp, hide_index=True,
                column_config={"Chủ thắng": pct, "Hòa": pct, "Khách thắng": pct},
            )
            votes = comp["Dự đoán"].value_counts()
            st.caption(f"Đa số các mô hình nghiêng về: **{votes.index[0]}** ({votes.iloc[0]}/{len(comp)} mô hình).")
    elif last:
        st.info("Bạn vừa đổi đội hoặc mô hình. Bấm **PHÂN TÍCH & DỰ ĐOÁN** để cập nhật kết quả.")

    if st.session_state["history"]:
        with st.expander(f"🕘 Lịch sử dự đoán ({len(st.session_state['history'])})"):
            hist = pd.DataFrame(st.session_state["history"])
            show_df(hist, hide_index=True)
            h1, h2 = st.columns(2)
            h1.download_button(
                "⬇️ Tải lịch sử (CSV)", hist.to_csv(index=False).encode("utf-8-sig"),
                "lich_su_du_doan.csv", "text/csv",
            )
            h2.button("🗑️ Xóa lịch sử", on_click=clear_history)

# ----------------------------------------------------------
# TAB 2: BẢNG XẾP HẠNG
# ----------------------------------------------------------
with tab2:
    st.markdown('<div class="section-title">🏆 Bảng xếp hạng tính từ dữ liệu</div>', unsafe_allow_html=True)
    data_t = df
    if "Season" in df.columns:
        seasons = ["Tất cả"] + sorted(df["Season"].dropna().astype(str).unique().tolist())
        sel = st.selectbox("Mùa giải", seasons)
        if sel != "Tất cả":
            data_t = df[df["Season"].astype(str) == sel]
    table = league_table(data_t)

    t1, t2, t3, t4 = st.columns(4)
    t1.metric("🥇 Dẫn đầu", table.iloc[0]["Đội"], f"{int(table.iloc[0]['Điểm'])} điểm")
    t2.metric("⚔️ Tấn công tốt nhất", table.sort_values("BT", ascending=False).iloc[0]["Đội"],
              f"{int(table['BT'].max())} bàn")
    t3.metric("🛡️ Phòng ngự tốt nhất", table.sort_values("BB").iloc[0]["Đội"], f"{int(table['BB'].min())} bàn thua")
    t4.metric("📈 Hiệu số cao nhất", table.sort_values("HS", ascending=False).iloc[0]["Đội"],
              f"{int(table['HS'].max()):+d}")

    show_df(
        table, hide_index=True, height=min(40 * (len(table) + 1), 800),
        column_config={
            "Điểm": st.column_config.ProgressColumn(
                "Điểm", min_value=0, max_value=int(max(1, table["Điểm"].max())), format="%d"
            )
        },
    )
    top10 = table.head(10).sort_values("Điểm")
    fig = px.bar(top10, x="Điểm", y="Đội", orientation="h", text="Điểm", title="Top 10 theo điểm số",
                 color="Điểm", color_continuous_scale="Purples")
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(style_fig(fig, 420))

# ----------------------------------------------------------
# TAB 3: THỐNG KÊ
# ----------------------------------------------------------
with tab3:
    st.markdown('<div class="section-title">📊 Bức tranh dữ liệu</div>', unsafe_allow_html=True)
    vc = df["Result"].value_counts(normalize=True)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Tổng số trận", f"{len(df):,}")
    m2.metric("Số câu lạc bộ", len(teams))
    m3.metric("Bàn thắng/trận", f"{(df['FTHG'] + df['FTAG']).mean():.2f}")
    m4.metric("Chủ nhà thắng", f"{vc.get('H', 0):.1%}")
    m5.metric("Hòa", f"{vc.get('D', 0):.1%}")

    g1, g2 = st.columns(2)
    with g1:
        dist = df["Result"].map(RESULT_LABEL).value_counts().reset_index()
        dist.columns = ["Kết quả", "Số trận"]
        fig = px.pie(dist, names="Kết quả", values="Số trận", hole=0.55, title="Phân bố kết quả",
                     color="Kết quả",
                     color_discrete_map={"Chủ nhà thắng": RESULT_COLOR["H"], "Hòa": RESULT_COLOR["D"],
                                         "Khách thắng": RESULT_COLOR["A"]})
        st.plotly_chart(style_fig(fig, 340))
    with g2:
        goals = (df["FTHG"] + df["FTAG"]).value_counts().sort_index().reset_index()
        goals.columns = ["Tổng bàn thắng", "Số trận"]
        fig = px.bar(goals, x="Tổng bàn thắng", y="Số trận", title="Phân bố tổng số bàn thắng mỗi trận",
                     color_discrete_sequence=["#6b2fa0"])
        st.plotly_chart(style_fig(fig, 340))

    g3, g4 = st.columns(2)
    with g3:
        att = stats.sort_values("gf", ascending=False).head(10).reset_index().sort_values("gf")
        fig = px.bar(att, x="gf", y="Team", orientation="h", title="Top 10 hàng công (bàn thắng/trận)",
                     labels={"gf": "Bàn thắng/trận", "Team": ""}, color_discrete_sequence=["#00a86b"])
        st.plotly_chart(style_fig(fig, 380))
    with g4:
        dfn = stats.sort_values("ga").head(10).reset_index().sort_values("ga", ascending=False)
        fig = px.bar(dfn, x="ga", y="Team", orientation="h", title="Top 10 hàng thủ (bàn thua/trận ít nhất)",
                     labels={"ga": "Bàn thua/trận", "Team": ""}, color_discrete_sequence=["#2f6bff"])
        st.plotly_chart(style_fig(fig, 380))

    sc = stats.reset_index()
    fig = px.scatter(sc, x="a_pts", y="h_pts", text="Team", title="Lợi thế sân nhà: điểm/trận sân nhà so với sân khách",
                     labels={"a_pts": "Điểm/trận sân khách", "h_pts": "Điểm/trận sân nhà"},
                     color_discrete_sequence=["#e90052"])
    lim = float(max(sc["a_pts"].max(), sc["h_pts"].max())) + 0.2
    fig.add_shape(type="line", x0=0, y0=0, x1=lim, y1=lim, line=dict(dash="dash", color="#8b93a7"))
    fig.update_traces(textposition="top center", marker=dict(size=9))
    st.plotly_chart(style_fig(fig, 480))
    st.caption("Các đội nằm phía trên đường chéo là đội đá sân nhà tốt hơn sân khách.")

# ----------------------------------------------------------
# TAB 4: MÔ HÌNH AI
# ----------------------------------------------------------
with tab4:
    st.markdown('<div class="section-title">🧠 Đánh giá và so sánh mô hình</div>', unsafe_allow_html=True)
    summ = bundle["summary"]
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Trận huấn luyện", bundle["n_train"])
    b2.metric("Trận kiểm tra", bundle["n_test"])
    b3.metric("Mô hình tốt nhất", best["Mô hình"], f"{best['Accuracy']:.1%}")
    b4.metric("Baseline (luôn đoán chủ thắng)", f"{bundle['baseline']:.1%}")

    long = summ.melt(id_vars="Mô hình", var_name="Chỉ số", value_name="Giá trị")
    fig = px.bar(long, x="Mô hình", y="Giá trị", color="Chỉ số", barmode="group", text_auto=".1%",
                 title="So sánh các mô hình trên tập kiểm tra",
                 color_discrete_sequence=["#6b2fa0", "#00c46a"])
    fig.add_hline(y=bundle["baseline"], line_dash="dash", line_color="#e90052",
                  annotation_text="Baseline", annotation_position="top left")
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(style_fig(fig, 380))

    pick = st.selectbox("Xem chi tiết mô hình", MODEL_NAMES, index=MODEL_NAMES.index("Random Forest"))
    d1, d2 = st.columns([1, 1.2])
    with d1:
        cm = bundle["details"][pick]["cm"]
        names = ["Chủ thắng", "Hòa", "Khách thắng"]
        fig = px.imshow(cm, x=names, y=names, text_auto=True, color_continuous_scale="Purples",
                        labels=dict(x="AI dự đoán", y="Thực tế"), title=f"Ma trận nhầm lẫn · {pick}")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style_fig(fig, 360))
    with d2:
        st.markdown("**Báo cáo phân loại**")
        show_df(
            bundle["details"][pick]["report"].style.format(
                {"Precision": "{:.2f}", "Recall": "{:.2f}", "F1": "{:.2f}", "Số trận": "{:.0f}"}
            )
        )
        st.caption("Precision: trong các lần AI đoán lớp đó, bao nhiêu % đúng. Recall: trong các trận thực sự thuộc "
                   "lớp đó, AI bắt được bao nhiêu %.")

    imp = bundle["importance"]
    fig = px.bar(imp, x="Độ quan trọng", y="Đặc trưng", orientation="h",
                 title="Đặc trưng nào ảnh hưởng nhiều nhất? (Random Forest)",
                 color_discrete_sequence=["#6b2fa0"])
    st.plotly_chart(style_fig(fig, 400))

    with st.expander("📘 Phương pháp & lưu ý khi viết báo cáo"):
        st.markdown(
            """
**Đặc trưng (features).** Mỗi trận được mô tả bằng sức mạnh hai đội: điểm/trận, bàn thắng và bàn thua ở sân nhà/sân khách,
chênh lệch điểm/trận và hiệu số. Cách này thay cho việc mã hóa tên đội thành số (LabelEncoder), vì khoảng cách giữa
các mã số đó không mang ý nghĩa bóng đá, nên KNN và các mô hình khác không thể học đúng.

**Đánh giá.** Dữ liệu được chia train/test; thống kê đội chỉ tính trên tập train để tránh "nhìn trộm" tập kiểm tra.
Mô hình dùng để dự đoán trên giao diện được huấn luyện lại trên toàn bộ dữ liệu.

**Giới hạn.** Bóng đá có tính ngẫu nhiên cao nên độ chính xác khoảng 50–55% là mức thường gặp. Hãy so sánh với baseline
để thấy mô hình có thực sự học được gì. Mô hình chưa biết chấn thương, đội hình, lịch thi đấu hay phong độ theo từng ngày.
"""
        )

# ----------------------------------------------------------
# TAB 5: DỮ LIỆU
# ----------------------------------------------------------
with tab5:
    st.markdown('<div class="section-title">🗂️ Khám phá dữ liệu</div>', unsafe_allow_html=True)
    f1c, f2c = st.columns([2, 1])
    sel_teams = f1c.multiselect("Lọc theo đội bóng", teams, placeholder="Chọn một hoặc nhiều đội")
    sel_res = f2c.multiselect("Kết quả", list(RESULT_LABEL.values()), placeholder="Tất cả")

    view = df.copy()
    view["Kết quả"] = view["Result"].map(RESULT_LABEL)
    if sel_teams:
        view = view[view["HomeTeam"].isin(sel_teams) | view["AwayTeam"].isin(sel_teams)]
    if sel_res:
        view = view[view["Kết quả"].isin(sel_res)]
    view = view.drop(columns=["Result"])
    if "Date" in view.columns:
        view["Date"] = view["Date"].dt.strftime("%d/%m/%Y")

    st.caption(f"Hiển thị {len(view):,} / {len(df):,} trận")
    show_df(view, hide_index=True, height=460)
    st.download_button("⬇️ Tải dữ liệu đã lọc (CSV)", view.to_csv(index=False).encode("utf-8-sig"),
                       "du_lieu_bong_da.csv", "text/csv")

    with st.expander("🔎 Kiểm tra chất lượng dữ liệu"):
        q1, q2 = st.columns(2)
        q1.markdown("**Thống kê mô tả**")
        q1.dataframe(df[["FTHG", "FTAG"]].describe().round(2))
        q2.markdown("**Giá trị thiếu theo cột**")
        q2.dataframe(df.isna().sum().rename("Số ô thiếu").to_frame())

st.markdown(
    '<div class="footer">⚽ Premier League AI Predictor · Nhóm 5 · Trường ĐH GTVT TP.HCM · '
    "Xây dựng bằng Streamlit, scikit-learn và Plotly</div>",
    unsafe_allow_html=True,
)
