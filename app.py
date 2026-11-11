import streamlit as st
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

# ==========================================
# 1. CẤU HÌNH TRANG
# ==========================================
st.set_page_config(page_title="AI Bóng Đá - Nhóm 5", page_icon="🌈", layout="wide")

# ==========================================
# 2. CSS "ĐỘ" GIAO DIỆN SIÊU MÀU SẮC
# ==========================================
st.markdown("""
    <style>
    /* Tiêu đề dải màu lấp lánh */
    .title-gradient {
        background: linear-gradient(90deg, #ff007f, #ff8c00, #40e0d0, #ba55d3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 45px;
        font-weight: 900;
        text-align: center;
        margin-bottom: 0px;
        padding-bottom: 10px;
    }
    
    /* Phụ đề đỏ nổi bật */
    .subtitle {
        text-align: center;
        font-size: 20px;
        color: #ff4b4b;
        font-weight: 700;
        margin-bottom: 30px;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }

    /* Nút bấm 3D rực rỡ, có hiệu ứng phóng to khi di chuột */
    .stButton>button {
        background: linear-gradient(to right, #ff416c, #ff4b2b) !important;
        color: white !important;
        font-size: 22px !important;
        font-weight: 900 !important;
        border-radius: 12px !important;
        border: none !important;
        box-shadow: 0 4px 15px 0 rgba(255, 75, 43, 0.5) !important;
        transition: all 0.3s ease-in-out !important;
        height: 60px !important;
    }
    .stButton>button:hover {
        transform: scale(1.03) !important;
        box-shadow: 0 6px 20px 0 rgba(255, 75, 43, 0.8) !important;
    }

    /* Bảng màu Đội nhà (Xanh dương) */
    .team-home {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 10px;
        border-radius: 8px 8px 0px 0px;
        text-align: center;
        font-size: 20px;
        font-weight: 800;
        margin-bottom: -15px;
    }
    
    /* Bảng màu Đội khách (Đỏ) */
    .team-away {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        color: white;
        padding: 10px;
        border-radius: 8px 8px 0px 0px;
        text-align: center;
        font-size: 20px;
        font-weight: 800;
        margin-bottom: -15px;
    }

    /* Tô màu viền cho các thông số thống kê */
    div[data-testid="stMetric"] {
        background-color: #f8fbff;
        border-left: 6px solid #ff4b4b;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
    }
    div[data-testid="stMetric"]:nth-child(2) { border-left-color: #40e0d0; }
    div[data-testid="stMetric"]:nth-child(3) { border-left-color: #ba55d3; }
    </style>
""", unsafe_allow_html=True)

# Hiển thị tiêu đề
st.markdown('<div class="title-gradient">⚽ HỆ THỐNG AI DỰ ĐOÁN NGOẠI HẠNG ANH ⚽</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Đồ án: Trí tuệ nhân tạo & Ứng dụng - Nhóm 5</div>', unsafe_allow_html=True)

# ==========================================
# 3. HÀM XỬ LÝ (GIỮ NGUYÊN LOGIC)
# ==========================================
@st.cache_data
def load_data():
    df = pd.read_excel('nhom5_bongda_train.xlsx')
    def get_result(row):
        if row['FTHG'] > row['FTAG']: return 'H'
        elif row['FTHG'] == row['FTAG']: return 'D'
        else: return 'A'
    df['Result'] = df.apply(get_result, axis=1)
    return df

@st.cache_resource
def train_model(df):
    le = LabelEncoder()
    all_teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    le.fit(all_teams)

    X = pd.DataFrame({
        'HomeTeam_enc': le.transform(df['HomeTeam']),
        'AwayTeam_enc': le.transform(df['AwayTeam'])
    })
    y = df['Result']

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)
    return knn, le, all_teams

df = load_data()
knn, le, all_teams = train_model(df)

# ==========================================
# 4. THANH BÊN (SIDEBAR) ĐẦY MÀU SẮC
# ==========================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/f/f2/Premier_League_Logo.svg/280px-Premier_League_Logo.svg.png", width=250)
    st.markdown("## 📌 :rainbow[Thông tin đồ án]")
    st.success("**Môn học:** Trí tuệ nhân tạo và Ứng dụng")
    st.info("**Nhóm thực hiện:** Nhóm 5")
    st.warning("**Thuật toán:** K-Nearest Neighbors (KNN)")
    st.error("**Mục tiêu:** Dự đoán kết quả (Thắng/Hòa/Thua) bằng Học Máy.")
    st.divider()

# ==========================================
# 5. CHIA TAB CHÍNH CỰC ĐẸP
# ==========================================
tab1, tab2 = st.tabs(["🚀 BẢNG ĐIỀU KHIỂN DỰ ĐOÁN", "📊 DỮ LIỆU & THỐNG KÊ"])

with tab1:
    st.markdown("### 🏟️ :rainbow[Thiết lập Trận đấu]")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="team-home">🏠 ĐỘI CHỦ NHÀ (HOME)</div>', unsafe_allow_html=True)
        home_team = st.selectbox("", sorted(all_teams), key="home")
    
    with col2:
        st.markdown('<div class="team-away">✈️ ĐỘI KHÁCH (AWAY)</div>', unsafe_allow_html=True)
        away_team = st.selectbox("", sorted(all_teams), key="away")

    st.markdown("<br>", unsafe_allow_html=True)

    predict_btn = st.button("🔮 PHÂN TÍCH VÀ DỰ ĐOÁN NGAY", use_container_width=True)

    if predict_btn:
        if home_team == away_team:
            st.error("🚨 LỖI: Đội nhà và Đội khách không thể giống nhau. Bạn vui lòng chọn lại nhé!")
        else:
            input_data = pd.DataFrame({
                'HomeTeam_enc': [le.transform([home_team])[0]],
                'AwayTeam_
