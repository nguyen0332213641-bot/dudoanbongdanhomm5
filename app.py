import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION & CUSTOM STYLING (DARK ANALYTICS)
# ---------------------------------------------------------
st.set_page_config(
    page_title="Premier League Predictor - Group 5",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Dashboard CSS
st.markdown("""
    <style>
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
    }
    
    .header-box {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
    .header-title {
        font-size: 30px;
        font-weight: 800;
        color: #f8fafc;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .header-subtitle {
        font-size: 15px;
        color: #94a3b8;
        margin-top: 8px;
        font-weight: 500;
    }

    .team-card-home {
        background-color: #1e293b;
        border-top: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
        font-size: 16px;
        font-weight: 700;
        color: #60a5fa;
        margin-bottom: 10px;
    }
    .team-card-away {
        background-color: #1e293b;
        border-top: 4px solid #ef4444;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
        font-size: 16px;
        font-weight: 700;
        color: #f87171;
        margin-bottom: 10px;
    }
    
    .vs-badge {
        text-align: center;
        font-size: 22px;
        font-weight: 900;
        color: #64748b;
        padding-top: 35px;
    }

    .stButton > button {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        color: #ffffff !important;
        font-size: 16px !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
        border: 1px solid #3b82f6 !important;
        padding: 12px 24px !important;
        transition: all 0.2s ease-in-out !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3) !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%) !important;
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.5) !important;
        transform: translateY(-1px);
    }

    div[data-testid="stMetric"] {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
    }
    div[data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 14px !important;
    }
    div[data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-size: 24px !important;
        font-weight: 700 !important;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. DATA PIPELINE & MODEL ENGINE
# ---------------------------------------------------------
@st.cache_data
def load_dataset():
    df = pd.read_excel('nhom5_bongda_train.xlsx')
    def compute_result(row):
        if row['FTHG'] > row['FTAG']:
            return 'H'
        elif row['FTHG'] == row['FTAG']:
            return 'D'
        else:
            return 'A'
    df['Result'] = df.apply(compute_result, axis=1)
    return df

@st.cache_resource
def initialize_model(df):
    encoder = LabelEncoder()
    teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    encoder.fit(teams)

    X_train = pd.DataFrame({
        'HomeTeam_enc': encoder.transform(df['HomeTeam']),
        'AwayTeam_enc': encoder.transform(df['AwayTeam'])
    })
    y_train = df['Result']

    model = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
    model.fit(X_train, y_train)
    return model, encoder, sorted(teams)

try:
    df_data = load_dataset()
    knn_model, team_encoder, team_list = initialize_model(df_data)
except Exception as e:
    st.error(f"Lỗi khởi tạo dữ liệu: {str(e)}")
    st.stop()

# ---------------------------------------------------------
# 3. SIDEBAR SYSTEM METADATA
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🛠️ Thông tin Hệ thống")
    st.markdown("---")
    st.markdown("**Môn học:** Trí tuệ nhân tạo và Ứng dụng")
    st.markdown("**Đơn vị:** Trường Đại học GTVT TP.HCM (UTH)")
    st.markdown("**Nhóm thực hiện:** Nhóm 05")
    st.markdown("---")
    st.markdown("**Thông số mô hình:**")
    st.markdown("- **Thuật toán:** K-Nearest Neighbors")
    st.markdown("- **K láng giềng:** K = 3")
    st.markdown("- **Khoảng cách:** Euclidean")
    st.markdown("- **Mã hóa:** Label Encoding")
    st.markdown("---")
    st.caption("© 2026 UTH Academic Project")

# ---------------------------------------------------------
# 4. MAIN USER INTERFACE
# ---------------------------------------------------------
st.markdown("""
    <div class="header-box">
        <div class="header-title">HỆ THỐNG DỰ ĐOÁN KẾT QUẢ NGOẠI HẠNG ANH</div>
        <div class="header-subtitle">Mô hình Trí tuệ nhân tạo dự báo kết quả trận đấu | Đồ án Nhóm 5</div>
    </div>
""", unsafe_allow_html=True)

main_tab, data_tab = st.tabs(["📊 Bảng Điều Khiển Dự Đoán", "📁 Phân Tích Tập Dữ Liệu"])

# TAB 1: PREDICTION CONSOLE
with main_tab:
    st.markdown("##### ⚽ Thiết Lập Trận Đấu")
    col_home, col_vs, col_away = st.columns([5, 1, 5])
    
    with col_home:
        st.markdown('<div class="team-card-home">ĐỘI CHỦ NHÀ (HOME)</div>', unsafe_allow_html=True)
        home_selected = st.selectbox("Chọn Đội Nhà", team_list, key="home_select", label_visibility="collapsed")
        
    with col_vs:
        st.markdown('<div class="vs-badge">VS</div>', unsafe_allow_html=True)
        
    with col_away:
        st.markdown('<div class="team-card-away">ĐỘI KHÁCH (AWAY)</div>', unsafe_allow_html=True)
        away_selected = st.selectbox("Chọn Đội Khách", team_list, key="away_select", label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)
    execute_prediction = st.button("CHẠY MÔ HÌNH DỰ ĐOÁN", use_container_width=True)

    if execute_prediction:
        if home_selected == away_selected:
            st.warning("⚠️ Đội nhà và Đội khách phải là hai đội bóng khác nhau.")
        else:
            input_features = pd.DataFrame({
                'HomeTeam_enc': [team_encoder.transform([home_selected])[0]],
                'AwayTeam_enc': [team_encoder.transform([away_selected])[0]]
            })
            
            prediction_class = knn_model.predict(input_features)[0]
            probability = knn_model.predict_proba(input_features)[0]
            class_labels = list(knn_model.classes_)
            
            prob_home = probability[class_labels.index('H')] * 100 if 'H' in class_labels else 0
            prob_draw = probability[class_labels.index('D')] * 100 if 'D' in class_labels else 0
            prob_away = probability[class_labels.index('A')] * 100 if 'A' in class_labels else 0
            
            st.markdown("---")
            st.markdown("##### 📈 Kết Quả Dự Đoán & Xác Suất Mô Hình")
            
            res_left, res_right = st.columns([1, 1])
            
            with res_left:
                if prediction_class == 'H':
                    st.success(f"**Kết quả dự đoán:** Đội nhà thắng — **{home_selected}** giành ưu thế chiến thắng.")
                elif prediction_class == 'D':
                    st.info(f"**Kết quả dự đoán:** Hòa — **{home_selected}** và **{away_selected}** cân bằng lực lượng.")
                else:
                    st.error(f"**Kết quả dự đoán:** Đội khách thắng — **{away_selected}** chiếm ưu thế trên sân khách.")
            
            with res_right:
                st.markdown("**Phân bố xác suất:**")
                st.progress(int(prob_home), text=f"Đội nhà thắng ({home_selected}): {prob_home:.1f}%")
                st.progress(int(prob_draw), text=f"Hòa: {prob_draw:.1f}%")
                st.progress(int(prob_away), text=f"Đội khách thắng ({away_selected}): {prob_away:.1f}%")

# TAB 2: DATASET ANALYSIS
with data_tab:
    st.markdown("##### 📊 Tổng Quan Dữ Liệu Huấn Luyện")
    m1, m2, m3 = st.columns(3)
    m1.metric("Tổng Số Trận Đấu (Train)", f"{len(df_data)} trận")
    m2.metric("Số Lượng Đội Bóng", f"{len(team_list)} đội")
    m3.metric("Thuật Toán Sử Dụng", "KNN (K=3)")
    
    st.markdown("---")
    st.markdown("##### 📋 Truy Vấn Dữ Liệu Gốc")
    search_query = st.text_input("Lọc dữ liệu theo tên đội bóng:")
    if search_query:
        filtered_records = df_data[
            df_data['HomeTeam'].str.contains(search_query, case=False) | 
            df_data['AwayTeam'].str.contains(search_query, case=False)
        ]
        st.dataframe(filtered_records, use_container_width=True)
    else:
        st.dataframe(df_data, use_container_width=True)
