import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go

# =========================================================
# 1. CẤU HÌNH TRANG & CSS CHUYÊN NGHIỆP (DARK DASHBOARD)
# =========================================================
st.set_page_config(page_title="AI Bóng Đá - Nhóm 5 UTH", page_icon="⚽", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    
    /* Header Container */
    .header-container {
        background: linear-gradient(90deg, #1f2937, #111827);
        padding: 2rem; border-radius: 12px; margin-bottom: 2rem;
        border-left: 6px solid #3b82f6; box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .main-title { font-size: 38px; font-weight: 900; color: #ffffff; text-transform: uppercase; margin-bottom: 0.5rem; }
    .sub-title { font-size: 18px; color: #9ca3af; font-weight: 500; }

    /* Team Cards */
    .team-card {
        background-color: #1f2937; padding: 20px; border-radius: 10px;
        text-align: center; border: 1px solid #374151; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .home-card { border-top: 4px solid #3b82f6; }
    .away-card { border-top: 4px solid #ef4444; }
    .team-label { font-size: 16px; color: #9ca3af; font-weight: bold; text-transform: uppercase; letter-spacing: 1px; }
    
    .vs-text { font-size: 30px; font-weight: 900; color: #6b7280; text-align: center; padding-top: 40px; }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
        color: white !important; font-size: 18px !important; font-weight: bold !important;
        border-radius: 8px !important; border: none !important; padding: 15px !important;
        transition: all 0.3s ease !important;
    }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 8px 15px rgba(37,99,235,0.4) !important; }
    </style>
""", unsafe_allow_html=True)

# =========================================================
# 2. DATA PIPELINE (XỬ LÝ DỮ LIỆU)
# =========================================================
@st.cache_data
def load_data():
    df = pd.read_excel('nhom5_bongda_train.xlsx')
    
    # Gán nhãn kết quả (H: Home, D: Draw, A: Away)
    def get_result(row):
        if row['FTHG'] > row['FTAG']: return 'H'
        elif row['FTHG'] == row['FTAG']: return 'D'
        else: return 'A'
    df['Result'] = df.apply(get_result, axis=1)
    
    # Thêm cột tổng bàn thắng để chạy Hồi quy tuyến tính
    df['TotalGoals'] = df['FTHG'] + df['FTAG']
    return df

@st.cache_resource
def train_models(df, k_neighbors):
    # 1. Mã hóa dữ liệu (Label Encoding)
    le = LabelEncoder()
    all_teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    le.fit(all_teams)

    X = pd.DataFrame({
        'HomeTeam_enc': le.transform(df['HomeTeam']),
        'AwayTeam_enc': le.transform(df['AwayTeam'])
    })
    
    # 2. Mô hình KNN (Phân loại Kết quả)
    y_knn = df['Result']
    knn = KNeighborsClassifier(n_neighbors=k_neighbors, metric='euclidean')
    knn.fit(X, y_knn)
    
    # 3. Mô hình Hồi quy tuyến tính (Dự đoán Tổng bàn thắng)
    y_lr = df['TotalGoals']
    lr = LinearRegression()
    lr.fit(X, y_lr)
    
    return knn, lr, le, all_teams

try:
    df = load_data()
except Exception as e:
    st.error("Không tìm thấy file dữ liệu 'nhom5_bongda_train.xlsx'. Vui lòng kiểm tra lại GitHub.")
    st.stop()

# =========================================================
# 3. SIDEBAR (BẢNG ĐIỀU KHIỂN & TINH CHỈNH THAM SỐ)
# =========================================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/f/f2/Premier_League_Logo.svg/280px-Premier_League_Logo.svg.png", width=180)
    st.markdown("### ⚙️ HỆ THỐNG ĐIỀU KHIỂN")
    st.markdown("---")
    
    st.markdown("#### 🧠 Tinh chỉnh Siêu tham số AI")
    # Cho phép người dùng chỉnh K của KNN
    k_val = st.slider("Số láng giềng (K) cho KNN:", min_value=1, max_value=9, value=3, step=2, 
                      help="K lẻ giúp mô hình không bị hòa khi bầu chọn.")
    
    # Huấn luyện mô hình với K tùy chỉnh
    knn_model, lr_model, encoder, team_list = train_models(df, k_val)
    
    st.markdown("---")
    st.markdown("#### 📋 Thông tin Đồ án")
    st.markdown("**Môn học:** Trí tuệ nhân tạo & Ứng dụng")
    st.markdown("**Nhóm thực hiện:** Nhóm 05")
    st.markdown("**Trường:** ĐH Giao thông Vận tải TP.HCM")
    st.caption("Version 2.0.0 - Professional Edition")

# =========================================================
# 4. GIAO DIỆN CHÍNH (MAIN DASHBOARD)
# =========================================================
st.markdown("""
    <div class="header-container">
        <div class="main-title">HỆ THỐNG DỰ BÁO BÓNG ĐÁ NGOẠI HẠNG ANH</div>
        <div class="sub-title">Ứng dụng đa mô hình AI (K-Nearest Neighbors & Linear Regression) trong phân tích thể thao</div>
    </div>
""", unsafe_allow_html=True)

# Khởi tạo 3 Tabs
tab_predict, tab_stats, tab_charts = st.tabs(["🚀 ENGINE DỰ ĐOÁN", "📋 PHÂN TÍCH ĐỘI BÓNG", "📊 BIỂU ĐỒ TRỰC QUAN"])

# ---------------------------------------------------------
# TAB 1: DỰ ĐOÁN VỚI AI
# ---------------------------------------------------------
with tab_predict:
    st.markdown("#### 🏟️ THIẾT LẬP TRẬN ĐẤU")
    col1, col_vs, col2 = st.columns([5, 1, 5])
    
    with col1:
        st.markdown('<div class="team-card home-card"><div class="team-label">ĐỘI CHỦ NHÀ (HOME)</div></div>', unsafe_allow_html=True)
        home_team = st.selectbox("", sorted(team_list), key="home", label_visibility="collapsed")
        
    with col_vs:
        st.markdown('<div class="vs-text">VS</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="team-card away-card"><div class="team-label">ĐỘI KHÁCH (AWAY)</div></div>', unsafe_allow_html=True)
        away_team = st.selectbox("", sorted(team_list), key="away", label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("KÍCH HOẠT ĐỘNG CƠ TRÍ TUỆ NHÂN TẠO", use_container_width=True)

    if predict_btn:
        if home_team == away_team:
            st.warning("⚠️ Lỗi logic: Đội nhà và Đội khách phải khác nhau.")
        else:
            # Mã hóa đầu vào
            input_features = pd.DataFrame({
                'HomeTeam_enc': [encoder.transform([home_team])[0]],
                'AwayTeam_enc': [encoder.transform([away_team])[0]]
            })
            
            # Chạy mô hình
            pred_class = knn_model.predict(input_features)[0]
            pred_probs = knn_model.predict_proba(input_features)[0]
            pred_goals = lr_model.predict(input_features)[0] # Hồi quy tính bàn thắng
            
            classes = list(knn_model.classes_)
            prob_H = pred_probs[classes.index('H')] * 100 if 'H' in classes else 0
            prob_D = pred_probs[classes.index('D')] * 100 if 'D' in classes else 0
            prob_A = pred_probs[classes.index('A')] * 100 if 'A' in classes else 0
            
            st.divider()
            st.markdown("#### 🎯 KẾT QUẢ ĐẦU RA TỪ MÔ HÌNH")
            
            r_col1, r_col2 = st.columns(2)
            with r_col1:
                st.markdown("**1. Dự đoán Phân loại (Thuật toán KNN):**")
                if pred_class == 'H':
                    st.success(f"🏆 Nhận định: **{home_team}** giành chiến thắng.")
                elif pred_class == 'D':
                    st.info(f"⚖️ Nhận định: Bất phân thắng bại (**Hòa**).")
                else:
                    st.error(f"🔥 Nhận định: **{away_team}** giành chiến thắng.")
                
                st.markdown("**2. Dự đoán Hồi quy (Thuật toán Linear Regression):**")
                st.warning(f"⚽ Tổng số bàn thắng dự kiến: **{max(0, pred_goals):.1f} bàn**")
                
            with r_col2:
                st.markdown("**Phân phối Xác suất (KNN Probabilities):**")
                st.progress(int(prob_H), text=f"Xác suất {home_team} thắng: {prob_H:.1f}%")
                st.progress(int(prob_D), text=f"Xác suất Hòa: {prob_D:.1f}%")
                st.progress(int(prob_A), text=f"Xác suất {away_team} thắng: {prob_A:.1f}%")

# ---------------------------------------------------------
# TAB 2: THỐNG KÊ & PHÂN TÍCH ĐỘI BÓNG
# ---------------------------------------------------------
with tab_stats:
    st.markdown("#### 🔍 THÔNG SỐ DATASET")
    m1, m2, m3 = st.columns(3)
    m1.metric("Tổng số trận đấu", len(df))
    m2.metric("Số lượng câu lạc bộ", len(team_list))
    m3.metric("Tổng bàn thắng đã ghi", df['TotalGoals'].sum())
    
    st.divider()
    st.markdown("#### 📖 DỮ LIỆU HUẤN LUYỆN GỐC")
    st.dataframe(df[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'Result', 'TotalGoals']], use_container_width=True)

# ---------------------------------------------------------
# TAB 3: BIỂU ĐỒ TRỰC QUAN (EDA - EXPLORATORY DATA ANALYSIS)
# ---------------------------------------------------------
with tab_charts:
    st.markdown("#### 📈 BIỂU ĐỒ PHÂN TÍCH CHUYÊN SÂU")
    
    c1, c2 = st.columns(2)
    with c1:
        # Biểu đồ tròn: Tỷ lệ kết quả trận đấu
        result_counts = df['Result'].value_counts().reset_index()
        result_counts.columns = ['Kết quả', 'Số lượng']
        # Map lại nhãn cho dễ đọc
        result_counts['Kết quả'] = result_counts['Kết quả'].map({'H': 'Đội nhà thắng', 'A': 'Đội khách thắng', 'D': 'Hòa'})
        
        fig_pie = px.pie(result_counts, values='Số lượng', names='Kết quả', 
                         title="Tỷ lệ Kết quả Các trận đấu",
                         color='Kết quả',
                         color_discrete_map={'Đội nhà thắng':'#3b82f6', 'Đội khách thắng':'#ef4444', 'Hòa':'#9ca3af'})
        fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white")
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        # Biểu đồ cột: Top đội ghi bàn sân nhà
        home_goals = df.groupby('HomeTeam')['FTHG'].sum().reset_index()
        home_goals = home_goals.sort_values('FTHG', ascending=False).head(5)
        
        fig_bar = px.bar(home_goals, x='HomeTeam', y='FTHG', 
                         title="Top 5 Đội Ghi Bàn Nhiều Nhất (Sân Nhà)",
                         labels={'HomeTeam': 'Đội bóng', 'FTHG': 'Số bàn thắng'},
                         color='FTHG', color_continuous_scale='Blues')
        fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white")
        st.plotly_chart(fig_bar, use_container_width=True)
