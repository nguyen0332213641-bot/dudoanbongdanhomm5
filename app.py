import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import plotly.express as px

# =========================================================
# 1. CẤU HÌNH TRANG & CSS GIAO DIỆN SIÊU CẤP (KHÔNG LOGO)
# =========================================================
st.set_page_config(page_title="AI Bóng Đá - Nhóm 5 UTH", page_icon="⚽", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    /* Nền tổng thể tối màu chuẩn Dashboard Analytics */
    .stApp { background-color: #090c15; color: #f8fafc; font-family: 'Segoe UI', Tahoma, sans-serif; }
    
    /* Header Kính mờ cực kỳ sang trọng */
    .glass-header {
        background: rgba(15, 23, 42, 0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        padding: 2.5rem; 
        border-radius: 16px; 
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-left: 6px solid #00f2fe; 
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        text-align: center;
    }
    
    /* Chữ Gradient đổi màu */
    .gradient-title { 
        font-size: 40px; font-weight: 900; 
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 2px;
    }
    .sub-title { font-size: 18px; color: #94a3b8; font-weight: 500; letter-spacing: 1px; }
    
    /* Thẻ Đội Bóng 3D */
    .team-card { 
        background-color: #1e293b; padding: 25px; border-radius: 12px; 
        text-align: center; border: 1px solid #334155; 
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    .home-card { border-top: 5px solid #3b82f6; }
    .away-card { border-top: 5px solid #ef4444; }
    .team-label { font-size: 14px; color: #94a3b8; font-weight: 800; letter-spacing: 2px; margin-bottom: 10px; }
    
    /* Biểu tượng VS Neon rực rỡ */
    .vs-circle {
        background: linear-gradient(135deg, #ff0844 0%, #ffb199 100%);
        border-radius: 50%; width: 70px; height: 70px;
        display: flex; align-items: center; justify-content: center;
        font-weight: 900; font-size: 26px; color: white;
        margin: 20px auto 0 auto;
        box-shadow: 0 4px 15px rgba(255, 8, 68, 0.5);
        border: 4px solid #090c15;
    }
    
    /* Nút bấm hiệu ứng nổi */
    .stButton>button {
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%) !important;
        color: #0f172a !important; font-size: 20px !important; font-weight: 900 !important;
        border-radius: 12px !important; border: none !important; padding: 15px !important;
        transition: all 0.3s ease !important; box-shadow: 0 6px 15px rgba(0, 242, 254, 0.3) !important;
        text-transform: uppercase; letter-spacing: 1px; margin-top: 10px;
    }
    .stButton>button:hover { 
        transform: translateY(-3px) scale(1.02); 
        box-shadow: 0 12px 25px rgba(0, 242, 254, 0.5) !important; 
    }
    
    [data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; border: 1px solid #334155; }
    </style>
""", unsafe_allow_html=True)

# =========================================================
# 2. XỬ LÝ DỮ LIỆU & HUẤN LUYỆN MÔ HÌNH
# =========================================================
@st.cache_data
def load_data():
    df = pd.read_excel('nhom5_bongda_train.xlsx')
    def get_result(row):
        if row['FTHG'] > row['FTAG']: return 'H'
        elif row['FTHG'] == row['FTAG']: return 'D'
        else: return 'A'
    df['Result'] = df.apply(get_result, axis=1)
    df['TotalGoals'] = df['FTHG'] + df['FTAG']
    return df

@st.cache_resource
def train_models(df, k_neighbors):
    le = LabelEncoder()
    all_teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    le.fit(all_teams)
    
    X = pd.DataFrame({
        'HomeTeam_enc': le.transform(df['HomeTeam']),
        'AwayTeam_enc': le.transform(df['AwayTeam'])
    })
    
    knn = KNeighborsClassifier(n_neighbors=k_neighbors, metric='euclidean')
    knn.fit(X, df['Result'])
    
    lr = LinearRegression()
    lr.fit(X, df['TotalGoals'])
    
    return knn, lr, le, all_teams

# Hàm tính toán phong độ (Tỷ lệ thắng lịch sử)
def calculate_win_rate(team, df):
    home_matches = df[df['HomeTeam'] == team]
    away_matches = df[df['AwayTeam'] == team]
    total = len(home_matches) + len(away_matches)
    if total == 0: return 0, 0
    wins = len(home_matches[home_matches['Result'] == 'H']) + len(away_matches[away_matches['Result'] == 'A'])
    return total, (wins / total) * 100

df = load_data()

# =========================================================
# 3. SIDEBAR ĐIỀU KHIỂN 
# =========================================================
with st.sidebar:
    st.markdown("## ⚙️ ĐIỀU KHIỂN MÔ HÌNH")
    k_val = st.slider("Chỉnh số láng giềng KNN (K):", 1, 9, 3, 2, help="Nên chọn số lẻ (1, 3, 5, 7)")
    knn_model, lr_model, encoder, team_list = train_models(df, k_val)
    
    st.markdown("---")
    st.markdown("### 📋 THÔNG TIN ĐỒ ÁN")
    st.markdown("**Môn học:** Trí tuệ nhân tạo & Ứng dụng")
    st.markdown("**Đơn vị:** Trường ĐH GTVT TP.HCM (UTH)")
    st.markdown("**Thực hiện:** Nhóm 05")
    st.caption("✨ Engine Version: 4.0 - E-Sports Edition")

# =========================================================
# 4. GIAO DIỆN CHÍNH
# =========================================================
st.markdown("""
    <div class="glass-header">
        <div class="gradient-title">HỆ THỐNG DỰ BÁO NGOẠI HẠNG ANH</div>
        <div class="sub-title">Cỗ máy AI (K-Nearest Neighbors & Linear Regression) - Đồ án Nhóm 5</div>
    </div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🚀 ĐỘNG CƠ DỰ ĐOÁN (AI ENGINE)", "⚔️ LỊCH SỬ ĐỐI ĐẦU (H2H)", "📊 BÁO CÁO DỮ LIỆU (EDA)"])

# --- TAB 1: DỰ ĐOÁN & GIẢI THÍCH AI ---
with tab1:
    st.write("<br>", unsafe_allow_html=True)
    col1, col_vs, col2 = st.columns([4, 1, 4])
    
    with col1:
        st.markdown('<div class="team-card home-card"><div class="team-label">🏰 ĐỘI CHỦ NHÀ (HOME)</div></div>', unsafe_allow_html=True)
        home_team = st.selectbox(" ", sorted(team_list), key="home", label_visibility="collapsed")
        
        # Thống kê phong độ nhanh
        t_home, wr_home = calculate_win_rate(home_team, df)
        st.caption(f"📊 Đã đá: {t_home} trận | Tỷ lệ thắng: **{wr_home:.1f}%**")
        
    with col_vs:
        st.markdown('<div class="vs-circle">VS</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="team-card away-card"><div class="team-label">🚀 ĐỘI KHÁCH (AWAY)</div></div>', unsafe_allow_html=True)
        away_team = st.selectbox(" ", sorted(team_list), key="away", label_visibility="collapsed")
        
        # Thống kê phong độ nhanh
        t_away, wr_away = calculate_win_rate(away_team, df)
        st.caption(f"📊 Đã đá: {t_away} trận | Tỷ lệ thắng: **{wr_away:.1f}%**")

    st.write("<br>", unsafe_allow_html=True)
    predict_btn = st.button("KÍCH HOẠT HỆ THỐNG TRÍ TUỆ NHÂN TẠO", use_container_width=True)

    if predict_btn:
        if home_team == away_team:
            st.error("🚨 LỖI HỆ THỐNG: Đội nhà và Đội khách không thể giống nhau!")
        else:
            input_features = pd.DataFrame({
                'HomeTeam_enc': [encoder.transform([home_team])[0]],
                'AwayTeam_enc': [encoder.transform([away_team])[0]]
            })
            
            pred_class = knn_model.predict(input_features)[0]
            pred_probs = knn_model.predict_proba(input_features)[0]
            pred_goals = lr_model.predict(input_features)[0]
            distances, indices = knn_model.kneighbors(input_features)
            nearest_matches = df.iloc[indices[0]]
            
            st.divider()
            r_col1, r_col2 = st.columns([1.2, 1])
            
            with r_col1:
                st.markdown("### 🏆 BẢN KẾT LUẬN TỪ MÔ HÌNH")
                if pred_class == 'H':
                    st.success(f"🔥 Dự đoán KNN (Phân loại): **{home_team}** tận dụng lợi thế sân nhà để giành chiến thắng.")
                    st.balloons()
                elif pred_class == 'D':
                    st.info(f"⚖️ Dự đoán KNN (Phân loại): Trận đấu căng thẳng, kết thúc với tỷ số **HÒA**.")
                else:
                    st.error(f"⚔️ Dự đoán KNN (Phân loại): **{away_team}** gây bất ngờ, giành chiến thắng trên sân khách.")
                
                st.warning(f"⚽ Dự đoán Hồi quy (Regression): Trận đấu sẽ có tổng cộng khoảng **{max(0, pred_goals):.1f} bàn thắng**.")
                
            with r_col2:
                st.markdown("### 🎯 BIỂU ĐỒ XÁC SUẤT (PROBABILITY)")
                classes = list(knn_model.classes_)
                prob_H = pred_probs[classes.index('H')] * 100 if 'H' in classes else 0
                prob_D = pred_probs[classes.index('D')] * 100 if 'D' in classes else 0
                prob_A = pred_probs[classes.index('A')] * 100 if 'A' in classes else 0
                
                st.progress(int(prob_H), text=f"🔵 Cửa {home_team} thắng: {prob_H:.1f}%")
                st.progress(int(prob_D), text=f"⚪ Cửa Hòa: {prob_D:.1f}%")
                st.progress(int(prob_A), text=f"🔴 Cửa {away_team} thắng: {prob_A:.1f}%")

            st.divider()
            st.markdown("### 🔍 TẠI SAO AI LẠI DỰ ĐOÁN NHƯ VẬY? (EXPLAINABLE AI)")
            st.caption(f"Dựa trên thuật toán KNN (K={k_val}), hệ thống đã tìm ra **{k_val} trận đấu** có bối cảnh (Home/Away) tương đồng nhất trong lịch sử để làm căn cứ ra quyết định:")
            st.dataframe(nearest_matches[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'Result']], use_container_width=True)

# --- TAB 2: LỊCH SỬ ĐỐI ĐẦU H2H ---
with tab2:
    st.markdown(f"### ⚔️ THỐNG KÊ CHẠM TRÁN: **{home_team}** vs **{away_team}**")
    h2h_data = df[((df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team)) | 
                  ((df['HomeTeam'] == away_team) & (df['AwayTeam'] == home_team))]
    
    if h2h_data.empty:
        st.info("⚠️ Hệ thống chưa ghi nhận trận đối đầu nào giữa hai câu lạc bộ này trong cơ sở dữ liệu.")
    else:
        st.dataframe(h2h_data[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'Result']], use_container_width=True)
        h2h_home_wins = len(h2h_data[h2h_data['Result'] == 'H'])
        h2h_draws = len(h2h_data[h2h_data['Result'] == 'D'])
        h2h_away_wins = len(h2h_data[h2h_data['Result'] == 'A'])
        
        st.markdown("#### 📊 Tóm tắt thành tích đối đầu")
        col_s1, col_s2, col_s3 = st.columns(3)
        col_s1.metric("Chủ nhà thắng", h2h_home_wins)
        col_s2.metric("Hòa", h2h_draws)
        col_s3.metric("Khách thắng", h2h_away_wins)

# --- TAB 3: BIỂU ĐỒ EDA ---
with tab3:
    st.markdown("### 📈 TRỰC QUAN HÓA BỘ DỮ LIỆU")
    c1, c2 = st.columns(2)
    with c1:
        res_counts = df['Result'].value_counts().reset_index()
        res_counts.columns = ['Kết quả', 'Số lượng']
        res_counts['Kết quả'] = res_counts['Kết quả'].map({'H': 'Chủ nhà thắng', 'A': 'Khách thắng', 'D': 'Hòa'})
        fig_pie = px.pie(res_counts, values='Số lượng', names='Kết quả', 
                         title="Tỷ lệ Kết quả Các trận Ngoại Hạng Anh",
                         color_discrete_sequence=['#3b82f6', '#ef4444', '#10b981'], hole=0.5)
        fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color="white", title_x=0.5)
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        home_goals = df.groupby('HomeTeam')['FTHG'].sum().reset_index().sort_values('FTHG', ascending=False).head(5)
        fig_bar = px.bar(home_goals, x='HomeTeam', y='FTHG', 
                         title="Top 5 Hỏa Lực Sân Nhà Mạnh Nhất",
                         labels={'HomeTeam': 'Câu lạc bộ', 'FTHG': 'Tổng số bàn'}, 
                         color='FTHG', color_continuous_scale='Blues')
        fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white", title_x=0.5)
        st.plotly_chart(fig_bar, use_container_width=True)
