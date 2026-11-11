import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import plotly.express as px

# =========================================================
# 1. CẤU HÌNH TRANG & CSS DARK MODE CHUYÊN NGHIỆP
# =========================================================
st.set_page_config(page_title="AI Bóng Đá - Nhóm 5 UTH", page_icon="⚽", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    .header-container {
        background: linear-gradient(90deg, #1e293b, #0f172a);
        padding: 2rem; border-radius: 12px; margin-bottom: 2rem;
        border-left: 6px solid #10b981; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.5);
    }
    .main-title { font-size: 38px; font-weight: 900; color: #ffffff; margin-bottom: 0.5rem; text-transform: uppercase;}
    .sub-title { font-size: 18px; color: #9ca3af; font-weight: 500; }
    
    .team-card { background-color: #1f2937; padding: 20px; border-radius: 10px; text-align: center; border: 1px solid #374151; }
    .home-card { border-top: 4px solid #3b82f6; }
    .away-card { border-top: 4px solid #ef4444; }
    .team-label { font-size: 14px; color: #9ca3af; font-weight: bold; letter-spacing: 1px; margin-bottom: 5px; }
    .team-name { font-size: 24px; color: white; font-weight: 900; }
    
    .vs-text { font-size: 28px; font-weight: 900; color: #4b5563; text-align: center; padding-top: 35px; }
    
    .stButton>button {
        background: linear-gradient(135deg, #10b981, #059669) !important;
        color: white !important; font-size: 18px !important; font-weight: 900 !important;
        border-radius: 8px !important; border: none !important; padding: 15px !important;
        transition: all 0.3s ease !important; box-shadow: 0 4px 6px rgba(16, 185, 129, 0.2) !important;
    }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 8px 15px rgba(16, 185, 129, 0.4) !important; }
    
    /* Làm đẹp bảng dữ liệu Streamlit */
    [data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
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

df = load_data()

# =========================================================
# 3. SIDEBAR ĐIỀU KHIỂN
# =========================================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/f/f2/Premier_League_Logo.svg/280px-Premier_League_Logo.svg.png", width=180)
    st.markdown("### ⚙️ CÀI ĐẶT MÔ HÌNH")
    k_val = st.slider("Chỉnh số láng giềng (K) cho thuật toán KNN:", 1, 9, 3, 2, help="Nên chọn số lẻ")
    knn_model, lr_model, encoder, team_list = train_models(df, k_val)
    
    st.markdown("---")
    st.markdown("### 📋 THÔNG TIN ĐỒ ÁN")
    st.markdown("**Môn học:** Trí tuệ nhân tạo & Ứng dụng")
    st.markdown("**Trường:** ĐH GTVT TP.HCM (UTH)")
    st.markdown("**Thực hiện:** Nhóm 05")
    st.caption("Phiên bản PRO - Tích hợp Explainable AI")

# =========================================================
# 4. GIAO DIỆN CHÍNH
# =========================================================
st.markdown("""
    <div class="header-container">
        <div class="main-title">HỆ THỐNG PHÂN TÍCH & DỰ BÁO NGOẠI HẠNG ANH</div>
        <div class="sub-title">🤖 Tích hợp K-Nearest Neighbors & Hồi quy tuyến tính đa biến</div>
    </div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🚀 ENGINE DỰ ĐOÁN TƯƠNG TÁC", "⚔️ LỊCH SỬ ĐỐI ĐẦU (H2H)", "📊 BIỂU ĐỒ TRỰC QUAN"])

# --- TAB 1: DỰ ĐOÁN & GIẢI THÍCH AI ---
with tab1:
    col1, col_vs, col2 = st.columns([4, 1, 4])
    with col1:
        st.markdown('<div class="team-card home-card"><div class="team-label">🏰 CHỦ NHÀ (HOME)</div></div>', unsafe_allow_html=True)
        home_team = st.selectbox("", sorted(team_list), key="home", label_visibility="collapsed")
    with col_vs:
        st.markdown('<div class="vs-text">VS</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="team-card away-card"><div class="team-label">🚀 ĐỘI KHÁCH (AWAY)</div></div>', unsafe_allow_html=True)
        away_team = st.selectbox("", sorted(team_list), key="away", label_visibility="collapsed")

    st.write("")
    predict_btn = st.button("KÍCH HOẠT HỆ THỐNG TRÍ TUỆ NHÂN TẠO", use_container_width=True)

    if predict_btn:
        if home_team == away_team:
            st.warning("⚠️ Lỗi: Đội nhà và đội khách không thể giống nhau.")
        else:
            input_features = pd.DataFrame({
                'HomeTeam_enc': [encoder.transform([home_team])[0]],
                'AwayTeam_enc': [encoder.transform([away_team])[0]]
            })
            
            # Tính toán kết quả
            pred_class = knn_model.predict(input_features)[0]
            pred_probs = knn_model.predict_proba(input_features)[0]
            pred_goals = lr_model.predict(input_features)[0]
            
            # Tìm K láng giềng gần nhất để giải thích AI
            distances, indices = knn_model.kneighbors(input_features)
            nearest_matches = df.iloc[indices[0]]
            
            st.divider()
            r_col1, r_col2 = st.columns([1.2, 1])
            
            with r_col1:
                st.markdown("### 🏆 KẾT QUẢ ĐẦU RA")
                if pred_class == 'H':
                    st.success(f"🔥 Dự đoán KNN: **{home_team}** giành chiến thắng.")
                    st.balloons()
                elif pred_class == 'D':
                    st.info(f"⚖️ Dự đoán KNN: Trận đấu kết thúc với tỷ số **HÒA**.")
                else:
                    st.error(f"⚔️ Dự đoán KNN: **{away_team}** giành chiến thắng trên sân khách.")
                
                st.warning(f"⚽ Dự đoán Hồi quy (Linear Regression): Trận đấu sẽ có khoảng **{max(0, pred_goals):.1f} bàn thắng**.")
                
            with r_col2:
                st.markdown("### 🎯 XÁC SUẤT BẦU CHỌN")
                classes = list(knn_model.classes_)
                prob_H = pred_probs[classes.index('H')] * 100 if 'H' in classes else 0
                prob_D = pred_probs[classes.index('D')] * 100 if 'D' in classes else 0
                prob_A = pred_probs[classes.index('A')] * 100 if 'A' in classes else 0
                
                st.progress(int(prob_H), text=f"Xác suất {home_team} thắng: {prob_H:.0f}%")
                st.progress(int(prob_D), text=f"Xác suất Hòa: {prob_D:.0f}%")
                st.progress(int(prob_A), text=f"Xác suất {away_team} thắng: {prob_A:.0f}%")

            # Phần "ăn tiền": Giải thích AI
            st.divider()
            st.markdown("### 🔍 GIẢI THÍCH QUYẾT ĐỊNH CỦA AI (EXPLAINABLE AI)")
            st.caption(f"Dựa trên thuật toán KNN (K={k_val}), máy tính đã tìm ra {k_val} trận đấu trong quá khứ có điều kiện bối cảnh giống với trận đấu này nhất để đưa ra quyết định cuối cùng:")
            st.dataframe(nearest_matches[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'Result']], use_container_width=True)

# --- TAB 2: LỊCH SỬ ĐỐI ĐẦU H2H ---
with tab2:
    st.markdown(f"#### ⚔️ LỊCH SỬ ĐỐI ĐẦU: {home_team} vs {away_team}")
    h2h_data = df[((df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team)) | 
                  ((df['HomeTeam'] == away_team) & (df['AwayTeam'] == home_team))]
    
    if h2h_data.empty:
        st.info("Chưa có lịch sử đối đầu nào giữa hai đội bóng này trong tập dữ liệu huấn luyện.")
    else:
        st.dataframe(h2h_data[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'Result']], use_container_width=True)
        st.caption(f"Tìm thấy tổng cộng {len(h2h_data)} trận đối đầu.")

# --- TAB 3: BIỂU ĐỒ EDA ---
with tab3:
    c1, c2 = st.columns(2)
    with c1:
        res_counts = df['Result'].value_counts().reset_index()
        res_counts.columns = ['Kết quả', 'Số lượng']
        res_counts['Kết quả'] = res_counts['Kết quả'].map({'H': 'Chủ nhà thắng', 'A': 'Khách thắng', 'D': 'Hòa'})
        fig_pie = px.pie(res_counts, values='Số lượng', names='Kết quả', title="Tỷ lệ kết quả bóng đá Anh",
                         color_discrete_sequence=['#3b82f6', '#ef4444', '#10b981'])
        fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color="white")
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        home_goals = df.groupby('HomeTeam')['FTHG'].sum().reset_index().sort_values('FTHG', ascending=False).head(5)
        fig_bar = px.bar(home_goals, x='HomeTeam', y='FTHG', title="Top 5 Hỏa lực Sân nhà",
                         labels={'HomeTeam': 'Câu lạc bộ', 'FTHG': 'Số bàn'}, color='FTHG', color_continuous_scale='Mint')
        fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white")
        st.plotly_chart(fig_bar, use_container_width=True)
