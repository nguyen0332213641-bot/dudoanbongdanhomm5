import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import os

# =========================================================
# 1. CẤU HÌNH TRANG & CSS VIP (GLASSMORPHISM & NEON)
# =========================================================
st.set_page_config(page_title="AI Bóng Đá - Nhóm 5 UTH", page_icon="⚽", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    /* Nền tổng thể */
    .stApp { background-color: #0b0f19; color: #FAFAFA; }
    
    /* Hiệu ứng Kính mờ (Glassmorphism) cho Header */
    .header-container {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        padding: 2rem; border-radius: 16px; margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-left: 6px solid #10b981;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        display: flex; align-items: center; gap: 20px;
    }
    
    .main-title { font-size: 36px; font-weight: 900; color: #ffffff; margin-bottom: 0.2rem; text-transform: uppercase; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);}
    .sub-title { font-size: 18px; color: #10b981; font-weight: 600; letter-spacing: 1px;}
    
    /* Thẻ Đội bóng */
    .team-card { 
        background: rgba(31, 41, 55, 0.8); padding: 20px; border-radius: 12px; 
        text-align: center; border: 1px solid rgba(255,255,255,0.05); 
        box-shadow: 0 4px 15px rgba(0,0,```python
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import base64

# =========================================================
# 0. HÀM CHUYỂN ĐỔI ẢNH LOGO (Base64)
# =========================================================
# (Tùy chọn) Chuyển logo trường thành base64 để nhúng thẳng vào HTML
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Nếu bạn không có file ảnh logo UTH cục bộ thì có thể dùng đường link trực tiếp
uth_logo_url = "[https://ut.edu.vn/app/webroot/upload/image/logo-uth.png](https://ut.edu.vn/app/webroot/upload/image/logo-uth.png)" # Hoặc link ảnh của bạn

# =========================================================
# 1. CẤU HÌNH TRANG & CSS DARK MODE CHUYÊN NGHIỆP + GIAO DIỆN NÂNG CẤP
# =========================================================
st.set_page_config(page_title="AI Bóng Đá - Nhóm 5 UTH", page_icon="⚽", layout="wide", initial_sidebar_state="expanded")

st.markdown(f"""
    <style>
    /* Tổng thể nền và màu chữ */
    .stApp {{ background-color: #0b0f19; color: #f8fafc; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }}
    
    /* Header Container - Lấp lánh và bo tròn mượt mà hơn */
    .header-container {{
        background: linear-gradient(135deg, #1e293b, #0f172a);
        padding: 2.5rem; 
        border-radius: 16px; 
        margin-bottom: 2rem;
        border-left: 6px solid #10b981; 
        box-shadow: 0 15px 25px -5px rgba(0,0,0,0.6);
        position: relative; /* Để chứa logo góc */
    }}
    
    /* Tiêu đề chính */
    .main-title {{ 
        font-size: 42px; 
        font-weight: 900; 
        background: -webkit-linear-gradient(45deg, #10b981, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem; 
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    
    /* Phụ đề */
    .sub-title {{ font-size: 18px; color: #cbd5e1; font-weight: 500; font-style: italic; }}
    
    /* Cấu hình Logo góc trái Header (Nếu muốn chèn trực tiếp) */
    .logo-container {{
        position: absolute;
        top: 20px;
        right: 30px;
        opacity: 0.8;
    }}
    
    /* Thẻ Đội Bóng (Team Cards) - 3D Effect */
    .team-card {{ 
        background-color: #1e293b; 
        padding: 25px; 
        border-radius: 12px; 
        text-align: center; 
        border: 1px solid #334155; 
        transition: transform 0.3s ease;
    }}
    .team-card:hover {{
        transform: translateY(-5px);
    }}
    
    .home-card {{ border-top: 5px solid #3b82f6; box-shadow: 0 10px 15px rgba(59, 130, 246, 0.2); }}
    .away-card {{ border-top: 5px solid #ef4444; box-shadow: 0 10px 15px rgba(239, 68, 68, 0.2); }}
    
    .team-label {{ font-size: 15px; color: #94a3b8; font-weight: bold; letter-spacing: 2px; margin-bottom: 10px; }}
    .team-name {{ font-size: 26px; color: white; font-weight: 900; }}
    
    /* Chữ VS */
    .vs-text {{ font-size: 32px; font-weight: 900; color: #475569; text-align: center; padding-top: 45px; text-shadow: 2px 2px 4px rgba(0,0,0,0.5); }}
    
    /* Nút bấm (Button) - Glow Effect */
    .stButton>button {{
        background: linear-gradient(135deg, #10b981, #059669) !important;
        color: white !important; font-size: 20px !important; font-weight: 900 !important;
        border-radius: 10px !important; border: none !important; padding: 18px !important;
        transition: all 0.3s ease !important; 
        box-shadow: 0 6px 15px rgba(16, 185, 129, 0.3) !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    .stButton>button:hover {{ 
        transform: translateY(-3px) scale(1.02); 
        box-shadow: 0 12px 20px rgba(16, 185, 129, 0.5) !important; 
        background: linear-gradient(135deg, #059669, #047857) !important;
    }}
    
    /* Làm đẹp bảng dữ liệu Streamlit */
    [data-testid="stDataFrame"] {{ 
        border-radius: 12px; 
        overflow: hidden; 
        box-shadow: 0 8px 16px rgba(0,0,0,0.2); 
        border: 1px solid #334155;
    }}
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}
    .stTabs [data-baseweb="tab"] {{
        height: 50px;
        border-radius: 8px 8px 0px 0px;
        background-color: #1e293b;
        padding: 0 20px;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: #3b82f6 !important;
        color: white !important;
    }}
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
# 3. SIDEBAR ĐIỀU KHIỂN & CHÈN LOGO
# =========================================================
with st.sidebar:
    # Chèn Logo Trường Đại học GTVT TP.HCM (UTH)
    # Bạn có thể dùng link url hoặc đổi thành st.image("path_to_your_local_uth_logo.png")
    st.image(uth_logo_url, width=150) 
    
    st.markdown("---")
    st.markdown("### ⚙️ CÀI ĐẶT MÔ HÌNH")
    k_val = st.slider("Chỉnh số láng giềng (K) cho thuật toán KNN:", 1, 9, 3, 2, help="Nên chọn số lẻ (1, 3, 5, 7) để tránh trường hợp hòa khi vote.")
    knn_model, lr_model, encoder, team_list = train_models(df, k_val)
    
    st.markdown("---")
    st.markdown("### 📋 THÔNG TIN ĐỒ ÁN")
    st.info("**Môn học:** Trí tuệ nhân tạo & Ứng dụng")
    st.success("**Trường:** ĐH GTVT TP.HCM (UTH)")
    st.warning("**Thực hiện:** Nhóm 05")
    st.caption("✨ Phiên bản VIP - Nâng cấp Giao diện & Tính năng")

# =========================================================
# 4. GIAO DIỆN CHÍNH (MAIN HEADER)
# =========================================================
# Có thể chèn logo Premier League ở góc phải của header
pl_logo = "[https://upload.wikimedia.org/wikipedia/en/thumb/f/f2/Premier_League_Logo.svg/120px-Premier_League_Logo.svg.png](https://upload.wikimedia.org/wikipedia/en/thumb/f/f2/Premier_League_Logo.svg/120px-Premier_League_Logo.svg.png)"

st.markdown(f"""
    <div class="header-container">
        <div class="main-title">HỆ THỐNG PHÂN TÍCH & DỰ BÁO NGOẠI HẠNG ANH</div>
        <div class="sub-title">🤖 Ứng dụng Học máy: K-Nearest Neighbors & Hồi quy tuyến tính đa biến</div>
        <div class="logo-container">
            <img src="{pl_logo}" width="80" alt="PL Logo">
        </div>
    </div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🚀 DỰ ĐOÁN VỚI AI (PREDICT)", "⚔️ LỊCH SỬ ĐỐI ĐẦU (H2H)", "📊 BÁO CÁO PHÂN TÍCH (EDA)"])

# --- TAB 1: DỰ ĐOÁN & GIẢI THÍCH AI ---
with tab1:
    st.write("<br>", unsafe_allow_html=True)
    col1, col_vs, col2 = st.columns([4, 1, 4])
    with col1:
        st.markdown('<div class="team-card home-card"><div class="team-label">🏰 CHỦ NHÀ (HOME)</div></div>', unsafe_allow_html=True)
        home_team = st.selectbox("", sorted(team_list), key="home", label_visibility="collapsed")
    with col_vs:
        st.markdown('<div class="vs-text">VS</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="team-card away-card"><div class="team-label">🚀 ĐỘI KHÁCH (AWAY)</div></div>', unsafe_allow_html=True)
        away_team = st.selectbox("", sorted(team_list), key="away", label_visibility="collapsed")

    st.write("<br>", unsafe_allow_html=True)
    predict_btn = st.button("KÍCH HOẠT HỆ THỐNG TRÍ TUỆ NHÂN TẠO", use_container_width=True)

    if predict_btn:
        if home_team == away_team:
            st.error("🚨 Lỗi Nhập liệu: Đội nhà và Đội khách phải là hai đội bóng khác nhau.")
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
                st.markdown("### 🏆 BẢN KẾT LUẬN TỪ MÔ HÌNH")
                if pred_class == 'H':
                    st.success(f"🔥 Dự đoán KNN (Phân loại): **{home_team}** tận dụng lợi thế sân nhà để giành chiến thắng.")
                    st.balloons()
                elif pred_class == 'D':
                    st.info(f"⚖️ Dự đoán KNN (Phân loại): Trận đấu căng thẳng, kết thúc với tỷ số **HÒA**.")
                else:
                    st.error(f"⚔️ Dự đoán KNN (Phân loại): **{away_team}** gây bất ngờ, giành chiến thắng trên sân khách.")
                
                st.warning(f"⚽ Dự đoán Hồi quy (Regresson): Trận đấu hứa hẹn sẽ có tổng cộng khoảng **{max(0, pred_goals):.1f} bàn thắng**.")
                
            with r_col2:
                st.markdown("### 🎯 BIỂU ĐỒ XÁC SUẤT BẦU CHỌN")
                classes = list(knn_model.classes_)
                prob_H = pred_probs[classes.index('H')] * 100 if 'H' in classes else 0
                prob_D = pred_probs[classes.index('D')] * 100 if 'D' in classes else 0
                prob_A = pred_probs[classes.index('A')] * 100 if 'A' in classes else 0
                
                st.progress(int(prob_H), text=f"Cửa {home_team} thắng (Home): {prob_H:.1f}%")
                st.progress(int(prob_D), text=f"Cửa Hòa (Draw): {prob_D:.1f}%")
                st.progress(int(prob_A), text=f"Cửa {away_team} thắng (Away): {prob_A:.1f}%")

            # Phần "ăn tiền": Giải thích AI
            st.divider()
            st.markdown("### 🔍 TẠI SAO AI LẠI DỰ ĐOÁN NHƯ VẬY? (EXPLAINABLE AI)")
            st.caption(f"Dựa trên thuật toán K-Nearest Neighbors với độ đo Euclidean (K={k_val}), hệ thống đã rà soát bộ nhớ và tìm ra **{k_val} trận đấu lịch sử** có 'Bối cảnh' (Đội nhà/Đội khách) tương đồng nhất để làm căn cứ ra quyết định:")
            st.dataframe(nearest_matches[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'Result']], use_container_width=True)

# --- TAB 2: LỊCH SỬ ĐỐI ĐẦU H2H ---
with tab2:
    st.markdown(f"### ⚔️ THỐNG KÊ LỊCH SỬ CHẠM TRÁN (H2H): **{home_team}** vs **{away_team}**")
    st.write("Dữ liệu được trích xuất trực tiếp từ tập dữ liệu huấn luyện.")
    h2h_data = df[((df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team)) | 
                  ((df['HomeTeam'] == away_team) & (df['AwayTeam'] == home_team))]
    
    if h2h_data.empty:
        st.warning("⚠️ Hệ thống chưa ghi nhận trận đối đầu nào giữa hai câu lạc bộ này trong cơ sở dữ liệu.")
    else:
        st.dataframe(h2h_data[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'Result']], use_container_width=True)
        
        # Thống kê nhanh H2H
        st.markdown("#### 📊 Tóm tắt thành tích đối đầu")
        h2h_home_wins = len(h2h_data[h2h_data['Result'] == 'H'])
        h2h_draws = len(h2h_data[h2h_data['Result'] == 'D'])
        h2h_away_wins = len(h2h_data[h2h_data['Result'] == 'A'])
        
        col_s1, col_s2, col_s3 = st.columns(3)
        col_s1.metric("Chủ nhà thắng", h2h_home_wins)
        col_s2.metric("Hòa", h2h_draws)
        col_s3.metric("Khách thắng", h2h_away_wins)

# --- TAB 3: BIỂU ĐỒ EDA ---
with tab3:
    st.markdown("### 📈 TRỰC QUAN HÓA BỘ DỮ LIỆU BÓNG ĐÁ")
    st.write("Báo cáo phân tích dữ liệu khám phá (Exploratory Data Analysis - EDA)")
    
    c1, c2 = st.columns(2)
    with c1:
        res_counts = df['Result'].value_counts().reset_index()
        res_counts.columns = ['Kết quả', 'Số lượng']
        res_counts['Kết quả'] = res_counts['Kết quả'].map({'H': 'Chủ nhà thắng', 'A': 'Khách thắng', 'D': 'Hòa'})
        fig_pie = px.pie(res_counts, values='Số lượng', names='Kết quả', 
                         title="Tỷ lệ Kết quả Các trận Ngoại Hạng Anh",
                         color_discrete_sequence=['#3b82f6', '#ef4444', '#10b981'],
                         hole=0.4) # Donut chart cho đẹp
        fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color="white", title_x=0.5)
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        home_goals = df.groupby('HomeTeam')['FTHG'].sum().reset_index().sort_values('FTHG', ascending=False).head(5)
        fig_bar = px.bar(home_goals, x='HomeTeam', y='FTHG', 
                         title="Top 5 Đội Bóng Ghi Nhiều Bàn Nhất (Sân Nhà)",
                         labels={'HomeTeam': 'Câu lạc bộ', 'FTHG': 'Tổng số bàn'}, 
                         color='FTHG', color_continuous_scale='Mint')
        fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white", title_x=0.5)
        st.plotly_chart(fig_bar, use_container_width=True)
