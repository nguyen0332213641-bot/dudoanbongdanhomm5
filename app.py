import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

# ==========================================
# 1. CẤU HÌNH TRANG & GIAO DIỆN (UI/UX)
# ==========================================
st.set_page_config(page_title="AI Bóng Đá - Nhóm 5", page_icon="⚽", layout="wide")

# Chèn CSS tùy chỉnh để làm đẹp font chữ và nút bấm
st.markdown("""
    <style>
    .main-title { font-size: 42px; color: #153E7E; font-weight: 800; text-align: center; text-transform: uppercase; margin-bottom: 5px;}
    .sub-title { font-size: 20px; color: #E03A3E; text-align: center; font-weight: 600; margin-bottom: 30px;}
    .team-text { font-size: 24px; font-weight: bold; color: #333333; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">⚽ HỆ THỐNG AI DỰ ĐOÁN NGOẠI HẠNG ANH</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Đồ án: Trí tuệ nhân tạo & Ứng dụng - Thực hiện bởi Nhóm 5</div>', unsafe_allow_html=True)

# ==========================================
# 2. XỬ LÝ DỮ LIỆU & HUẤN LUYỆN AI (CACHE)
# ==========================================
# Dùng @st.cache_data để web không phải load lại dữ liệu mỗi lần bấm nút
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

    # Sử dụng KNN với K=3
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)
    return knn, le, all_teams

# Gọi hàm
df = load_data()
knn, le, all_teams = train_model(df)

# ==========================================
# 3. THANH BÊN (SIDEBAR) - THÔNG TIN DỰ ÁN
# ==========================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/f/f2/Premier_League_Logo.svg/280px-Premier_League_Logo.svg.png", width=200)
    st.header("📌 Thông tin đồ án")
    st.info("""
    **Môn học:** Trí tuệ nhân tạo và Ứng dụng\n
    **Nhóm thực hiện:** Nhóm 5\n
    **Thuật toán:** K-Nearest Neighbors (KNN)\n
    **Mục tiêu:** Dự đoán kết quả (Thắng/Hòa/Thua) dựa trên dữ liệu lịch sử thi đấu.
    """)
    st.divider()
    st.markdown("© 2026 - UTH")

# ==========================================
# 4. CHIA TABS GIAO DIỆN CHÍNH
# ==========================================
tab1, tab2 = st.tabs(["🎯 BẢNG ĐIỀU KHIỂN DỰ ĐOÁN", "📊 DỮ LIỆU & THỐNG KÊ"])

with tab1:
    st.markdown("### 🏟️ Lựa chọn trận đấu")
    
    # Chia làm 2 cột cho Đội nhà và Đội khách
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p class="team-text">🏠 ĐỘI CHỦ NHÀ (HOME)</p>', unsafe_allow_html=True)
        home_team = st.selectbox("Chọn đội chủ nhà:", sorted(all_teams), key="home")
    
    with col2:
        st.markdown('<p class="team-text">✈️ ĐỘI KHÁCH (AWAY)</p>', unsafe_allow_html=True)
        away_team = st.selectbox("Chọn đội khách:", sorted(all_teams), key="away")

    st.markdown("<br>", unsafe_allow_html=True) # Khoảng trắng

    # Nút bấm dự đoán
    predict_btn = st.button("🚀 PHÂN TÍCH & DỰ ĐOÁN KẾT QUẢ", use_container_width=True, type="primary")

    if predict_btn:
        if home_team == away_team:
            st.warning("⚠️ Đội nhà và Đội khách không thể giống nhau. Vui lòng chọn lại!")
        else:
            # Tiến hành dự đoán
            input_data = pd.DataFrame({
                'HomeTeam_enc': [le.transform([home_team])[0]],
                'AwayTeam_enc': [le.transform([away_team])[0]]
            })
            
            prediction = knn.predict(input_data)[0]
            probabilities = knn.predict_proba(input_data)[0]
            classes = knn.classes_ # Lấy danh sách nhãn ['A', 'D', 'H']
            
            st.divider()
            st.markdown("### 🏆 KẾT QUẢ TỪ TRÍ TUỆ NHÂN TẠO")
            
            # Cột hiển thị kết quả
            res_col1, res_col2 = st.columns([1, 1])
            
            with res_col1:
                if prediction == 'H':
                    st.success(f"🔥 Dự đoán: **{home_team}** sẽ giành chiến thắng trên sân nhà!")
                    st.balloons() # Hiệu ứng chúc mừng
                elif prediction == 'D':
                    st.info("🤝 Dự đoán: Trận đấu sẽ có kết quả **HÒA**!")
                else:
                    st.error(f"⚔️ Dự đoán: **{away_team}** sẽ tạo bất ngờ và chiến thắng trên sân khách!")
                    
            with res_col2:
                st.markdown("**Độ tự tin của AI (Tỷ lệ %):**")
                # Lấy xác suất từng cửa
                prob_A = probabilities[list(classes).index('A')] * 100
                prob_D = probabilities[list(classes).index('D')] * 100
                prob_H = probabilities[list(classes).index('H')] * 100
                
                st.progress(int(prob_H), text=f"Cửa {home_team} thắng: {prob_H:.0f}%")
                st.progress(int(prob_D), text=f"Cửa Hòa: {prob_D:.0f}%")
                st.progress(int(prob_A), text=f"Cửa {away_team} thắng: {prob_A:.0f}%")

with tab2:
    st.markdown("### 📈 Thống kê tập dữ liệu huấn luyện")
    
    # Tạo 3 thẻ thông tin (Metrics)
    m1, m2, m3 = st.columns(3)
    m1.metric("Tổng số trận đấu (Train)", f"{len(df)} trận")
    m2.metric("Số lượng đội bóng", f"{len(all_teams)} đội")
    m3.metric("Thuật toán", "KNN (K=3)")
    
    st.markdown("#### 📋 Trích xuất dữ liệu gốc")
    # Thanh tìm kiếm dữ liệu
    search_term = st.text_input("🔍 Tìm kiếm đội bóng trong dữ liệu:")
    if search_term:
        filtered_df = df[df['HomeTeam'].str.contains(search_term, case=False) | df['AwayTeam'].str.contains(search_term, case=False)]
        st.dataframe(filtered_df, use_container_width=True)
    else:
        st.dataframe(df, use_container_width=True)
