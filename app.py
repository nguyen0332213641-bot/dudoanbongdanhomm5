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
                'AwayTeam_Để làm cho trang web của bạn trở nên thực sự nổi bật, mang đậm chất thể thao, nhiều màu sắc rực rỡ và hiện đại hơn, mình đã viết thêm một bộ **CSS tùy chỉnh (Gradient, hiệu ứng nổi, nút bấm 3D)** nhúng trực tiếp vào mã nguồn. 

Giao diện mới sẽ có:
*   🌈 **Tiêu đề màu Gradient** (chuyển màu mượt mà từ đỏ sang cam).
*   🕹️ **Nút bấm "Dự đoán" cực bắt mắt** với màu xanh neon và hiệu ứng phình to (zoom) khi di chuột vào.
*   🎴 **Thẻ Đội bóng (Team Cards)** có màu nền pastel rực rỡ và đổ bóng (Shadow) hiện đại.
*   🎉 **Màu sắc kết quả rõ ràng:** Xanh lá cho chiến thắng, Vàng cho hòa và Đỏ cho thất bại, kết hợp hiệu ứng pháo hoa/bóng bay.

Bạn hãy copy toàn bộ đoạn mã này và **dán đè lên file `app.py`** trên GitHub nhé:

```python
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

# ==========================================
# 1. CẤU HÌNH TRANG & GIAO DIỆN (UI/UX)
# ==========================================
st.set_page_config(page_title="AI Bóng Đá - Nhóm 5", page_icon="⚽", layout="wide")

# MÃ CSS TÙY CHỈNH GIÚP WEB NHIỀU MÀU SẮC VÀ ẢO DIỆU HƠN
st.markdown("""
    <style>
    /* Nền toàn trang hơi xám nhẹ để làm nổi bật các khối màu */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Tiêu đề chính chuyển màu Gradient rực rỡ */
    .main-title {
        font-size: 50px;
        font-weight: 900;
        background: -webkit-linear-gradient(45deg, #FF416C, #FF4B2B);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 5px;
        text-transform: uppercase;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .sub-title {
        font-size: 22px;
        color: #4A00E0;
        text-align: center;
        font-weight: 700;
        margin-bottom: 30px;
    }

    /* Khối thẻ Đội bóng với màu nền Gradient pastel */
    .team-card-home {
        background: linear-gradient(135deg, #74ebd5 0%, #ACB6E5 100%);
        padding: 15px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border: 2px solid #ffffff;
    }
    .team-card-away {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        padding: 15px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border: 2px solid #ffffff;
    }
    
    .team-text { font-size: 22px; font-weight: 900; color: #333333; margin: 0; }

    /* Nút bấm Dự đoán 3D cực đẹp */
    .stButton > button {
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        color: #004d40;
        font-size: 22px !important;
        font-weight: 900 !important;
        border-radius: 50px;
        border: none;
        padding: 10px 24px;
        transition: all 0.3s ease;
        box-shadow: 0px 8px 15px rgba(0, 0, 0, 0.1);
    }
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0px 15px 20px rgba(46, 229, 157, 0.4);
        color: #000000;
    }
    
    /* Chữ VS ở giữa */
    .vs-text {
        font-size: 40px;
        font-weight: 900;
        color: #ffaa00;
        text-align: center;
        margin-top: 30px;
        text-shadow: 2px 2px #333;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">⚽ DỰ ĐOÁN NGOẠI HẠNG ANH BẰNG AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">🤖 Đồ án: Trí tuệ nhân tạo - Thực hiện bởi Nhóm 5</div>', unsafe_allow_html=True)

# ==========================================
# 2. XỬ LÝ DỮ LIỆU & HUẤN LUYỆN AI (CACHE)
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
# 3. THANH BÊN (SIDEBAR) ĐẦY MÀU SẮC
# ==========================================
with st.sidebar:
    st.image("[https://upload.wikimedia.org/wikipedia/en/thumb/f/f2/Premier_League_Logo.svg/280px-Premier_League_Logo.svg.png](https://upload.wikimedia.org/wikipedia/en/thumb/f/f2/Premier_League_Logo.svg/280px-Premier_League_Logo.svg.png)", width=200)
    st.markdown("## ⚙️ BẢNG ĐIỀU KHIỂN")
    st.info("💡 **Gợi ý:** Hãy chọn 2 đội bóng khác nhau để xem AI của Nhóm 5 đưa ra nhận định nhé!")
    st.success("🧠 **Thuật toán:** K-Nearest Neighbors (KNN)")
    st.warning("🗂️ **Dữ liệu:** Lịch sử bóng đá Anh")
    st.divider()
    st.markdown("👩‍💻 **Nhóm 5 - Trường Đại học GTVT TP.HCM**")

# ==========================================
# 4. CHIA TABS GIAO DIỆN CHÍNH
# ==========================================
tab1, tab2 = st.tabs(["🎮 THỰC HIỆN DỰ ĐOÁN", "📊 KHÁM PHÁ DỮ LIỆU"])

with tab1:
    st.write("---")
    # Chia làm 3 cột (Đội nhà - VS - Đội khách)
    col1, col_vs, col2 = st.columns([4, 1, 4])
    
    with col1:
        st.markdown('<div class="team-card-home"><p class="team-text">🏰 ĐỘI CHỦ NHÀ (HOME)</p></div>', unsafe_allow_html=True)
        home_team = st.selectbox("", sorted(all_teams), key="home", label_visibility="collapsed")
    
    with col_vs:
        st.markdown('<div class="vs-text">VS</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="team-card-away"><p class="team-text">🚀 ĐỘI KHÁCH (AWAY)</p></div>', unsafe_allow_html=True)
        away_team = st.selectbox("", sorted(all_teams), key="away", label_visibility="collapsed")

    st.markdown("<br><br>", unsafe_allow_html=True)

    # Nút bấm dự đoán
    predict_btn = st.button("⚡ KÍCH HOẠT AI DỰ ĐOÁN KẾT QUẢ ⚡", use_container_width=True)

    if predict_btn:
        if home_team == away_team:
            st.error("🚨 Ối! Đội nhà và Đội khách không thể là cùng một đội. Vui lòng chọn lại nha!")
        else:
            # Tiến hành dự đoán
            input_data = pd.DataFrame({
                'HomeTeam_enc': [le.transform([home_team])[0]],
                'AwayTeam_enc': [le.transform([away_team])[0]]
            })
            
            prediction = knn.predict(input_data)[0]
            probabilities = knn.predict_proba(input_data)[0]
            classes = knn.classes_ 
            
            st.write("---")
            st.markdown("### 🏆 AI ĐÃ ĐƯA RA KẾT LUẬN:")
            
            res_col1, res_col2 = st.columns([1.2, 1])
            
            with res_col1:
                if prediction == 'H':
                    st.success(f"🎉 **CHIẾN THẮNG:** **{home_team}** sẽ tận dụng lợi thế sân nhà để giành 3 điểm!")
                    st.balloons() 
                elif prediction == 'D':
                    st.info("⚖️ **BẤT PHÂN THẮNG BẠI:** Trận đấu nhiều khả năng sẽ kết thúc với tỷ số **HÒA**!")
                else:
                    st.error(f"🔥 **BẤT NGỜ XẢY RA:** **{away_team}** sẽ xuất sắc đánh bại đội chủ nhà!")
                    st.snow() # Hiệu ứng tuyết rơi cho sự bất ngờ
                    
            with res_col2:
                st.markdown("🎯 **TỶ LỆ XÁC SUẤT (PROBABILITY):**")
                prob_A = probabilities[list(classes).index('A')] * 100
                prob_D = probabilities[list(classes).index('D')] * 100
                prob_H = probabilities[list(classes).index('H')] * 100
                
                # Hiển thị thanh tỷ lệ
                st.progress(int(prob_H), text=f"🔵 {home_team} thắng: {prob_H:.0f}%")
                st.progress(int(prob_D), text=f"⚪ Hòa nhau: {prob_D:.0f}%")
                st.progress(int(prob_A), text=f"🔴 {away_team} thắng: {prob_A:.0f}%")

with tab2:
    st.markdown("### 🌈 Bức tranh dữ liệu toàn cảnh")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("📚 Tổng số trận đấu", f"{len(df)} trận")
    m2.metric("🛡️ Số lượng câu lạc bộ", f"{len(all_teams)} đội")
    m3.metric("🧠 Thuật toán AI", "K-Nearest Neighbors")
    
    st.markdown("#### 🔍 Bộ lọc dữ liệu thông minh")
    search_term = st.text_input("Nhập tên đội bóng bạn muốn tìm (VD: Arsenal, Chelsea...):")
    if search_term:
        filtered_df = df[df['HomeTeam'].str.contains(search_term, case=False) | df['AwayTeam'].str.contains(search_term, case=False)]
        st.dataframe(filtered_df, use_container_width=True)
    else:
        st.dataframe(df, use_container_width=True)
