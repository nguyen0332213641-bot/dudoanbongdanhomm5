import streamlit as st
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

# 1. Cấu hình trang
st.set_page_config(page_title="AI Bóng Đá - Nhóm 5", page_icon="⚽", layout="wide")

# 2. CSS trang trí giao diện 
st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; }
    .main-title {
        font-size: 45px; font-weight: 900;
        background: linear-gradient(45deg, #FF416C, #FF4B2B);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; margin-bottom: 5px; text-transform: uppercase;
    }
    .sub-title { font-size: 20px; color: #4A00E0; text-align: center; font-weight: bold; margin-bottom: 30px; }
    .team-box-home { background: linear-gradient(135deg, #74ebd5, #ACB6E5); padding: 15px; border-radius: 10px; text-align: center; font-size: 20px; font-weight: bold; }
    .team-box-away { background: linear-gradient(135deg, #ff9a9e, #fecfef); padding: 15px; border-radius: 10px; text-align: center; font-size: 20px; font-weight: bold; }
    .stButton>button { background: linear-gradient(90deg, #00C9FF, #92FE9D); color: #000; font-size: 20px; font-weight: bold; border-radius: 30px; padding: 10px; }
    .vs-text { font-size: 35px; font-weight: 900; color: #ffaa00; text-align: center; margin-top: 20px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">⚽ DỰ ĐOÁN NGOẠI HẠNG ANH BẰNG AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">🤖 Đồ án: Trí tuệ nhân tạo - Thực hiện: Nhóm 5</div>', unsafe_allow_html=True)

# 3. Hàm tải dữ liệu và huấn luyện AI
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

# 4. Giao diện chính
tab1, tab2 = st.tabs(["🎮 THỰC HIỆN DỰ ĐOÁN", "📊 KHÁM PHÁ DỮ LIỆU"])

with tab1:
    st.write("---")
    col1, col_vs, col2 = st.columns([4, 1, 4])
    
    with col1:
        st.markdown('<div class="team-box-home">🏰 ĐỘI NHÀ (HOME)</div>', unsafe_allow_html=True)
        home_team = st.selectbox("", sorted(all_teams), key="home", label_visibility="collapsed")
    
    with col_vs:
        st.markdown('<div class="vs-text">VS</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="team-box-away">🚀 ĐỘI KHÁCH (AWAY)</div>', unsafe_allow_html=True)
        away_team = st.selectbox("", sorted(all_teams), key="away", label_visibility="collapsed")

    st.write("")
    predict_btn = st.button("⚡ KÍCH HOẠT AI DỰ ĐOÁN KẾT QUẢ ⚡", use_container_width=True)

    if predict_btn:
        if home_team == away_team:
            st.error("🚨 Ối! Đội nhà và Đội khách không thể giống nhau!")
        else:
            input_data = pd.DataFrame({
                'HomeTeam_enc': [le.transform([home_team])[0]],
                'AwayTeam_enc': [le.transform([away_team])[0]]
            })
            
            prediction = knn.predict(input_data)[0]
            probabilities = knn.predict_proba(input_data)[0]
            classes = knn.classes_ 
            
            st.write("---")
            st.subheader("🏆 KẾT QUẢ TỪ TRÍ TUỆ NHÂN TẠO:")
            
            res_col1, res_col2 = st.columns([1.2, 1])
            with res_col1:
                if prediction == 'H':
                    st.success(f"🎉 **CHIẾN THẮNG:** {home_team} sẽ tận dụng lợi thế sân nhà để giành 3 điểm!")
                    st.balloons() 
                elif prediction == 'D':
                    st.info("⚖️ **BẤT PHÂN THẮNG BẠI:** Trận đấu sẽ có kết quả HÒA!")
                else:
                    st.error(f"🔥 **THẤT BẠI:** {away_team} sẽ xuất sắc đánh bại đội chủ nhà!")
                    st.snow()
                    
            with res_col2:
                st.write("**Tỷ lệ phần trăm:**")
                prob_H = probabilities[list(classes).index('H')] * 100
                prob_D = probabilities[list(classes).index('D')] * 100
                prob_A = probabilities[list(classes).index('A')] * 100
                st.progress(int(prob_H), text=f"Cửa {home_team} thắng: {prob_H:.0f}%")
                st.progress(int(prob_D), text=f"Cửa Hòa: {prob_D:.0f}%")
                st.progress(int(prob_A), text=f"Cửa {away_team} thắng: {prob_A:.0f}%")

with tab2:
    st.write("### Dữ liệu các trận đấu")
    st.dataframe(df, use_container_width=True)
