import streamlit as st
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

# --- CẤU HÌNH GIAO DIỆN WEB ---
st.set_page_config(page_title="Dự đoán Bóng đá - Nhóm 5", layout="centered")
st.title("⚽ Ứng dụng AI Dự đoán Ngoại Hạng Anh")
st.markdown("**Đồ án: Trí tuệ nhân tạo và ứng dụng - Nhóm 5**")

# --- HÀM TẢI VÀ XỬ LÝ DỮ LIỆU ---
@st.cache_data
def load_data():
    df = pd.read_excel('nhom5_bongda_train.xlsx')
    
    # Tạo nhãn kết quả
    def get_result(row):
        if row['FTHG'] > row['FTAG']: return 'H'
        elif row['FTHG'] == row['FTAG']: return 'D'
        else: return 'A'
        
    df['Result'] = df.apply(get_result, axis=1)
    return df

df = load_data()

# --- HUẤN LUYỆN MÔ HÌNH ---
# Mã hóa tên đội
le = LabelEncoder()
all_teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
le.fit(all_teams)

# Chuẩn bị dữ liệu học
X = pd.DataFrame({
    'HomeTeam_enc': le.transform(df['HomeTeam']),
    'AwayTeam_enc': le.transform(df['AwayTeam'])
})
y = df['Result']

# Chạy KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

# --- THIẾT KẾ GIAO DIỆN (UI) ---
st.sidebar.header("⚙️ Tùy chỉnh Trận đấu")
home_team = st.sidebar.selectbox("🏠 Chọn Đội nhà (Home):", sorted(all_teams))
away_team = st.sidebar.selectbox("✈️ Chọn Đội khách (Away):", sorted(all_teams))

if home_team == away_team:
    st.sidebar.warning("⚠️ Đội nhà và Đội khách không thể trùng nhau!")

if st.sidebar.button("🚀 Dự đoán ngay"):
    if home_team != away_team:
        # Chuyển đổi lựa chọn của người dùng thành số
        input_data = pd.DataFrame({
            'HomeTeam_enc': [le.transform([home_team])[0]],
            'AwayTeam_enc': [le.transform([away_team])[0]]
        })
        
        # Mô hình dự đoán
        prediction = knn.predict(input_data)[0]
        
        # Hiển thị kết quả
        st.subheader("📊 Kết quả phân tích từ AI:")
        if prediction == 'H':
            st.success(f"🏆 Dự đoán: **{home_team}** sẽ THẮNG trên sân nhà!")
        elif prediction == 'D':
            st.info("🤝 Dự đoán: Hai đội sẽ HÒA nhau!")
        else:
            st.error(f"🔥 Dự đoán: **{away_team}** sẽ giành chiến thắng trên sân khách!")
            
        st.markdown("---")
        st.write("🔍 **Dữ liệu huấn luyện mẫu (5 trận gần nhất):**")
        st.dataframe(df[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'Result']].head(5))