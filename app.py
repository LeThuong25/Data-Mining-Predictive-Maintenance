import streamlit as st
import pandas as pd
import sys
import time
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# --- 1. CẤU HÌNH GIAO DIỆN ---
st.set_page_config(page_title="Predictive Maintenance", layout="wide")

# Phần Header mang phong cách Đồ án môn học
st.title("Hệ thống Phân tích và Dự đoán Lỗi Máy móc")
st.markdown("**Môn học:** Khai phá dữ liệu | **Dữ liệu:** AI4I 2020 Predictive Maintenance (UCI)")
st.markdown("*Mô hình sử dụng: Random Forest Classifier (Cân bằng lớp) & Luật kết hợp Apriori*")
st.markdown("---")

# --- 2. TRỎ ĐƯỜNG DẪN & TẢI MÔ HÌNH ---
sys.path.append(str(Path(__file__).resolve().parent))
from src.data.loader import fetch_and_load_data
from src.data.cleaner import clean_and_preprocess

@st.cache_resource
def load_and_train_model():
    df_raw = fetch_and_load_data()
    df_processed = clean_and_preprocess(df_raw)
    
    features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'Type']
    X = df_processed[features]
    y = df_raw['Machine failure']
    
    model = RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=100)
    model.fit(X, y)
    
    return model, df_raw

try:
    model, df_raw = load_and_train_model()
except Exception as e:
    st.error(f"Lỗi khởi tạo hệ thống: {e}")
    st.stop()

# --- 3. SIDEBAR: BẢNG ĐIỀU KHIỂN THÔNG SỐ ---
st.sidebar.header("THÔNG SỐ ĐẦU VÀO")
st.sidebar.markdown("Điều chỉnh các giá trị cảm biến để kiểm tra rủi ro hỏng hóc.")

type_input = st.sidebar.selectbox("Chất lượng thiết bị (Type)", ["L (Low)", "M (Medium)", "H (High)"], index=1)
t_val = type_input[0]

air_temp = st.sidebar.number_input("Nhiệt độ không khí [K]", min_value=290.0, max_value=310.0, value=298.0, step=0.1)
process_temp = st.sidebar.number_input("Nhiệt độ quá trình [K]", min_value=300.0, max_value=320.0, value=310.0, step=0.1)
rpm = st.sidebar.slider("Tốc độ quay [rpm]", min_value=1100, max_value=3000, value=1500, step=10)
torque = st.sidebar.slider("Mô-men xoắn [Nm]", min_value=3.0, max_value=80.0, value=40.0, step=0.5)
tool_wear = st.sidebar.slider("Độ mòn dụng cụ [min]", min_value=0, max_value=300, value=100, step=1)

# --- 4. XỬ LÝ DỮ LIỆU ĐẦU VÀO ---
def preprocess_input(t_val, air, process, rpm, torq, wear, df_raw):
    type_mapping = {'H': 0, 'L': 1, 'M': 2}
    input_df = pd.DataFrame([[air, process, rpm, torq, wear, type_mapping[t_val]]], 
                            columns=['Air temperature [K]', 'Process temperature [K]', 
                                     'Rotational speed [rpm]', 'Torque [Nm]', 
                                     'Tool wear [min]', 'Type'])
    num_cols = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    temp_df = df_raw[num_cols].copy()
    temp_df.loc[len(temp_df)] = [air, process, rpm, torq, wear]
    
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(temp_df)
    input_df[num_cols] = scaled_values[-1]
    return input_df, num_cols

X_input, num_cols = preprocess_input(t_val, air_temp, process_temp, rpm, torque, tool_wear, df_raw)

# --- 5. HIỂN THỊ MAIN LAYOUT ---
col_metric1, col_metric2, col_metric3 = st.columns(3)
temp_diff = process_temp - air_temp
col_metric1.metric("Chênh lệch nhiệt (Tản nhiệt)", f"{temp_diff:.1f} K", "- Bất thường" if temp_diff < 8.6 else "")
col_metric2.metric("Mức hao mòn hiện tại", f"{tool_wear} phút", "- Cần thay thế" if tool_wear > 200 else "")
col_metric3.metric("Công suất ước tính", f"{(rpm * torque / 9550):.2f} kW")

st.markdown("### 1. Phân tích vector dữ liệu đầu vào")
st.write("Dữ liệu cảm biến sau khi đi qua bộ `StandardScaler` (Mean=0, Std=1) trước khi đưa vào Random Forest:")
st.dataframe(X_input)

# Nút kích hoạt
st.markdown("### 2. Kết quả dự đoán")
if st.button("Thực thi mô hình Random Forest", type="primary"):
    with st.spinner("Đang tính toán xác suất..."):
        time.sleep(0.5)
        
    prediction = model.predict(X_input)[0]
    probability = model.predict_proba(X_input)[0][1]
    
    res_col1, res_col2 = st.columns([1, 2])
    
    with res_col1:
        if prediction == 1:
            st.error("CẢNH BÁO LỖI: CÓ RỦI RO HỎNG HÓC")
        else:
            st.success("TRẠNG THÁI: HOẠT ĐỘNG BÌNH THƯỜNG")
            
        st.write(f"**Xác suất lỗi (Probability):** {probability:.4f}")
        st.progress(float(probability))

    with res_col2:
        st.write("**Ghi chú vận hành (Rút trích từ luật Apriori):**")
        if probability > 0.5:
            if tool_wear > 200:
                st.write("- Lỗi hao mòn (TWF): Độ mòn vượt ngưỡng an toàn (200 phút).")
            if torque > 50 and rpm < 1400:
                st.write("- Lỗi quá tải (OSF): Cảnh báo mô-men xoắn cao kết hợp vòng tua thấp.")
            if temp_diff < 8.6:
                st.write("- Lỗi tản nhiệt (HDF): Chênh lệch nhiệt độ không đảm bảo thoát nhiệt.")
        else:
            st.write("- Các thông số vận hành đang nằm trong phạm vi cụm phân phối an toàn.")
            st.write("- Đề xuất tiếp tục duy trì lịch bảo trì định kỳ theo chu kỳ.")