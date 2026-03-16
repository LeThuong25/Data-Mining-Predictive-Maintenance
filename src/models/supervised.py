import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier   # <-- ĐÃ THÊM IMPORT Ở ĐÂY
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import average_precision_score, f1_score, mean_absolute_error, mean_squared_error

def run_classification_models(df_processed, df_raw):
    """Huấn luyện mô hình Phân lớp dự đoán Lỗi máy móc"""
    # 1. Chuẩn bị features và target
    features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'Type']
    X = df_processed[features]
    y = df_raw['Machine failure']
    
    # Chia tập train/test (Dùng stratify để giữ nguyên tỷ lệ imbalance)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 2. Baseline 1: Logistic Regression (Có xử lý mất cân bằng lớp)
    lr_base = LogisticRegression(class_weight='balanced', random_state=42)
    lr_base.fit(X_train, y_train)
    y_pred_lr = lr_base.predict(X_test)
    y_prob_lr = lr_base.predict_proba(X_test)[:, 1]
    
    # --- ĐÃ THÊM: Baseline 2: Decision Tree ---
    dt_base = DecisionTreeClassifier(class_weight='balanced', random_state=42, max_depth=5)
    dt_base.fit(X_train, y_train)
    y_pred_dt = dt_base.predict(X_test)
    y_prob_dt = dt_base.predict_proba(X_test)[:, 1]
    
    # 3. Mô hình cải tiến: Random Forest
    rf_model = RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=100)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    y_prob_rf = rf_model.predict_proba(X_test)[:, 1]
    
    # 4. Tính toán Metrics (PR-AUC và F1)
    metrics = {
        'Baseline 1 (Logistic Regression)': {
            'PR-AUC': average_precision_score(y_test, y_prob_lr),
            'F1-Score': f1_score(y_test, y_pred_lr)
        },
        'Baseline 2 (Decision Tree)': {         # <-- ĐÃ THÊM KẾT QUẢ BASELINE 2
            'PR-AUC': average_precision_score(y_test, y_prob_dt),
            'F1-Score': f1_score(y_test, y_pred_dt)
        },
        'Cải tiến (Random Forest)': {
            'PR-AUC': average_precision_score(y_test, y_prob_rf),
            'F1-Score': f1_score(y_test, y_pred_rf)
        }
    }
    
    # 5. Phân tích lỗi (Error Analysis) theo loại lỗi gốc
    test_indices = X_test.index
    df_test_raw = df_raw.loc[test_indices]
    
    # Lọc ra các ca Máy Hỏng (y=1) nhưng mô hình đoán Bình Thường (y_pred=0)
    fn_mask = (y_test == 1) & (y_pred_rf == 0)
    fn_cases = df_test_raw[fn_mask]
    
    error_analysis = {
        'TWF (Lỗi hao mòn)': fn_cases['TWF'].sum(),
        'HDF (Lỗi tản nhiệt)': fn_cases['HDF'].sum(),
        'PWF (Lỗi nguồn)': fn_cases['PWF'].sum(),
        'OSF (Lỗi quá tải)': fn_cases['OSF'].sum(),
        'RNF (Lỗi ngẫu nhiên)': fn_cases['RNF'].sum()
    }
    
    return metrics, error_analysis

def run_regression_timeseries(df_processed, df_raw):
    """Huấn luyện mô hình Hồi quy dự đoán Tool Wear (Không shuffle)"""
    # Đã sửa 'UID' thành 'UDI' theo đúng thực tế của file CSV
    df_ts = df_processed.copy()
    df_ts['UDI'] = df_raw['UDI']
    df_ts = df_ts.sort_values(by='UDI') # Sắp xếp theo thứ tự quan sát
    
    # Tạo Lag-features (Đặc trưng trễ của chu kỳ trước)
    cols_to_lag = ['Air temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]']
    for col in cols_to_lag:
        df_ts[f'{col}_lag1'] = df_ts[col].shift(1)
        
    df_ts = df_ts.dropna() # Bỏ dòng đầu tiên bị NaN do hàm shift
    
    # Chuẩn bị X, y (Target là Tool wear [min])
    features_ts = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Type'] + [f'{c}_lag1' for c in cols_to_lag]
    X_ts = df_ts[features_ts]
    y_ts = df_ts['Tool wear [min]']
    
    # ĐIỂM ĂN TIỀN: Tuyệt đối không shuffle khi chia train/test cho chuỗi thời gian
    X_train, X_test, y_train, y_test = train_test_split(X_ts, y_ts, test_size=0.2, shuffle=False)
    
    # Huấn luyện
    rf_reg = RandomForestRegressor(random_state=42, n_estimators=100)
    rf_reg.fit(X_train, y_train)
    y_pred = rf_reg.predict(X_test)
    
    metrics = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred))
    }
    return metrics