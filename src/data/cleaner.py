import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import yaml
from pathlib import Path

def get_project_root():
    return Path(__file__).resolve().parent.parent.parent

def load_config():
    root = get_project_root()
    config_path = root / "configs" / "params.yaml"
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def clean_and_preprocess(df):
    """
    Hàm thực hiện tiền xử lý dữ liệu: xóa cột thừa, encode và scale.
    """
    config = load_config()
    cols_to_drop = config['preprocessing']['drop_columns']
    
    # 1. Tránh Data Leakage: Xóa các cột mã lỗi chi tiết và ID
    df_cleaned = df.drop(columns=cols_to_drop, errors='ignore')
    
    # 2. Mã hóa biến phân loại (Encoding)
    if 'Type' in df_cleaned.columns:
        le = LabelEncoder()
        df_cleaned['Type'] = le.fit_transform(df_cleaned['Type'])
        
    # 3. Chuẩn hóa dữ liệu (Scaling)
    numeric_cols = ['Air temperature [K]', 'Process temperature [K]', 
                    'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    
    scaler = StandardScaler()
    df_cleaned[numeric_cols] = scaler.fit_transform(df_cleaned[numeric_cols])
    
    return df_cleaned