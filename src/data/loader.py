# src/data/loader.py
import pandas as pd
import os
import yaml
from pathlib import Path

def get_project_root():
    """Tự động tìm đường dẫn gốc của project (DATA_MINING_PROJECT)"""
    # File này nằm ở src/data/loader.py -> lùi ra 3 cấp sẽ tới thư mục gốc
    return Path(__file__).resolve().parent.parent.parent

def load_config():
    """Đọc file cấu hình bằng đường dẫn tuyệt đối"""
    root = get_project_root()
    config_path = root / "configs" / "params.yaml"
    
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def fetch_and_load_data():
    config = load_config()
    root = get_project_root()
    
    # Kết hợp thư mục gốc với đường dẫn trong file config
    raw_path = root / config['data']['raw_path']
    raw_url = config['data']['raw_url']
    
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(raw_path.parent, exist_ok=True)
    
    # Tải dữ liệu nếu file chưa có
    if not raw_path.exists():
        print(f"Đang tải dữ liệu từ {raw_url} ...")
        df = pd.read_csv(raw_url)
        df.to_csv(raw_path, index=False)
        print("Tải dữ liệu thành công!")
    else:
        print("Dữ liệu đã tồn tại trong data/raw/.")
        df = pd.read_csv(raw_path)
        
    return df