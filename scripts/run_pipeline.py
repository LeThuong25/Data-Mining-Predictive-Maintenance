import os
import sys

# Thêm đường dẫn gốc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.loader import fetch_and_load_data
from src.data.cleaner import clean_and_preprocess
from src.mining.association import mine_failure_rules
from src.mining.clustering import cluster_machine_behavior
from src.models.supervised import run_classification_models, run_regression_timeseries

def main():
    print("🚀 BẮT ĐẦU CHẠY PIPELINE TỰ ĐỘNG...")
    
    # 1. Dữ liệu
    print("\n[1/4] Đang tải và tiền xử lý dữ liệu...")
    df_raw = fetch_and_load_data()
    df_processed = clean_and_preprocess(df_raw)
    
    # 2. Khai phá (Mining)
    print("\n[2/4] Đang chạy thuật toán Khai phá (Apriori & K-Means)...")
    rules = mine_failure_rules(df_raw)
    _, sil_score, _ = cluster_machine_behavior(df_processed)
    print(f"-> Tìm thấy {len(rules)} luật kết hợp. Silhouette Score: {sil_score:.3f}")
    
    # 3. Mô hình hóa (Modeling)
    print("\n[3/4] Đang huấn luyện Mô hình (Phân lớp & Hồi quy)...")
    class_metrics, _ = run_classification_models(df_processed, df_raw)
    reg_metrics = run_regression_timeseries(df_processed, df_raw)
    print("-> Đã huấn luyện xong Random Forest (PR-AUC, MAE).")
    
    # 4. Hoàn tất
    print("\n[4/4] Pipeline hoàn tất thành công! Xem chi tiết trong thư mục notebooks/")
    print("🎉 Gõ 'streamlit run app.py' để mở giao diện web.")

if __name__ == "__main__":
    main()