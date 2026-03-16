import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import yaml
from pathlib import Path

def get_project_root():
    return Path(__file__).resolve().parent.parent.parent

def load_config():
    config_path = get_project_root() / "configs" / "params.yaml"
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def cluster_machine_behavior(df_processed):
    """
    Phân cụm trạng thái máy móc và đánh giá bằng Silhouette score.
    """
    config = load_config()
    n_clusters = config['mining']['kmeans_clusters']
    
    # Chọn các cột thông số vận hành (bỏ các cột đã encode loại máy)
    features = ['Air temperature [K]', 'Process temperature [K]', 
                'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    X = df_processed[features]
    
    # Khởi tạo và huấn luyện KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    clusters = kmeans.fit_predict(X)
    
    # Đánh giá cụm
    sil_score = silhouette_score(X, clusters)
    
    # Gắn nhãn cụm vào dataframe để phân tích (profiling)
    df_processed['Behavior_Cluster'] = clusters
    
    return df_processed, sil_score, kmeans