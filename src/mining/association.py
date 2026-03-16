import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import yaml
from pathlib import Path

def get_project_root():
    return Path(__file__).resolve().parent.parent.parent

def load_config():
    config_path = get_project_root() / "configs" / "params.yaml"
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def mine_failure_rules(df_raw):
    """
    Rời rạc hóa các thông số cảm biến và tìm luật kết hợp dẫn đến lỗi.
    """
    config = load_config()
    min_sup = config['mining']['apriori_min_support']
    min_lift = config['mining']['apriori_min_lift']
    
    # Rời rạc hóa (Binning) trạng thái máy theo ngưỡng nguy hiểm
    df_bins = pd.DataFrame()
    df_bins['Temp_High'] = df_raw['Air temperature [K]'] > 300
    df_bins['Torque_High'] = df_raw['Torque [Nm]'] > 50
    df_bins['Speed_Low'] = df_raw['Rotational speed [rpm]'] < 1400
    df_bins['Tool_Wear_High'] = df_raw['Tool wear [min]'] > 200
    df_bins['Machine_Failure'] = df_raw['Machine failure'] == 1
    
    # Tìm tập phổ biến (Frequent Itemsets)
    freq_items = apriori(df_bins, min_support=min_sup, use_colnames=True)
    
    # Rút trích luật kết hợp (Association Rules)
    rules = association_rules(freq_items, metric="lift", min_threshold=min_lift)
    
    # Lọc ra các luật mà Hậu quả (Consequents) là gây ra hỏng máy
    failure_rules = rules[rules['consequents'].apply(lambda x: 'Machine_Failure' in str(x))]
    
    # Sắp xếp theo độ tin cậy (Confidence) giảm dần
    failure_rules = failure_rules.sort_values(by='confidence', ascending=False)
    
    return failure_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]