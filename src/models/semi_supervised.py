import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.metrics import average_precision_score

def run_semi_supervised_experiment(df_processed, df_raw):
    """Thực nghiệm bán giám sát với các mức % nhãn khác nhau"""
    features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'Type']
    X = df_processed[features]
    y = df_raw['Machine failure'].values
    
    # Chia tập train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Các mức % dữ liệu có nhãn để vẽ Learning Curve
    label_percentages = [0.05, 0.1, 0.2, 0.3]
    results = []
    
    # Biến lưu trữ rủi ro ở mốc 20%
    false_alarms = 0
    
    base_rf = RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=50)
    
    for p in label_percentages:
        # Giả lập thiếu nhãn: Giữ lại p% nhãn, phần còn lại gán là -1 (unlabeled)
        rng = np.random.RandomState(42)
        random_unlabeled_points = rng.rand(len(y_train)) > p
        
        y_train_semi = np.copy(y_train)
        y_train_semi[random_unlabeled_points] = -1 
        
        # 1. Huấn luyện mô hình Supervised (Chỉ học trên phần ít ỏi có nhãn)
        X_train_labeled = X_train[y_train_semi != -1]
        y_train_labeled = y_train_semi[y_train_semi != -1]
        
        if len(np.unique(y_train_labeled)) > 1: 
            base_rf.fit(X_train_labeled, y_train_labeled)
            y_prob_sup = base_rf.predict_proba(X_test)[:, 1]
            pr_auc_sup = average_precision_score(y_test, y_prob_sup)
        else:
            pr_auc_sup = 0
            
        # 2. Huấn luyện mô hình Semi-supervised (Tự học có ngưỡng cao 0.85)
        self_training_model = SelfTrainingClassifier(base_rf, threshold=0.85)
        self_training_model.fit(X_train, y_train_semi)
        y_prob_semi = self_training_model.predict_proba(X_test)[:, 1]
        pr_auc_semi = average_precision_score(y_test, y_prob_semi)
        
        results.append({
            'Label_Percent': p * 100,
            'Supervised_Only': pr_auc_sup,
            'Semi_Supervised': pr_auc_semi
        })
        
        # 3. Tính toán rủi ro Pseudo-label (False alarm) tại mốc 20%
        if p == 0.2:
            pseudo_labels = self_training_model.transduction_
            mask_unlabeled = (y_train_semi == -1)
            actual_labels = y_train[mask_unlabeled]
            predicted_pseudo = pseudo_labels[mask_unlabeled]
            
            # Đếm số ca máy BÌNH THƯỜNG (0) nhưng mô hình tự gán nhãn là LỖI (1)
            false_alarms = np.sum((predicted_pseudo == 1) & (actual_labels == 0))

    return pd.DataFrame(results), false_alarms