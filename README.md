# Data Mining Project - Đề tài 16: Phân tích và Dự đoán lỗi máy móc (Predictive Maintenance)

##  Giới thiệu dự án
Dự án này ứng dụng các kỹ thuật Khai phá dữ liệu (Data Mining) và Học máy (Machine Learning) để phân tích hành vi vận hành và dự đoán rủi ro hỏng hóc của máy móc công nghiệp. Hệ thống giúp tối ưu hóa lịch bảo trì, giảm thiểu thời gian chết (downtime) và tiết kiệm chi phí cho nhà máy.

## 1. Nguồn dữ liệu (Data Source)
* **Dataset:** AI4I 2020 Predictive Maintenance Dataset (UCI).
* **Đặc trưng:** Bao gồm 10,000 bản ghi dữ liệu cảm biến (Nhiệt độ không khí, Nhiệt độ quá trình, Tốc độ quay, Mô-men xoắn, Độ mòn dụng cụ).
* **Link tải:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv)
* *Lưu ý: Dữ liệu sẽ được script tự động tải về thư mục `data/raw/` trong lần chạy đầu tiên.*

## 2. Quy trình thực hiện (Methodology)
Dự án tuân thủ chặt chẽ quy trình khai phá dữ liệu tiêu chuẩn:
1. **Tiền xử lý & EDA:** Xử lý rủi ro mất cân bằng lớp (Class Imbalance) và ngăn chặn Data Leakage bằng cách loại bỏ các nhãn phụ. Chuẩn hóa dữ liệu bằng `StandardScaler`.
2. **Khai phá tri thức (Data Mining Core):** Áp dụng thuật toán **Apriori** để tìm luật kết hợp giữa các nguyên nhân gây lỗi, và **K-Means** để phân cụm trạng thái hoạt động.
3. **Mô hình hóa (Modeling):** Xây dựng các Baseline (Logistic Regression, Decision Tree) và mô hình cải tiến (**Random Forest**).
4. **Bán giám sát (Semi-supervised):** Thực nghiệm giả lập kịch bản thiếu nhãn (chỉ có 10% - 30% dữ liệu có nhãn) bằng kỹ thuật Pseudo-labeling.

## 3. Kết quả nổi bật (Key Results)

### A. Kết quả Mô hình Phân lớp (Phát hiện lỗi)
Mô hình **Random Forest (Balanced Class Weight)** vượt trội hơn hẳn các Baseline do xử lý tốt tình trạng mất cân bằng dữ liệu:
* **PR-AUC:** Đạt độ chính xác cao trong việc nhận diện đúng lớp thiểu số (Máy hỏng).
* **F1-Score:** Cân bằng tốt giữa Precision và Recall, hạn chế tối đa rủi ro bỏ lọt lỗi (False Negative).

### B. Kết quả Khai phá & Insight (Actionable Insights)
Dựa trên luật kết hợp Apriori và phân tích đặc trưng, hệ thống trích xuất được 5 khuyến nghị vận hành thực tiễn:
1. **Cảnh báo hao mòn (TWF):** Rủi ro tăng đột biến khi độ mòn dụng cụ (Tool Wear) vượt mốc **200 phút**.
2. **Cảnh báo quá tải (OSF):** Lỗi xảy ra khi máy bị ép tải (Mô-men xoắn > 50 Nm) trong khi tốc độ tua máy thấp (< 1400 rpm).
3. **Cảnh báo tản nhiệt (HDF):** Hệ thống tản nhiệt hoạt động kém khi độ chênh lệch giữa Nhiệt độ quá trình và Nhiệt độ không khí **< 8.6 K**.
4. **Bảo trì theo cụm (Clustering):** Các thiết bị phân khúc chất lượng thấp (Type L) cần chu kỳ kiểm tra ngắn hơn.
5. **Dán nhãn tự động:** Mô hình Bán giám sát (Semi-supervised) cho thấy tiềm năng tự động hóa việc dán nhãn dữ liệu cảm biến thô với độ tin cậy chấp nhận được khi có >30% dữ liệu gốc.

## 4. Cấu trúc Project Repo

Dự án được cấu trúc theo chuẩn module hóa (Industry Standard):

```text
DATA_MINING_PROJECT/
├── configs/          # Chứa tham số cấu hình (params.yaml)
├── data/             # Thư mục chứa dữ liệu (tự động bỏ qua bởi .gitignore)
├── notebooks/        # File báo cáo Jupyter Notebook (01 -> 05)
├── outputs/          # Lưu trữ hình ảnh, bảng biểu, model xuất ra
├── scripts/          # Script chạy tự động (run_pipeline.py)
├── src/              # Mã nguồn thực thi lõi (Tiền xử lý, Khai phá, Mô hình)
├── app.py            # Giao diện Streamlit Dashboard
├── requirements.txt  # Danh sách thư viện môi trường
└── README.md


##  5. Hướng dẫn chạy lại mã nguồn (Reproducible)
Bước 1: Cài đặt môi trường
Mở Terminal tại thư mục gốc của dự án và chạy lệnh:

pip install -r requirements.txt

Bước 2: Chạy tự động toàn bộ Pipeline
Để tái lập toàn bộ quy trình từ tải dữ liệu, tiền xử lý, huấn luyện đến đánh giá, hãy chạy script:

python scripts/run_pipeline.py

Bước 3: Trải nghiệm Giao diện AI Dashboard
Hệ thống tích hợp một Dashboard giám sát thời gian thực. Khởi chạy bằng lệnh:

streamlit run app.py

