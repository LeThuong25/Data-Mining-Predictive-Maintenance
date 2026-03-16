# Data Mining Project - Đề tài 16: Phân tích lỗi sản xuất & dự đoán lỗi

## 1. Nguồn dữ liệu
* **Dataset:** AI4I 2020 Predictive Maintenance Dataset (UCI).
* **Link tải:** [https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv]

## 2. Cấu trúc Project Repo
Dự án được cấu trúc theo chuẩn module hóa:
* `configs/`: Chứa các tham số cấu hình (`params.yaml`).
* `src/`: Mã nguồn thực thi (Tiền xử lý, Apriori, K-Means, Supervised, Semi-supervised).
* `notebooks/`: Các file trình bày kết quả EDA, Khai phá và Mô hình hóa.
* `app.py`: Demo ứng dụng dự đoán lỗi bằng Streamlit.

## 3. Hướng dẫn chạy lại mã nguồn (Reproducible)
**Bước 1:** Cài đặt môi trường và thư viện:
`pip install -r requirements.txt`

**Bước 2:** Chạy các file Notebook theo thứ tự từ `01` đến `05`. Dữ liệu sẽ được tự động tải về thư mục `data/raw/` nhờ script `src/data/loader.py`.

**Bước 3:** Khởi chạy Demo App:
`streamlit run app.py`