# final_project_AI

Demo AI dự đoán nguy cơ tiểu đường bằng Python và Streamlit.

App hỗ trợ 3 mô hình:

- Logistic Regression
- Decision Tree
- KNN

## Cài thư viện

```bash
python -m pip install -r requirements.txt
```

## Huấn luyện model

```bash
python train_model.py
```

Lệnh này sẽ tạo file model tại `models/diabetes_models.joblib`.

## Chạy app demo

```bash
streamlit run app.py
```

Sau khi chạy, mở địa chỉ Streamlit hiển thị trong terminal, thường là:

```text
http://localhost:8501
```
