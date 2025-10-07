import pandas as pd
import os
from utils import *
from sklearn.preprocessing import LabelEncoder
import joblib
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

seed_everything(22520465)

root_folder = 'output_true'
model_folders = [f for f in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, f))]
train_file = 'dev_predictions_with_probs.csv'
test_file = 'submit_with_probs_privatetest.csv'

train_meta = feature_engineering(root_folder, model_folders, file=train_file)

test_meta = feature_engineering(root_folder, model_folders, file=test_file, train=False)


# Đọc dữ liệu nhãn
train_df = pd.read_csv('data/train_dsc.csv')  # chứa 'id' và 'label' (chuỗi)

# Encode nhãn (chuỗi -> số nguyên)
le = LabelEncoder()
train_df['label_encoded'] = le.fit_transform(train_df['label'])

# Gộp với train_meta theo 'id' để tạo X, y tương ứng
merged_df = pd.merge(train_meta, train_df[['id', 'label_encoded']], on='id')

# Tách X và y
X = merged_df.drop(columns=['id', 'label_encoded'])
y = merged_df['label_encoded']



# ==========================
# 1. CV cố định (nếu bạn còn cần dùng để đánh giá thủ công)
# ==========================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=22520465)

# ==========================
# 2. Tham số đã biết
# ==========================
best_params = {
    'colsample_bytree': 0.8,
    'gamma': 0,
    'learning_rate': 0.01,
    'max_depth': 3,
    'n_estimators': 500,
    'subsample': 1,
    'tree_method': 'hist',
    'device': 'cuda',
    'eval_metric': 'mlogloss',
    'random_state': 22520465,
    'n_jobs': -1
}

# ==========================
# 3. Train mô hình
# ==========================
best_model = XGBClassifier(**best_params)
best_model.fit(X, y)

# ==========================
# 4. (Tuỳ chọn) Đánh giá CV nếu cần kiểm tra lại F1 macro
# ==========================
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score

score = cross_val_score(
    best_model, X, y,
    cv=cv,
    scoring='f1_macro',
    n_jobs=-1
).mean()

print(f"✅ F1 Macro CV Score: {score:.4f}")

# 👉 Sau khi train xong bạn có thể lưu best_model bằng pickle hoặc joblib
joblib.dump(best_model, "xgb_best_model.pkl")

# Lấy đặc trưng đầu vào X_test (loại bỏ cột id)
X_test = test_meta.drop(columns=['id'])

# Dự đoán nhãn dạng số
y_test_pred = best_model.predict(X_test)

# Lấy lại tên nhãn tương ứng
y_test_label = le.inverse_transform(y_test_pred)

# Gộp id và nhãn dự đoán
submit_df = pd.DataFrame({
    'id': test_meta['id'],
    'predict_label': y_test_label
})

# Lưu file CSV
submit_df.to_csv('submit.csv', index=False)