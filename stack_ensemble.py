import pandas as pd
import os
from functools import reduce
from scipy.stats import entropy
import numpy as np 
from utils import *
from sklearn.preprocessing import LabelEncoder
import joblib
import random
import numpy as np
import optuna

from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

seed_everything(22520465)

root_folder = './'
model_folders = ['output_dangvantuan_vietnamese-document-embedding', 'output_microsoft_deberta-xlarge-mnli', 'output_cross-encoder_nli-deberta-v3-large', 'output_FacebookAI_roberta-large-mnli', 'output_SemViQA_tc-erniem-viwikifc', 'output_SemViQA_tc-xlmr-isedsc01']
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
# 2. CV cố định
# ==========================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=22520465)

# ==========================
# 3. Hàm objective cho Optuna
# ==========================
def objective(trial):
    # Không gian tìm kiếm rộng, liên tục
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 1700),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.01, log=True),
        'subsample': trial.suggest_float('subsample', 0.4, 0.8),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
        'gamma': trial.suggest_float('gamma', 0.1, 0.95),
        'min_child_weight': trial.suggest_float('min_child_weight', 1.0, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
        'tree_method': 'hist',
        'device': 'cuda',
        'eval_metric': 'mlogloss',
        'random_state': 22520465,
        'n_jobs': -1
    }

    model = XGBClassifier(**params)

    # 5-fold cross-validation để tính F1 Macro trung bình
    score = cross_val_score(
        model, X, y,
        cv=cv,
        scoring='f1_macro',
        n_jobs=-1
    ).mean()

    return score

# ==========================
# 4. Tạo study và chạy
# ==========================
study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=22520465),  # Bayesian TPE
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
)

# 👉 n_trials bạn có thể tăng tùy thời gian, càng cao càng tốt
# 200-300 trials thường đã rất mạnh
study.optimize(objective, n_trials=300, show_progress_bar=True)

# ==========================
# 5. Kết quả
# ==========================
print("🎯 Best F1 Macro:", study.best_value)
print("🏆 Best Hyperparameters:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")

# ==========================
# 6. Train lại mô hình tốt nhất
# ==========================
best_params = study.best_params.copy()
best_params.update({
    'tree_method': 'hist',
    'device': 'cuda',
    'eval_metric': 'mlogloss',
    'random_state': 22520465,
    'n_jobs': -1
})

best_model = XGBClassifier(**best_params)
best_model.fit(X, y)

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