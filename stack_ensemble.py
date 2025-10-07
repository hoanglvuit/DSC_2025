import pandas as pd
import os
from utils import *
from sklearn.preprocessing import LabelEncoder
import joblib
import numpy as np
import joblib
import argparse


from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

seed_everything(22520465)
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_folder", type=str, default = "output_true")
    parser.add_argument("--use_true", type=str2bool, default=True)
    args = parser.parse_args()
    root_folder = args.root_folder
    use_true = args.use_true

    if use_true: 
        model_folders = ['erniem', 'xlmr', 'deberta', 'cross', 'roberta', 'dvt'] 
    else: 
        rename_map = {
            "FacebookAI_roberta-large-mnli": "roberta",
            "microsoft_deberta-xlarge-mnli": "deberta",
            "SemViQA_tc-erniem-viwikifc": "erniem",
            "SemViQA_tc-xlmr-isedsc01": "xlmr", 
            "cross-encoder_nli-deberta-v3-large": "cross",
            "dangvantuan_vietnamese-document-embedding": "dvt",

        }

        # Lặp qua các cặp tên cũ - mới
        for old_name, new_name in rename_map.items():
            old_path = os.path.join(root_folder, old_name)
            new_path = os.path.join(root_folder, new_name)
            
            if os.path.exists(old_path):
                if not os.path.exists(new_path):  # tránh ghi đè nếu tên mới đã tồn tại
                    os.rename(old_path, new_path)
                    print(f"Đã đổi: {old_name} -> {new_name}")
                else:
                    print(f"Tên mới '{new_name}' đã tồn tại. Bỏ qua.")
            else:
                print(f"Thư mục '{old_name}' không tồn tại. Bỏ qua.")
        model_folders = ['erniem', 'xlmr', 'deberta', 'cross', 'roberta', 'dvt'] 


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

    # Load model từ file
    best_model = joblib.load("xgb_best_model.pkl")

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