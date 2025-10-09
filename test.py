import os
import pandas as pd

def find_csv_files(root_folder, filename="submit_with_probs_privatetest.csv"):
    csv_files = []
    for dirpath, _, filenames in os.walk(root_folder):
        for file in filenames:
            if file == filename:
                csv_files.append(os.path.join(dirpath, file))
    return csv_files

def soft_voting_ensemble(csv_files):
    dfs = []
    for file in csv_files:
        df = pd.read_csv(file)
        dfs.append(df)
    
    # Giả sử tất cả file đều có cùng id theo thứ tự
    combined_df = dfs[0][['id']].copy()
    probs = ['prob_intrinsic', 'prob_extrinsic', 'prob_no']

    # Cộng dồn xác suất
    for prob in probs:
        combined_df[prob] = sum(df[prob] for df in dfs) / len(dfs)

    # Tìm nhãn có xác suất lớn nhất
    combined_df['predict_label'] = combined_df[probs].idxmax(axis=1).str.replace('prob_', '')

    return combined_df[['id', 'predict_label']]

# Thư mục gốc chứa các folder con
root_folder = "output_true"  # ← Thay đường dẫn thư mục gốc của bạn vào đây

# Tìm tất cả các file CSV cần thiết
csv_files = find_csv_files(root_folder)

# Nếu tìm được thì thực hiện soft voting
if csv_files:
    ensemble_df = soft_voting_ensemble(csv_files)
    ensemble_df.to_csv("ensemble_submission.csv", index=False)
    print("✅ Done! File 'ensemble_submission.csv' đã được tạo.")
else:
    print("❌ Không tìm thấy file nào có tên 'submit_with_probs_privatetest.csv'")
