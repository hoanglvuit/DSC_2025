import os
import glob
import random
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
from scipy.special import softmax
import evaluate
from transformers import TrainerCallback
from datasets import Dataset
from transformers import AutoTokenizer


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Set seed for all GPU devices
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Disable benchmark for reproducibility
    # torch.use_deterministic_algorithms(True)  # Use deterministic algorithms when possible


def preparing_dataset(train_path: str, public_test_path: str, private_test_path: str, segment: bool, intrinsic: int, extrinsic: int, no: int, model_name: str): 
    train_dataset = pd.read_csv(train_path)
    pubtest_dataset = pd.read_csv(public_test_path)
    privatetest_dataset = pd.read_csv(private_test_path)

    # Convert data to string
    train_dataset = train_dataset.astype(str)
    pubtest_dataset = pubtest_dataset.astype(str)
    privatetest_dataset = privatetest_dataset.astype(str)
    if segment:
        import py_vncorenlp 
        py_vncorenlp.download_model(save_dir='./')
        rdrsegmenter = py_vncorenlp.VnCoreNLP(
            annotators=["wseg"], 
            save_dir='./')
        def segment_text(text):
            if pd.isna(text) or text.strip() == "":
                return ""
            return " ".join(rdrsegmenter.word_segment(text))
        
        for col in ["context_vi", "prompt_vi", "response_vi"]:
            if col in train_dataset.columns:
                train_dataset[col] = train_dataset[col].apply(segment_text)
            if col in pubtest_dataset.columns:
                pubtest_dataset[col] = pubtest_dataset[col].apply(segment_text)
            if col in privatetest_dataset.columns:
                privatetest_dataset[col] = privatetest_dataset[col].apply(segment_text)
        print(train_dataset['context_vi'].head(3))
        print(pubtest_dataset[["prompt_vi", "response_vi"]].head(3))

    # Define label2id 
    label2id = {
        "intrinsic": intrinsic,
        "extrinsic": extrinsic,
        "no": no
    }

    # Define id2label
    id2label = {v: k for k, v in label2id.items()}

    # Gán nhãn vào dataset theo mapping
    train_dataset["label"] = train_dataset["label"].map(label2id)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # convert to Dataset format
    train_dataset = Dataset.from_pandas(train_dataset)
    pubtest_dataset = Dataset.from_pandas(pubtest_dataset)
    privatetest_dataset = Dataset.from_pandas(privatetest_dataset)
    return train_dataset, pubtest_dataset, privatetest_dataset, tokenizer, id2label


def preprocess_and_tokenize(batch, max_length: int, use_prompt: str, tokenizer, lang: str):
    if use_prompt == "first":
        first = [p + " " + c for p, c in zip(batch[f"prompt_{lang}"], batch[f"context_{lang}"])]
        second = batch[f"response_{lang}"]
    elif use_prompt == "second": 
        first = batch[f"context_{lang}"]
        second = [p + " " + r for p, r in zip(batch[f"prompt_{lang}"], batch[f"response_{lang}"])]
    else:
        first = batch[f"context_{lang}"]
        second = batch[f"response_{lang}"]

    tokenized_outputs = tokenizer(
        text=first,             
        text_pair=second,          
        truncation="longest_first",       
        max_length=max_length,
        padding="max_length",
        return_token_type_ids=True,       
        return_attention_mask=True,
        return_tensors=None,             
    )
    return tokenized_outputs


def evaluate_dev(trainer, encoded_dev_dataset: Dataset, output_dir: str, id2label=None):
    # Tạo thư mục nếu chưa có
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "eval_results.txt")

    with open(log_path, "w", encoding="utf-8") as f:
        # --- Evaluate trên dev ---
        predictions = trainer.predict(encoded_dev_dataset)

        y_true = predictions.label_ids
        logits = predictions.predictions
        if isinstance(logits, tuple):  # đôi khi output là tuple
            logits = logits[0]

        probs = softmax(logits, axis=-1)
        y_pred = np.argmax(probs, axis=-1)

        # Metrics cơ bản
        f1_macro = f1_score(y_true, y_pred, average="macro")
        f1_micro = f1_score(y_true, y_pred, average="micro")
        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred)

        # Ghi ra log
        f.write(f"F1 Macro: {f1_macro:.4f}\n")
        f.write(f"F1 Micro: {f1_micro:.4f}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write("Confusion Matrix:\n")
        f.write(f"{cm}\n\n")
        f.write("Classification Report:\n")
        f.write(f"{report}\n")

        # Phân tích entropy và top1-top2 gap
        entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1)
        sorted_top2 = np.sort(probs, axis=1)[:, -2:]
        gap = sorted_top2[:, 1] - sorted_top2[:, 0]

        mask_correct = (y_true == y_pred)
        mask_wrong = ~mask_correct

        f.write(f"Average entropy (correct): {entropy[mask_correct].mean():.4f}\n")
        f.write(f"Average entropy (wrong): {entropy[mask_wrong].mean():.4f}\n")
        f.write(f"Average top1-top2 gap (correct): {gap[mask_correct].mean():.4f}\n")
        f.write(f"Average top1-top2 gap (wrong): {gap[mask_wrong].mean():.4f}\n")

        f.write("\nIndices of wrong predictions:\n")
        wrong_indices = np.where(mask_wrong)[0]
        f.write(", ".join(map(str, wrong_indices)))
        f.write("\n")

    # Lưu file csv dự đoán
    # Nếu không truyền id2label thì để y_pred là số
    if id2label is not None:
        pred_labels = [id2label[i] for i in y_pred]
    else:
        pred_labels = y_pred.tolist()

    submit_df = pd.DataFrame({
        "id": encoded_dev_dataset["id"],
        "predict_label": pred_labels,
        "true_label": y_true  # có thể thêm cột label thật để dễ đối chiếu
    })
    submit_df.to_csv(os.path.join(output_dir, "dev_predictions.csv"), index=False)

    probs_df = pd.DataFrame(probs, columns=[f"prob_{id2label[i] if id2label else i}" for i in range(probs.shape[1])])
    submit_with_probs = pd.concat([submit_df, probs_df], axis=1)
    submit_with_probs.to_csv(os.path.join(output_dir, "dev_predictions_with_probs.csv"), index=False)

    print(f"✅ Evaluation results saved in {output_dir}")
    print(f"✅ Dev predictions saved as csv in {output_dir}")



def evaluate_test(trainer, encoded_test_dataset: Dataset, id2label: dict, output_dir: str): 
    # --- Predict trên test ---
    predictions = trainer.predict(encoded_test_dataset)
    logits = predictions.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    
    probs = softmax(logits, axis=-1)
    pred_ids = np.argmax(probs, axis=-1)
    pred_labels = [id2label[i] for i in pred_ids]
    
    # File 1: chỉ id + predict_label
    submit_df = pd.DataFrame({
        "id": encoded_test_dataset["id"],
        "predict_label": pred_labels
    })
    submit_df.to_csv(os.path.join(output_dir, f"submit.csv"), index=False)
    
    # File 2: thêm xác suất 3 class
    probs_df = pd.DataFrame(probs, columns=[f"prob_{id2label[i]}" for i in range(probs.shape[1])])
    submit_with_probs = pd.concat([submit_df, probs_df], axis=1)
    submit_with_probs.to_csv(os.path.join(output_dir, f"submit_with_probs.csv"), index=False)
    
    # Số mẫu mỗi class
    print("Class counts on test:")
    print(submit_df["predict_label"].value_counts())
    
    submit_file = os.path.join(output_dir, f'submit.csv')
    submit_probs_file = os.path.join(output_dir, f'submit_with_probs.csv')
    print(f"Done. Files saved: {submit_file}, {submit_probs_file}")

def evaluate_pritest(trainer, encoded_test_dataset: Dataset, id2label: dict, output_dir: str): 
    # --- Predict trên test ---
    predictions = trainer.predict(encoded_test_dataset)
    logits = predictions.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    
    probs = softmax(logits, axis=-1)
    pred_ids = np.argmax(probs, axis=-1)
    pred_labels = [id2label[i] for i in pred_ids]
    
    # File 1: chỉ id + predict_label
    submit_df = pd.DataFrame({
        "id": encoded_test_dataset["id"],
        "predict_label": pred_labels
    })
    submit_df.to_csv(os.path.join(output_dir, f"submit_privatetest.csv"), index=False)
    
    # File 2: thêm xác suất 3 class
    probs_df = pd.DataFrame(probs, columns=[f"prob_{id2label[i]}" for i in range(probs.shape[1])])
    submit_with_probs = pd.concat([submit_df, probs_df], axis=1)
    submit_with_probs.to_csv(os.path.join(output_dir, f"submit_with_probs_privatetest.csv"), index=False)
    
    # Số mẫu mỗi class
    print("Class counts on privatetest:")
    print(submit_df["predict_label"].value_counts())
    
    submit_file = os.path.join(output_dir, f'submit_privatetest.csv')
    submit_probs_file = os.path.join(output_dir, f'submit_with_probs_privatetest.csv')
    print(f"Done. Files saved: {submit_file}, {submit_probs_file}")


def ensemble_submissions(id2label: dict, output_dir: str):
    """
    Ensemble predictions từ cross-validation folds:
    
    Cho TEST:
    1. submit_test_hard_vote.csv (majority vote, chỉ label)
    2. submit_test_soft_vote.csv (average prob, có cả xác suất)
    3. submit_test_soft_vote_label.csv (average prob, chỉ label)
    
    Cho DEV (OOF):
    4. train_with_oof_predictions.csv (train data với OOF predictions từ tất cả folds)
    
    Args:
        output_dir (str): path to folder containing fold_* subdirectories
        id2label (dict): mapping id->label
    """
    # Tìm tất cả fold directories
    fold_dirs = sorted(glob.glob(os.path.join(output_dir, "fold_*")))
    if not fold_dirs:
        raise ValueError(f"Cannot find fold_* directories in {output_dir}!")
    
    print(f"Found {len(fold_dirs)} folds: {[os.path.basename(d) for d in fold_dirs]}")
    
    # ===== XỬ LÝ TEST PREDICTIONS =====
    test_files = []
    for fold_dir in fold_dirs:
        test_file = os.path.join(fold_dir, "submit_with_probs.csv")
        if os.path.exists(test_file):
            test_files.append(test_file)
        else:
            print(f"Warning: {test_file} not found")
    
    if test_files:
        print(f"Processing {len(test_files)} test prediction files...")
        test_dfs = [pd.read_csv(f) for f in test_files]
        test_ids = test_dfs[0]["id"]
        
        # --- Hard Vote (Majority Vote) cho TEST ---
        test_vote_preds = []
        for i in range(len(test_ids)):
            labels_i = [df.loc[i, "predict_label"] for df in test_dfs]
            vote = pd.Series(labels_i).mode()[0]  
            test_vote_preds.append(vote)

        test_submit_vote = pd.DataFrame({
            "id": test_ids,
            "predict_label": test_vote_preds
        })
        test_submit_vote.to_csv(os.path.join(output_dir, "submit_test_hard_vote.csv"), index=False)

        # --- Soft Vote (Average Prob) cho TEST ---
        test_prob_cols = [c for c in test_dfs[0].columns if c.startswith("prob_")]
        test_all_probs = np.stack([df[test_prob_cols].values for df in test_dfs])  # shape = (k, N, C)
        test_avg_probs = test_all_probs.mean(axis=0)  # (N, C)

        test_avg_pred_ids = test_avg_probs.argmax(axis=1)
        test_avg_pred_labels = [id2label[i] for i in test_avg_pred_ids]

        # File có cả xác suất
        test_submit_avg = pd.DataFrame({
            "id": test_ids,
            "predict_label": test_avg_pred_labels
        })
        for j, col in enumerate(test_prob_cols):
            test_submit_avg[col] = test_avg_probs[:, j]
        test_submit_avg.to_csv(os.path.join(output_dir, "submit_test_soft_vote.csv"), index=False)

        # File chỉ label
        test_submit_avg_label = test_submit_avg[["id", "predict_label"]]
        test_submit_avg_label.to_csv(os.path.join(output_dir, "submit_test_soft_vote_label.csv"), index=False)
        
        print(f"✅ Test ensemble done: hard vote, soft vote, soft vote label")
    
    # ===== XỬ LÝ DEV PREDICTIONS (OOF) =====
    dev_files = []
    for fold_dir in fold_dirs:
        dev_file = os.path.join(fold_dir, "dev_predictions_with_probs.csv")
        if os.path.exists(dev_file):
            dev_files.append(dev_file)
        else:
            print(f"Warning: {dev_file} not found")
    
    if dev_files:
        print(f"Processing {len(dev_files)} dev prediction files for OOF...")
        dev_dfs = [pd.read_csv(f) for f in dev_files]
        
        # Gộp tất cả OOF predictions lại
        all_oof_predictions = []
        for df in dev_dfs:
            all_oof_predictions.append(df)
        
        # Concatenate tất cả OOF predictions
        train_with_oof = pd.concat(all_oof_predictions, ignore_index=True)
        
        # Sắp xếp lại theo id để đảm bảo thứ tự
        train_with_oof = train_with_oof.sort_values('id').reset_index(drop=True)
        
        # Lưu file train hoàn chỉnh với OOF predictions
        train_with_oof.to_csv(os.path.join(output_dir, "train_with_oof_predictions.csv"), index=False)
        
        print(f"✅ OOF predictions done: train_with_oof_predictions.csv")
        print(f"   Total samples: {len(train_with_oof)}")
        print(f"   Columns: {list(train_with_oof.columns)}")
    
    print(f"✅ Ensemble completed for {output_dir}")


def compute_metrics(eval_pred):
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids
    
    # Nếu predictions là tuple (ví dụ (logits, ...)) thì lấy phần đầu
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    preds = np.argmax(predictions, axis=-1)

    accuracy = accuracy_metric.compute(predictions=preds, references=labels)
    f1 = f1_metric.compute(predictions=preds, references=labels, average="macro")

    return {
        "accuracy": accuracy["accuracy"],
        "f1": f1["f1"],
    }

class StopOnValidLossCallback(TrainerCallback):
    def __init__(self, threshold=0.4):  # ví dụ threshold loss = 0.4
        self.threshold = threshold

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None and "eval_loss" in metrics:
            loss = metrics["eval_loss"]
            if loss < self.threshold:
                print(f"\n>>> Early stopping triggered: eval_loss {loss:.4f} (< {self.threshold}) <<<")
                control.should_training_stop = True
        return control


class StopOnF1Callback(TrainerCallback):
    def __init__(self, threshold=0.9):  # ví dụ F1 >= 0.9 thì dừng
        self.threshold = threshold

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None and "eval_f1" in metrics:  # chú ý tên metric
            f1 = metrics["eval_f1"]
            if f1 >= self.threshold:
                print(f"\n>>> Early stopping triggered: eval_f1 {f1:.4f} (>= {self.threshold}) <<<")
                control.should_training_stop = True
        return control
