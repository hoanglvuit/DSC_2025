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


def seed_everything(seed=22520465):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def preparing_dataset(train_path, public_test_path, segment, intrinsic, extrinsic, no, model_name): 
    train_dataset = pd.read_excel(train_path)
    pubtest_dataset = pd.read_excel(public_test_path)

    # Convert data to string
    train_dataset = train_dataset.astype(str)
    pubtest_dataset = pubtest_dataset.astype(str)

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
        
        for col in ["context_en", "prompt_en", "response_en"]:
            if col in train_dataset.columns:
                train_dataset[col] = train_dataset[col].apply(segment_text)
            if col in pubtest_dataset.columns:
                pubtest_dataset[col] = pubtest_dataset[col].apply(segment_text)
        print(train_dataset['context_en'].head(3))
        print(pubtest_dataset[["prompt_en", "response_en"]].head(3))

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
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False,trust_remote_code=True)

    # convert to Dataset format
    train_dataset = Dataset.from_pandas(train_dataset)
    pubtest_dataset = Dataset.from_pandas(pubtest_dataset)
    return train_dataset, pubtest_dataset, tokenizer, id2label


def preprocess_and_tokenize(batch, max_length, use_prompt, tokenizer):
    if use_prompt == "first":
        first = [p + " " + c for p, c in zip(batch["prompt_en"], batch["context_en"])]
        second = batch['response_en']
    elif use_prompt == "second": 
        first = batch['context_en']
        second = [p + " " + r for p, r in zip(batch["prompt_en"], batch["response_en"])]
    else:
        first = batch['context_en']
        second = batch['response_en']

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


def evaluate_dev(trainer, encoded_dev_dataset, output_dir): 
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

        # Optionally, lưu các index đúng/sai (nếu cần phân tích sâu hơn)
        f.write("\nIndices of wrong predictions:\n")
        wrong_indices = np.where(mask_wrong)[0]
        f.write(", ".join(map(str, wrong_indices)))
        f.write("\n")

    print(f"✅ Evaluation results saved in {output_dir}")


def evaluate_test(trainer, encoded_test_dataset, fold_idx, id2label, output_dir): 
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
    submit_df.to_csv(os.path.join(output_dir, f"submit_{fold_idx}.csv"), index=False)
    
    # File 2: thêm xác suất 3 class
    probs_df = pd.DataFrame(probs, columns=[f"prob_{id2label[i]}" for i in range(probs.shape[1])])
    submit_with_probs = pd.concat([submit_df, probs_df], axis=1)
    submit_with_probs.to_csv(os.path.join(output_dir, f"submit_with_probs_{fold_idx}.csv"), index=False)
    
    # Số mẫu mỗi class
    print("Class counts on test:")
    print(submit_df["predict_label"].value_counts())
    
    submit_file = os.path.join(output_dir, f'submit_{fold_idx}.csv')
    submit_probs_file = os.path.join(output_dir, f'submit_with_probs_{fold_idx}.csv')
    print(f"Done. Files saved: {submit_file}, {submit_probs_file}")


def ensemble_submissions(id2label, output_dir):
    """
    Tạo 3 file từ các file submit_with_probs trong folder:
    1. submit_vote.csv (majority vote, chỉ label)
    2. submit_avg.csv (average prob, có cả xác suất)
    3. submit_avg_label.csv (average prob, chỉ label)
    
    Args:
        folder (str): path to folder containing submit_with_probs_*.csv
        id2label (dict): mapping id->label nếu muốn (nếu None thì dùng cột 'predict_label' đầu tiên)
    """
    files = sorted(glob.glob(os.path.join(output_dir, "submit_with_probs_*.csv")))
    if not files:
        raise ValueError("Cannot find submit_with_probs_*.csv in folder!")

    dfs = [pd.read_csv(f) for f in files]
    print(f"Found {len(dfs)} files")
    # Lấy id (giả sử id giống nhau và cùng thứ tự trong tất cả folds)
    ids = dfs[0]["id"]

    # --- Majority Vote ---
    vote_preds = []
    for i in range(len(ids)):
        labels_i = [df.loc[i, "predict_label"] for df in dfs]
        vote = pd.Series(labels_i).mode()[0]  
        vote_preds.append(vote)

    submit_vote = pd.DataFrame({
        "id": ids,
        "predict_label": vote_preds
    })
    submit_vote.to_csv(os.path.join(output_dir, "submit_vote.csv"), index=False)

    # --- Average Prob ---
    prob_cols = [c for c in dfs[0].columns if c.startswith("prob_")]
    all_probs = np.stack([df[prob_cols].values for df in dfs])  # shape = (k, N, C)
    avg_probs = all_probs.mean(axis=0)  # (N, C)

    avg_pred_ids = avg_probs.argmax(axis=1)

    # If id2label is provided, use it, otherwise map the column names
    if id2label:
        avg_pred_labels = [id2label[i] for i in avg_pred_ids]
    else:
        label_names = [c.replace("prob_", "") for c in prob_cols]
        avg_pred_labels = [label_names[i] for i in avg_pred_ids]

    # File có cả xác suất
    submit_avg = pd.DataFrame({
        "id": ids,
        "predict_label": avg_pred_labels
    })
    for j, col in enumerate(prob_cols):
        submit_avg[col] = avg_probs[:, j]
    submit_avg.to_csv(os.path.join(output_dir, "submit_avg.csv"), index=False)

    # File chỉ label
    submit_avg_label = submit_avg[["id", "predict_label"]]
    submit_avg_label.to_csv(os.path.join(output_dir, "submit_avg_label.csv"), index=False)

    print(f"✅ Done. Đã tạo 3 file: {os.path.join(output_dir, 'submit_vote.csv')}, {os.path.join(output_dir, 'submit_avg.csv')}, {os.path.join(output_dir, 'submit_avg_label.csv')}")


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
