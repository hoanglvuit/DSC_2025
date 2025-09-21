from utils import *
import os 
import argparse
from sklearn.model_selection import StratifiedKFold
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
import numpy as np
seed_everything()
os.environ["WANDB_DISABLED"] = "true"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def ensemble_training(train_dataset, pubtest_dataset, tokenizer, id2label, config):
    labels = np.array(train_dataset["label"])
    skf = StratifiedKFold(n_splits=config["folds"], shuffle=True, random_state=config["seed"])
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_dataset, labels)):
        print(f"\n=== FOLD {fold_idx + 1}/{config["folds"]} ===")
    
        # Clear GPU memory trước khi load model mới
        if 'model' in locals():
            del model
        torch.cuda.empty_cache()
        
        train_split = train_dataset.select(train_idx)
        dev_split = train_dataset.select(val_idx)

        # Áp dụng tokenize lên cả hai phần
        print(f"Use {config["max_length"]} tokens")
        print(config["use_prompt"])
        encoded_train_dataset = train_split.map(
            preprocess_and_tokenize,
            batched=True,
            fn_kwargs={"max_length": config["max_length"], "use_prompt": config["use_prompt"], "tokenizer": tokenizer}
        )
    
        encoded_dev_dataset = dev_split.map(
            preprocess_and_tokenize,
            batched=True,
            fn_kwargs={"max_length": config["max_length"], "use_prompt": config["use_prompt"], "tokenizer": tokenizer}
        )
    
        encoded_test_dataset = pubtest_dataset.map(
            preprocess_and_tokenize,
            batched=True,
            fn_kwargs={"max_length": config["max_length"], "use_prompt": config["use_prompt"], "tokenizer": tokenizer}
        )

        # load model 
        if config["claim_model"]: 
            from semviqa.tvc.model import ClaimModelForClassification
            model = ClaimModelForClassification.from_pretrained(config["model_name"])
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
            config["model_name"], 
            num_labels=config["num_class"],
            ignore_mismatched_sizes=True,
            trust_remote_code=True)
    
        if config["gradient_checkpoint"]: 
            model.gradient_checkpointing_enable()
        
        # training 
        training_args = TrainingArguments(
            output_dir=f"./results/model_{config["model_name"]}_{fold_idx}",
            save_strategy="steps",
            save_steps=config["save_steps"],               
            save_total_limit=config["save_total_limit"],
            eval_strategy="steps",  
            eval_steps=config["eval_steps"],
            learning_rate=config["learning_rate"],           
            per_device_train_batch_size=config["per_device_train_batch_size"],
            per_device_eval_batch_size=config["per_device_eval_batch_size"],
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            num_train_epochs=config["num_train_epochs"],           
            seed = config["seed"],       
            logging_dir=f"./logs_{config["model_name"]}_{fold_idx}",
            logging_steps=config["logging_steps"],
            logging_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model=config["metric_for_best_model"],
            greater_is_better=True,
            warmup_ratio=config["warmup_ratio"], 
            label_smoothing_factor = config["label_smoothing_factor"], 
            fp16 = config["fp16"],
            gradient_checkpointing=config["gradient_checkpoint"])
    
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=encoded_train_dataset,
            eval_dataset=encoded_dev_dataset,
            compute_metrics=compute_metrics,
        )
    
        trainer.train()

        # evaluate
        output_dir = f"./output_{config["model_name"]}_{fold_idx}"
        os.makedirs(output_dir, exist_ok=True)
        evaluate_dev(trainer,encoded_dev_dataset, output_dir)
        evaluate_test(trainer,encoded_test_dataset,fold_idx, id2label, output_dir)
        del trainer, model
        torch.cuda.empty_cache()
        if config["ensemble"] == False: 
            break 

    if config["ensemble"]: 
        ensemble_submissions(id2label=id2label, output_dir=output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="MoritzLaurer/DeBERTa-v3-base-mnli")
    parser.add_argument("--num_class", type=int, default=3)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=22520465)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--use_prompt", type=bool, default=False)
    parser.add_argument("--gradient_checkpoint", type=bool, default=False)
    parser.add_argument("--ensemble", type=bool, default=False)
    parser.add_argument("--claim_model", type=bool, default=False)
    parser.add_argument("--train_path", type=str, default="data/vihallu-train-translated-fullen.xlsx")
    parser.add_argument("--public_test_path", type=str, default="data/vihallu-pubtest-translated-fullen.xlsx")
    parser.add_argument("--segment", type=bool, default=False)
    parser.add_argument("--intrinsic", type=int, default=2)
    parser.add_argument("--extrinsic", type=int, default=1)
    parser.add_argument("--no", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=0.00001)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--label_smoothing_factor", type=float, default=0)
    parser.add_argument("--metric_for_best_model", type=str, default="f1")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=1)
    parser.add_argument("--fp16", type=bool, default=True)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    args = parser.parse_args()
    config = vars(args)
    train_dataset, pubtest_dataset, tokenizer, id2label = preparing_dataset(config["train_path"], config["public_test_path"], config["segment"], config["intrinsic"], config["extrinsic"], config["no"], config["model_name"])
    ensemble_training(train_dataset, pubtest_dataset, tokenizer, id2label, config)
