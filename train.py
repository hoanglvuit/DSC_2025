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
        print(f"\n=== FOLD {fold_idx + 1}/{config['folds']} ===")
    
        # Clear GPU memory trước khi load model mới
        if 'model' in locals():
            del model
        torch.cuda.empty_cache()
        
        train_split = train_dataset.select(train_idx)
        dev_split = train_dataset.select(val_idx)

        # Áp dụng tokenize lên cả hai phần
        print(f"Use {config['max_length']} tokens")
        print(config["use_prompt"])
        encoded_train_dataset = train_split.map(
            preprocess_and_tokenize,
            batched=True,
            fn_kwargs={"max_length": config["max_length"], "use_prompt": config["use_prompt"], "tokenizer": tokenizer, "lang": "en"}
        )
    
        encoded_dev_dataset = dev_split.map(
            preprocess_and_tokenize,
            batched=True,
            fn_kwargs={"max_length": config["max_length"], "use_prompt": config["use_prompt"], "tokenizer": tokenizer, "lang": "en"}
        )
    
        encoded_test_dataset = pubtest_dataset.map(
            preprocess_and_tokenize,
            batched=True,
            fn_kwargs={"max_length": config["max_length"], "use_prompt": config["use_prompt"], "tokenizer": tokenizer, "lang": "en"}
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
        
        # training 
        training_args = TrainingArguments(
            output_dir=f"./results/model_{config['model_name']}_{fold_idx}",
            save_strategy="steps",
            save_steps=config["save_steps"],               
            save_total_limit=1,
            eval_strategy="steps",  
            eval_steps=100,
            learning_rate=config["learning_rate"],           
            per_device_train_batch_size=config["per_device_train_batch_size"],
            per_device_eval_batch_size=config["per_device_eval_batch_size"],
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            num_train_epochs=config["num_train_epochs"],           
            seed = config["seed"],       
            logging_dir=f"./logs_{config['model_name']}_{fold_idx}",
            logging_steps=100,
            logging_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model='f1',
            greater_is_better=True,
            warmup_ratio=0.1, 
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
        output_dir = f"./output_{config['model_name']}_{fold_idx}"
        os.makedirs(output_dir, exist_ok=True)
        evaluate_dev(trainer,encoded_dev_dataset, output_dir, id2label)
        evaluate_test(trainer,encoded_test_dataset,fold_idx, id2label, output_dir)
        del trainer, model
        torch.cuda.empty_cache()
        if config["ensemble"] == False: 
            break 

    if config["ensemble"]: 
        ensemble_submissions(id2label=id2label, output_dir=output_dir)

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
    parser.add_argument("--model_name", type=str, default="MoritzLaurer/DeBERTa-v3-base-mnli")
    parser.add_argument("--num_class", type=int, default=3)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=22520465)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--use_prompt", type=str, default="no")
    parser.add_argument("--gradient_checkpoint", type=str2bool, default=False)
    parser.add_argument("--ensemble", type=str2bool, default=False)
    parser.add_argument("--claim_model", type=str2bool, default=False)
    parser.add_argument("--train_path", type=str, default="data/vihallu-train-translated-fullen.xlsx")
    parser.add_argument("--public_test_path", type=str, default="data/vihallu-pubtest-translated-fullen.xlsx")
    parser.add_argument("--segment", type=str2bool, default=False)
    parser.add_argument("--intrinsic", type=int, default=2)
    parser.add_argument("--extrinsic", type=int, default=1)
    parser.add_argument("--no", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=0.00001)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=100)
    args = parser.parse_args()
    config = vars(args)
    train_dataset, pubtest_dataset, tokenizer, id2label = preparing_dataset(config["train_path"], config["public_test_path"], config["segment"], config["intrinsic"], config["extrinsic"], config["no"], config["model_name"])
    ensemble_training(train_dataset, pubtest_dataset, tokenizer, id2label, config)
