from utils import *
import os 
import argparse
from sklearn.model_selection import StratifiedKFold
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
import numpy as np
# os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ["WANDB_DISABLED"] = "true"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def ensemble_training(train_dataset, pubtest_dataset, privatetest_dataset, tokenizer, id2label, config):
    folds = list(range(config["start_fold"], config["end_fold"]))
    print(f"Training on folds: {folds}")
    labels = np.array(train_dataset["label"])
    skf = StratifiedKFold(n_splits=config["folds"], shuffle=True, random_state=config["seed"])
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_dataset, labels)):
        if fold_idx not in folds:
            continue
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
            fn_kwargs={"max_length": config["max_length"], "use_prompt": config["use_prompt"], "tokenizer": tokenizer, "lang": config["lang"]}
        )
    
        encoded_dev_dataset = dev_split.map(
            preprocess_and_tokenize,
            batched=True,
            fn_kwargs={"max_length": config["max_length"], "use_prompt": config["use_prompt"], "tokenizer": tokenizer, "lang": config["lang"]}
        )
    
        encoded_test_dataset = pubtest_dataset.map(
            preprocess_and_tokenize,
            batched=True,
            fn_kwargs={"max_length": config["max_length"], "use_prompt": config["use_prompt"], "tokenizer": tokenizer, "lang": config["lang"]}
        )

        encoded_privatetest_dataset = privatetest_dataset.map(
            preprocess_and_tokenize,
            batched=True,
            fn_kwargs={"max_length": config["max_length"], "use_prompt": config["use_prompt"], "tokenizer": tokenizer, "lang": config["lang"]}
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
        
        if config['gradient_checkpoint']:
            model.gradient_checkpointing_enable()
        safe_model_name = config['model_name'].replace("/", "_")
        # training 
        training_args = TrainingArguments(
            output_dir=f"./results/model_{safe_model_name}/fold_{fold_idx}",
            save_strategy=config['save_strategy'],
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
            logging_dir=f"./logs/{safe_model_name}/fold_{fold_idx}",
            logging_steps=100,
            logging_strategy="steps",
            load_best_model_at_end=config["load_best_model_at_end"],
            metric_for_best_model='f1',
            greater_is_better=True,
            warmup_ratio=0.1,     
        )
    
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=encoded_train_dataset,
            eval_dataset=encoded_dev_dataset,
            compute_metrics=compute_metrics,
        )
    
        trainer.train()
        trainer.save_model(f"./results/model_{safe_model_name}/fold_{fold_idx}")

        # evaluate
        output_dir = f"./output/{safe_model_name}/fold_{fold_idx}"
        os.makedirs(output_dir, exist_ok=True)
        evaluate_dev(trainer,encoded_dev_dataset, output_dir, id2label)
        evaluate_test(trainer,encoded_test_dataset, id2label, output_dir)
        evaluate_pritest(trainer,encoded_privatetest_dataset, id2label, output_dir)       
        del trainer, model
        torch.cuda.empty_cache()

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
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--num_class", type=int, default=3)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=22520465)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--use_prompt", type=str, default="no")
    parser.add_argument("--gradient_checkpoint", type=str2bool)
    parser.add_argument("--claim_model", type=str2bool)
    parser.add_argument("--train_path", type=str, default="data/train_dsc.csv")
    parser.add_argument("--public_test_path", type=str, default="data/public_test.csv")
    parser.add_argument("--private_test_path", type=str, default="data/private_test.csv")
    parser.add_argument("--segment", type=str2bool)
    parser.add_argument("--intrinsic", type=int)
    parser.add_argument("--extrinsic", type=int)
    parser.add_argument("--no", type=int)
    parser.add_argument("--learning_rate", type=float, default=0.00001)
    parser.add_argument("--per_device_train_batch_size", type=int)
    parser.add_argument("--per_device_eval_batch_size", type=int)
    parser.add_argument("--gradient_accumulation_steps", type=int)
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--lang", type=str)
    parser.add_argument("--start_fold", type=int)
    parser.add_argument("--end_fold", type=int)
    parser.add_argument("--save_strategy", type=str, default="no")
    args = parser.parse_args()
    config = vars(args)

    config['load_best_model_at_end'] = True
    if config["save_strategy"] == "no":
        config['load_best_model_at_end'] = False

    seed_everything(config["seed"])
    train_dataset, pubtest_dataset, privatetest_dataset, tokenizer, id2label = preparing_dataset(config["train_path"], config["public_test_path"], config["private_test_path"], config["segment"], config["intrinsic"], config["extrinsic"], config["no"], config["model_name"])
    ensemble_training(train_dataset, pubtest_dataset, privatetest_dataset, tokenizer, id2label, config)
