from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from utils import *
import argparse
import os 
from transformers.integrations import WandbCallback
os.environ["WANDB_DISABLED"] = "true"

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
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--claim_model", type=str2bool)
    parser.add_argument("--lang", type=str)
    parser.add_argument("--max_length", type=int)
    parser.add_argument("--use_prompt", type=str)
    parser.add_argument("--segment", type=str2bool)
    parser.add_argument("--intrinsic", type=int)
    parser.add_argument("--extrinsic", type=int)
    parser.add_argument("--no", type=int)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--fold", type=int)
    args = parser.parse_args()
    config = vars(args)

    if config["claim_model"]:
        from semviqa.tvc.model import ClaimModelForClassification
        model = ClaimModelForClassification.from_pretrained(config["model_path"])
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            config["model_path"],
            trust_remote_code=True
        )
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    trainer = Trainer(model=model, tokenizer=tokenizer)
    trainer.remove_callback(WandbCallback)

    train_dataset, pubtest_dataset, privatetest_dataset, tokenizer, id2label = preparing_dataset("data/train_dsc.csv", "data/public_test.csv", "data/private_test.csv", config["segment"], config["intrinsic"], config["extrinsic"], config["no"], config["model_name"])
    encoded_privatetest_dataset = privatetest_dataset.map(
        preprocess_and_tokenize,
        batched=True,
        fn_kwargs={"max_length": config["max_length"], "use_prompt": config["use_prompt"], "tokenizer": tokenizer, "lang": config["lang"]}
    )
    safe_model_name = config['model_name'].replace("/", "_")
    output_dir = f"output/{safe_model_name}/fold_{config['fold']}"
    os.makedirs(output_dir, exist_ok=True)
    evaluate_pritest(trainer,encoded_privatetest_dataset, id2label, output_dir)