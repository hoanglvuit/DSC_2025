import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
from utils import seed_everything
import os 

# set seed 
seed_everything(42)

# Load model
model_name = "VietAI/envit5-translation"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda")

# Hàm dịch
def translate_texts(texts, src_lang="vi", tgt_lang="en", max_length=512):
    inputs = [f"{src_lang}: {t}" for t in texts]
    encodings = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to("cuda")

    with torch.no_grad():
        outputs = model.generate(**encodings, max_length=max_length)

    results = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # Xóa tiền tố "en: " hoặc "vi: " nếu có
    cleaned = [r[len(f"{tgt_lang}: "):] if r.startswith(f"{tgt_lang}: ") else r for r in results]

    return cleaned

def translate_data(file_path:str, output_path:str): 
    df = pd.read_excel(file_path)
    df = df.astype(str)

    batch_size = 64
    context_en_list = []
    response_en_list = []
    prompt_en_list = []

    df = df.rename(columns={"context": "context_vi", "response": "response_vi", "prompt": "prompt_vi"})
    print(len(df))

    for i in tqdm(range(0, len(df), batch_size)):
        batch_contexts = df["context_vi"].iloc[i:i+batch_size].tolist()
        batch_responses = df["response_vi"].iloc[i:i+batch_size].tolist()
        batch_prompts = df["prompt_vi"].iloc[i:i+batch_size].tolist()
    
        context_en_list.extend(translate_texts(batch_contexts, src_lang="vi", tgt_lang="en"))
        response_en_list.extend(translate_texts(batch_responses, src_lang="vi", tgt_lang="en"))
        prompt_en_list.extend(translate_texts(batch_prompts, src_lang="vi", tgt_lang="en"))


    df["context_en"] = context_en_list
    df["response_en"] = response_en_list
    df["prompt_en"] = prompt_en_list
    
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    translate_data(file_path="ori_data/vihallu-train.xlsx", output_path="data/train_dsc.csv")
    translate_data(file_path="ori_data/vihallu-public-test.xlsx", output_path="data/public_test.csv")
    translate_data(file_path="ori_data/vihallu-private-test.xlsx", output_path="data/private_test.csv")