pip install -r requirements.txt

# Gán biến điều khiển (true hoặc false)
TRANSLATE=true

# Kiểm tra điều kiện trước khi chạy
if [ "$TRANSLATE" = true ]; then
    echo "=== Đang dịch dữ liệu ==="
    python translate_data.py
else
    echo "=== Bỏ qua bước dịch dữ liệu ==="
fi

# train deberta
python train.py \
    --model_name "microsoft/deberta-xlarge-mnli" \
    --max_length 512 \
    --use_prompt "no" \
    --claim_model False \
    --num_train_epochs 1 \
    --segment False \
    --intrinsic 0 \
    --extrinsic 1 \
    --no 2 \
    --learning_rate 0.00001 \
    --gradient_checkpoint True \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --save_strategy "no" \
    --lang "en" \
    --start_fold 0 \
    --end_fold 5

# train cross 
pip install transformers==4.52.4
python train.py --model_name "cross-encoder/nli-deberta-v3-large" \
    --max_length 512 \
    --use_prompt 'no' \
    --claim_model False \
    --num_train_epochs 2 \
    --segment False \
    --intrinsic 0 --extrinsic 2 --no 1 \
    --learning_rate 0.00001  \
    --gradient_checkpoint True --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --gradient_accumulation_steps 1 \
    --save_strategy 'no' \
    --lang "en" --start_fold 0 --end_fold 5

# train dvt 
python train.py --model_name "dangvantuan/vietnamese-document-embedding" \
    --max_length 512 \
    --use_prompt 'no' \
    --claim_model False \
    --num_train_epochs 4 \
    --segment False \
    --intrinsic 0 --extrinsic 1 --no 2 \
    --learning_rate 0.00001  \
    --gradient_checkpoint False --per_device_train_batch_size 8 --per_device_eval_batch_size 16 --gradient_accumulation_steps 2 \
    --save_strategy 'no' \
    --lang "vi" --start_fold 0 --end_fold 5

# train roberta 
python train.py --model_name "FacebookAI/roberta-large-mnli" \
    --max_length 512 \
    --use_prompt 'no' \
    --claim_model False \
    --num_train_epochs 2 \
    --segment False \
    --intrinsic 0 --extrinsic 1 --no 2 \
    --learning_rate 0.00001  \
    --gradient_checkpoint False --per_device_train_batch_size 8 --per_device_eval_batch_size 16 --gradient_accumulation_steps 2 \
    --save_strategy 'no' \
    --lang "en" --start_fold 0 --end_fold 5


pip install --upgrade transformers==4.42.3 peft==0.11.1 datasets==2.20.0 accelerate==0.32.1
pip install semviqa --no-deps

# train erniem 
python train.py --model_name "SemViQA/tc-erniem-viwikifc" \
    --max_length 512 \
    --use_prompt 'no' \
    --claim_model True \
    --num_train_epochs 3 \
    --segment False \
    --intrinsic 2 --extrinsic 0 --no 1 \
    --learning_rate 0.00001  \
    --gradient_checkpoint False --per_device_train_batch_size 4 --per_device_eval_batch_size 8 --gradient_accumulation_steps 4 \
    --save_strategy 'no' \
    --lang "vi" --start_fold 0  --end_fold 5

# train xlm-roberta 
python train.py --model_name "SemViQA/tc-xlmr-isedsc01" \
    --max_length 512 \
    --use_prompt 'no' \
    --claim_model True \
    --num_train_epochs 3 \
    --segment False \
    --intrinsic 2 --extrinsic 0 --no 1 \
    --learning_rate 0.00001  \
    --gradient_checkpoint False --per_device_train_batch_size 4 --per_device_eval_batch_size 16 --gradient_accumulation_steps 4 \
    --save_strategy 'no' \
    --lang "vi" --start_fold 0  --end_fold 5

# stack ensemble
python stack_ensemble.py