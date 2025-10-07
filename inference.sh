pip install -r requirements.txt

# Gán biến điều khiển (true hoặc false), giữ nguyên nếu đã có biến môi trường TRANSLATE
TRANSLATE=${TRANSLATE:-true}

# Kiểm tra điều kiện trước khi chạy
if [ "$TRANSLATE" = true ]; then
    echo "=== Đang dịch dữ liệu ==="
    python translate_data.py
else
    echo "=== Bỏ qua bước dịch dữ liệu ==="
fi

# train deberta
for fold in {0..5}
do
    echo "Running inference for fold $fold..."
    python inference.py \
        --model_name "microsoft/deberta-xlarge-mnli" \
        --model_path "results/model_microsoft_deberta-xlarge-mnli/fold_${fold}" \
        --max_length 512 \
        --use_prompt "no" \
        --claim_model False \
        --segment False \
        --intrinsic 0 \
        --extrinsic 1 \
        --no 2 \
        --lang "en" \
        --fold ${fold}
done

# train cross 
pip install transformers==4.52.4
for fold in {0..5}
do
    echo "Running inference for fold $fold..."
    python inference.py \
        --model_name "cross-encoder/nli-deberta-v3-large" \
        --model_path "results/model_cross-encoder_nli-deberta-v3-large/fold_${fold}" \
        --max_length 512 \
        --use_prompt 'no' \
        --claim_model False \
        --segment False \
        --intrinsic 0 --extrinsic 2 --no 1 \
        --lang "en" --fold ${fold}
done


# train dvt 
for fold in {0..5}
do
    echo "Running inference for fold $fold..."
    python inference.py \
        --model_name "dangvantuan/vietnamese-document-embedding" \
        --model_path "results/model_dangvantuan_vietnamese-document-embedding/fold_${fold}" \
    --max_length 512 \
    --use_prompt 'no' \
    --claim_model False \
    --segment False \
    --intrinsic 0 --extrinsic 1 --no 2 \
    --lang "vi" --fold ${fold}
done

# train roberta 
for fold in {0..5}
do
    echo "Running inference for fold $fold..."
    python inference.py \
        --model_name "FacebookAI/roberta-large-mnli" \
        --model_path "results/model_FacebookAI_roberta-large-mnli/fold_${fold}" \
    --max_length 512 \
    --use_prompt 'no' \
    --claim_model False \
    --segment False \
    --intrinsic 0 --extrinsic 1 --no 2 \
    --lang "en" --fold ${fold}
done


pip install --upgrade transformers==4.42.3 peft==0.11.1 datasets==2.20.0 accelerate==0.32.1
pip install semviqa --no-deps

# train erniem 
for fold in {0..5}
do
    echo "Running inference for fold $fold..."
    python inference.py \
        --model_name "SemViQA/tc-erniem-viwikifc" \
        --model_path "results/model_SemViQA_tc-erniem-viwikifc/fold_${fold}" \
    --max_length 512 \
    --use_prompt 'no' \
    --claim_model True \
    --segment False \
    --intrinsic 2 --extrinsic 0 --no 1 \
    --lang "vi" --fold ${fold}
done

# train xlm-roberta 
for fold in {0..5}
do
    echo "Running inference for fold $fold..."
    python inference.py \
        --model_name "SemViQA/tc-xlmr-isedsc01" \
        --model_path "results/model_SemViQA_tc-xlmr-isedsc01/fold_${fold}" \
    --max_length 512 \
    --use_prompt 'no' \
    --claim_model True \
    --segment False \
    --intrinsic 2 --extrinsic 0 --no 1 \
    --lang "vi" --fold ${fold}
done

# stack ensemble
python stack_ensemble.py --root_folder "output" --use_true False