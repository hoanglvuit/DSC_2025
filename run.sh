python train.py --model_name "microsoft/deberta-xlarge-mnli" \
--num_class 3 --folds 10 --seed 42 --max_length 512 \
--use_prompt 'no' --gradient_checkpoint True --ensemble False \
--claim_model False --train_path "data/train_dsc.csv" \
--public_test_path "data/public_test.csv" --segment False \
--intrinsic 0 --extrinsic 1 --no 2 --learning_rate 0.00001  \
--per_device_train_batch_size 16 --per_device_eval_batch_size 16 --gradient_accumulation_steps 1 --save_steps 10000 \
--num_train_epochs 3
