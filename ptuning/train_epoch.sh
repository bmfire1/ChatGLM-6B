PRE_SEQ_LEN=128
#LR=2e-2
LR=1e-4
epochs=3

dataset_path=/home/ubuntu/workspace/dataset/AdvertiseGen
original_model_dir=/home/ubuntu/workspace/models/GLM-6B
output=/home/ubuntu/workspace/models/GLM-pturning

CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --do_train \
    --train_file $dataset_path/train.json \
    --validation_file $dataset_path/dev.json \
    --prompt_column content \
    --response_column summary \
    --overwrite_cache \
    --model_name_or_path $original_model_dir \
    --output_dir $output/adgen-chatglm-6b-pt-$epochs-$PRE_SEQ_LEN-$LR \
    --overwrite_output_dir \
    --max_source_length 64 \
    --max_target_length 64 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --num_train_epochs $epochs \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 8

