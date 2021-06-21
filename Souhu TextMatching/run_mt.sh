export CUDA_VISIBLE_DEVICES=4

nohup python3 access_port_multi_task.py \
    --model_type mac-bert \
    --train_mode average \
    --output_dir output_fl_mac_bert_average_mt \
    --learning_rate 2e-5 \
    --per_gpu_train_batch_size 8 \
    --use_focalloss \
    --save_steps 9000 \
    --num_train_epochs 3. \
    --evaluate_during_training \
    --seed 42 > logs/output_fl_mac_bert_average_mt.log 2>&1 &&
    
nohup python3 access_port_multi_task.py \
    --model_type roberta-wwm-ext-large \
    --train_mode cls \
    --output_dir output_fl_roberta-wwm-ext-large_cls_mt \
    --learning_rate 2e-5 \
    --per_gpu_train_batch_size 8 \
    --use_focalloss \
    --save_steps 9000 \
    --num_train_epochs 3. \
    --evaluate_during_training \
    --seed 100 > logs/output_fl_roberta-wwm-ext-large_cls_mt_100.log 2>&1 &&
    
nohup python3 access_port_multi_task.py \
    --model_type roberta-wwm-ext-large \
    --train_mode average \
    --output_dir output_fl_roberta-wwm-ext-large_average_mt \
    --learning_rate 2e-5 \
    --per_gpu_train_batch_size 8 \
    --use_focalloss \
    --save_steps 9000 \
    --num_train_epochs 3. \
    --evaluate_during_training \
    --seed 100 > logs/output_fl_roberta-wwm-ext-large_average_mt_100.log 2>&1 &&
    
nohup python3 access_port_multi_task.py \
    --model_type mac-bert \
    --train_mode cls \
    --output_dir output_fl_mac_bert_cls_mt \
    --learning_rate 2e-5 \
    --per_gpu_train_batch_size 8 \
    --use_focalloss \
    --save_steps 9000 \
    --num_train_epochs 3. \
    --evaluate_during_training \
    --seed 100 > logs/output_fl_mac_bert_cls_mt_100.log 2>&1 &&
    
nohup python3 access_port_multi_task.py \
    --model_type mac-bert \
    --train_mode average \
    --output_dir output_fl_mac_bert_average_mt \
    --learning_rate 2e-5 \
    --per_gpu_train_batch_size 8 \
    --use_focalloss \
    --save_steps 9000 \
    --num_train_epochs 3. \
    --evaluate_during_training \
    --seed 100 > logs/output_fl_mac_bert_average_mt_100.log 2>&1 &