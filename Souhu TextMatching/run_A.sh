export CUDA_VISIBLE_DEVICES=6

nohup python3 access_port.py \
    --model_type roberta-wwm-ext-large \
    --train_mode cls \
    --task_name A \
    --output_dir A2.output_fl_roberta_cls_ema_fgm \
    --learning_rate 2e-5 \
    --per_gpu_train_batch_size 8 \
    --use_focalloss \
    --use_ema \
    --use_fgm \
    --save_steps 3000 \
    --num_train_epochs 2. \
    --evaluate_during_training \
    --seed 42 > logs/A2.output_fl_roberta_cls_ema_fgm.log 2>&1 &&
    
nohup python3 access_port.py \
    --model_type roberta-wwm-ext-large \
    --train_mode average \
    --task_name A \
    --output_dir A2.output_fl_roberta_average_ema_fgm  \
    --learning_rate 2e-5 \
    --per_gpu_train_batch_size 8 \
    --use_focalloss \
    --use_ema \
    --use_fgm \
    --save_steps 3000 \
    --num_train_epochs 2. \
    --evaluate_during_training \
    --seed 42 > logs/A2.output_fl_roberta_average_ema_fgm.log 2>&1 &&
    
nohup python3 access_port.py \
    --model_type roberta-wwm-ext-large \
    --train_mode bert_cls_avg \
    --task_name A \
    --output_dir A2.output_fl_roberta_cls_avg_ema_fgm  \
    --learning_rate 2e-5 \
    --per_gpu_train_batch_size 8 \
    --use_focalloss \
    --use_ema \
    --use_fgm \
    --save_steps 3000 \
    --num_train_epochs 2. \
    --evaluate_during_training \
    --seed 42 > logs/A2.output_fl_roberta_cls_avg_ema_fgm.log 2>&1 &&

nohup python3 access_port.py \
    --model_type mac-bert \
    --train_mode cls \
    --task_name A \
    --output_dir A3.output_fl_mac-bert_cls_ema_fgm \
    --learning_rate 2e-5 \
    --per_gpu_train_batch_size 8 \
    --use_focalloss \
    --use_ema \
    --use_fgm \
    --save_steps 3000 \
    --num_train_epochs 2. \
    --evaluate_during_training \
    --seed 42 > logs/A3.output_fl_mac-bert_cls_ema_fgm.log 2>&1 &&
    
nohup python3 access_port.py \
    --model_type mac-bert \
    --train_mode average \
    --task_name A \
    --output_dir A3.output_fl_mac-bert_average_ema_fgm  \
    --learning_rate 2e-5 \
    --per_gpu_train_batch_size 8 \
    --use_focalloss \
    --use_ema \
    --use_fgm \
    --save_steps 3000 \
    --num_train_epochs 2. \
    --evaluate_during_training \
    --seed 42 > logs/A3.output_fl_mac-bert_average_ema_fgm.log 2>&1 &&
    
nohup python3 access_port.py \
    --model_type mac-bert \
    --train_mode bert_cls_avg \
    --task_name A \
    --output_dir A3.output_fl_mac-bert_cls_avg_ema_fgm  \
    --learning_rate 2e-5 \
    --per_gpu_train_batch_size 8 \
    --use_focalloss \
    --use_ema \
    --use_fgm \
    --save_steps 3000 \
    --num_train_epochs 2. \
    --evaluate_during_training \
    --seed 42 > logs/A3.output_fl_mac-bert_cls_avg_ema_fgm.log 2>&1 &
