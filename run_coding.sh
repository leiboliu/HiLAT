#!/bin/bash
export XRT_TPU_CONFIG="tpu_worker;0;{TPU_IP}:8470"
export PYTHONPATH="${PYTHONPATH}:."

python ./hilat/xla_spawn.py \
   --num_cores 8 \
   ./lwat/run_coding.py \
   --model_name_or_path  ./pretrained/mimic3_total_64/ \
   --tokenizer_name xlnet-base-cased \
   --output_dir ./model/ \
   --overwrite_output_dir true \
   --task_name mimic3-50 \
   --max_seq_length 512 \
   --train_file ./data/mimic3/50/train_data_50_level_1_rand_reordered.csv \
   --validation_file ./data/mimic3/50/dev_data_50_level_1_rand_reordered.csv \
   --test_file ./data/mimic3/50/test_data_50_level_1_rand_reordered.csv \
   --label_dictionary_file ./data/mimic3/50/labels_dictionary_50_level_1.csv \
   --code_max_seq_length 32 \
   --code_batch_size 16 \
   --ignore_keys_for_eval preds label_attention_weights chunk_attention_weights \
   --use_cached_datasets true \
   --d_model 768 \
   --dropout 0.1 \
   --dropout_att 0.1 \
   --num_chunks_per_document 10 \
   --transformer_layer_update_strategy all \
   --use_code_representation false \
   --multi_head_attention false \
   --chunk_attention true \
   --document_pooling_strategy flat \
   --multi_head_chunk_attention false \
   --linear_init_mean 0.0 \
   --linear_init_std 0.03 \
   --do_train true \
   --do_eval true \
   --do_predict true \
   --evaluation_strategy steps \
   --per_device_train_batch_size 2 \
   --per_device_eval_batch_size 2 \
   --gradient_accumulation_steps 1 \
   --learning_rate 5e-5 \
   --weight_decay 0.1 \
   --num_train_epochs 100 \
   --max_steps 2500 \
   --warmup_steps 500 \
   --log_level info \
   --logging_strategy steps \
   --logging_steps 500 \
   --save_strategy steps \
   --save_steps 500 \
   --seed 2022 \
   --dataloader_drop_last false \
   --disable_tqdm false \
   --label_names targets \
   --load_best_model_at_end true \
   --metric_for_best_model micro_f1 \
   --greater_is_better true \
   --remove_unused_columns false
