CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes=2 mates/data_selection/prob_oracle_data_influence.py \
    --model_dir ./out \
    --model_name "llama3-1B" \
    --ckpt 1000 \
    --data_dir ./data/train/processed/ \
    --train_files cot/cot_data.jsonl \
    --ref_data_dir ./data \
    --reference_files cot/cot_data.jsonl \
    --output_dir ./out_influence \
    --task "bbh"
   