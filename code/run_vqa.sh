#!/bin/sh


# check python version
python3 --version

# to surpress annoyingly verbose warning
export TOKENIZERS_PARALLELISM=true
export CUDA_VISIBLE_DEVICES=3

# store repo path
REPO="<REPO-PATH>"

# set params
PROVIDER=Qwen 
MODEL_NAME=Qwen2-VL-72B-Instruct

for LANGUAGE in "full_russian"; do 

    for MODEL in  "stable-diffusion-3" "sdxl" "sd2.1" "kandinsky-3" "kandinsky-2-1" "altdiffusion-m9" "blackforest"; do 

        python3 vqa_image_descriptions.py \
            --model_name_or_path $PROVIDER/$MODEL_NAME \
            --language $LANGUAGE \
            --model $MODEL \
            --test_data_output_path $REPO/vqa_results/$LANGUAGE/$MODEL/$MODEL_NAME.csv \
            --load_in_8bit False \
            --log_level "error" \
            --cache_dir "$REPO/cache_dir" \
            --sample_size 0

    done;

done;