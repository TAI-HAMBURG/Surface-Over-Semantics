#!/bin/sh


# check python version
python3 --version

export CUDA_VISIBLE_DEVICES=2

# store repo path
REPO='<REPO_PATH>'

for MODEL in "blackforest"; do # "sd2.1" "kandinsky-2-1" "kandinsky-3"  "stable-diffusion-3" "blackforest" 'sdxl' "altdiffusion-m9"

    for EXPERIMENT in "full_dataset"; do

        python3 generate_images.py \
            --model_name $MODEL\
            --test_data_input_path $EXPERIMENT.csv \
            --input_col "prompt" \
            --test_data_output_path $REPO/cultural_bias/images/$EXPERIMENT/$MODEL/\
            --cache_dir "$REPO/cache_dir"

    done;


done;