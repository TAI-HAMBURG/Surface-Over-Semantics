import pandas as pd
import time 
import fire
import logging
import torch
import os
import glob

import requests
from PIL import Image
import copy
from secrets import token_hex
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForVision2Seq
from random import sample


def main(
        # data parameters
        language: str,
        model: str,
        test_data_output_path: str,

        # model parameters
        model_name_or_path: str,

        cache_dir:str,
          
        # quantization parameters
        load_in_8bit: bool,
           
        # misc parameters
        log_level: str,

        sample_size:int

        ):

    ###########################################################
    # SET UP
    ###########################################################

    repo='<REPO_PATH>'

     # set up logging
    logging.basicConfig(level=getattr(logging, log_level.upper()), format='%(asctime)s %(levelname)s %(message)s')

        # set up device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Running on device: {device}")
    if device == "cuda":
        logging.info(f"CUDA memory: {round(torch.cuda.mem_get_info()[0]/1024**3,2)}GB")

    ###########################################################
    # LOAD DATA
    ###########################################################

    # load TEST data
    image_urls = glob.glob(f'{repo}/images/{language}/{model}/*Person*.png')
    if sample_size > 0:
        image_urls = sample(image_urls, sample_size)
    print(len(image_urls))

    #image_urls = image_urls[:10]

    batch_size = 10

    images = []
    for url in image_urls: 
        image = Image.open(url)
        keep = image.copy()
        images.append(keep)
        image.close()



    processor = AutoProcessor.from_pretrained(model_name_or_path, cache_dir=cache_dir)
    model = AutoModelForVision2Seq.from_pretrained(model_name_or_path,
                                                    torch_dtype=torch.float16,
                                                    cache_dir=cache_dir,
                                                    device_map='cuda')#.to(device)


    prompt = ['Create a detailed description of everything that is depicted in this image.'] 


    templated_prompt = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},  # Dynamic image URL
                        {"type": "text", "text": "Describe the image in detail."},
                    ],
                },
            ]


    print(test_data_output_path)


    responses = []
    for i in tqdm(range(0, len(image_urls), batch_size), desc="Processing batches"):
        batch_images = images[i:i + batch_size]  # Images for the current batch
        batch_urls = image_urls[i:i + batch_size]  # URLs for the current batch
        

        input_texts = [processor.apply_chat_template(templated_prompt, tokenize=False, add_generation_prompt=True)]  * len(batch_images)


        inputs = processor(text=input_texts, images=batch_images, return_tensors="pt", padding=True)

        inputs = inputs.to(device)



        # Generate
        output_ids = model.generate(**inputs, max_new_tokens=1024)
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        generated_texts = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        responses.extend(generated_texts)

    prompts= [prompt] * len(images)
    df = pd.DataFrame(dict(prompt=prompts, response=responses, topic=image_urls))

    # write new model completions to new column
    df["model_name"] = model_name_or_path

    # check if output path exists, otherwise create it
    if not os.path.exists(test_data_output_path.rsplit("/", 1)[0]):
        logging.info(f"Creating new path {test_data_output_path.rsplit('/', 1)[0]}")
        os.makedirs(test_data_output_path.rsplit("/", 1)[0])

    logging.info(f"Saving completions to {test_data_output_path}")
    df.to_csv(test_data_output_path, index=False)


if __name__ == "__main__":
    st = time.time()
    fire.Fire(main)
    logging.info(f'Total execution time: {time.time() - st:.2f} seconds')











