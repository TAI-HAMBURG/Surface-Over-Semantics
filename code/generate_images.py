import torch
import os
import pandas as pd
import time 
import fire
import logging
from diffusers import Kandinsky3Pipeline, AutoPipelineForText2Image, DiffusionPipeline, StableDiffusionPipeline, StableDiffusion3Pipeline, AltDiffusionPipeline, DPMSolverMultistepScheduler, FluxPipeline
#from diffusers import AltDiffusionPipeline
#from flagai.auto_model.auto_loader import AutoLoader
#from flagai.model.predictor.predictor import Predictor
from tqdm import tqdm
import glob


def main(
        # data parameters
        test_data_input_path: str,
        input_col: str,
        test_data_output_path: str,

        # model parameters
        cache_dir:str,
        model_name:str

        ):

    ###########################################################
    # LOAD DATA
    ###########################################################

    # load TEST data
    if 'csv' in test_data_input_path:
        test_df = pd.read_csv(test_data_input_path)
    elif 'xlsx' in test_data_input_path:
        test_df = pd.read_excel(test_data_input_path)


    logging.info(f"Loaded TEST data: {test_df.shape[0]} rows")

    seed = 42

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


    if 'kandinsky-3' in model_name:
        print('Loading model Kandinsky-3')
        pipeline = Kandinsky3Pipeline.from_pretrained("kandinsky-community/kandinsky-3", variant="fp16", torch_dtype=torch.float16, cache_dir=cache_dir)
        pipeline = pipeline.to("cuda")
    elif 'kandinsky-2-1' in model_name:
        print('Loading model Kandinsky-2.1')
        pipeline = AutoPipelineForText2Image.from_pretrained("kandinsky-community/kandinsky-2-1", torch_dtype=torch.float16, cache_dir=cache_dir)
        pipeline = pipeline.to("cuda")
    elif 'altdiffusion-m9' in model_name:
        print('Loading model Altdiffusion')
        pipeline = AltDiffusionPipeline.from_pretrained("BAAI/AltDiffusion-m9", cache_dir=cache_dir)
        pipeline = pipeline.to("cuda")
    elif 'stable-diffusion-3' in model_name:
        print('Loading model Stable Diffusion-3')
        pipeline = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16, cache_dir=cache_dir)
        pipeline = pipeline.to("cuda")
    elif 'blackforest' in model_name:
        print('Loading model Blackforest')
        pipeline = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            revision="refs/pr/1",
            torch_dtype=torch.bfloat16, 
            cache_dir=cache_dir)
        pipeline = pipeline.to("cuda")
    elif 'sd2.1' in model_name:
        print('Loading model Stable Diffusion v2.1')
        pipeline = StableDiffusionPipeline.from_pretrained(
            'stabilityai/stable-diffusion-2-1',
            torch_dtype=torch.float16, 
            use_safetensors=True, 
            variant="fp16",
            cache_dir=cache_dir
        )
        pipeline = pipeline.to("cuda")
    elif 'sdxl' in model_name:
        print('Loading model Stable Diffusion XL')
        pipeline = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", 
            torch_dtype=torch.float16, 
            use_safetensors=True, 
            variant="fp16",
            cache_dir=cache_dir)
        pipeline = pipeline.to("cuda")
    elif 'deepfloyd' in model_name:
        print('Loading model Deepfloyd')
        pipeline = DiffusionPipeline.from_pretrained(
            "DeepFloyd/IF-II-L-v1.0", text_encoder=None, variant="fp16", torch_dtype=torch.float16, cache_dir=cache_dir)
        pipeline.enable_xformers_memory_efficient_attention()  # remove line if torch.__version__ >= 2.0.0
        pipeline.enable_model_cpu_offload()
        pipeline = pipeline.to("cuda")


    pipeline.set_progress_bar_config(disable=True)




    # check if output path exists, otherwise create it
    if not os.path.exists(test_data_output_path.rsplit("/", 1)[0]):
        logging.info(f"Creating new path {test_data_output_path.rsplit('/', 1)[0]}")
        os.makedirs(test_data_output_path.rsplit("/", 1)[0])


    for index, row in tqdm(test_df.iterrows()):
        if ('Person' in row['Topic']) and (len(glob.glob(f"{test_data_output_path}{row['Language']}_{row['Culture']}_{row['Topic']}_{row['prompt_template']}.png")) != 1):
            image = pipeline(row['prompt']).images[0]

            imagetitle = row['Language'] + '_' + row['Culture'] + '_' + row['Topic'] + '_' + row['prompt_template'] + '.png'

            image.save(test_data_output_path + imagetitle)






if __name__ == "__main__":
    st = time.time()
    fire.Fire(main)
    logging.info(f'Total execution time: {time.time() - st:.2f} seconds')