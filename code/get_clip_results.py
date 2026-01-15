import open_clip
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel, AutoTokenizer
import requests
import pandas as pd
import os
import glob
from tqdm import tqdm
import numpy as np



repo = '<REPO_PATH>'

cache_dir = repo + '/cache_dir'

clipmodel, preprocess = open_clip.create_model_from_pretrained('hf-hub:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k')
tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k')


all_rows = []
for language in tqdm(os.listdir(f'{repo}/images/')):
    for model in os.listdir(f'{repo}/images/{language}'):
        for prompt in tqdm(glob.glob(f'{repo}/images/{language}/{model}/*.png')):

            device = 'cuda:1'

            language_text = language[5:]

            if language_text[0] in ['a','e', 'i', 'o', 'u']:
                article = 'an'
            else:
                article = 'a'

            text_lang = 'A photo of ' + article + ' ' + language_text + ' person'

            culture_text = prompt.split('/')[-1].split('_')[1]

            if culture_text[0] in ['a','e', 'i', 'o', 'u']:
                article = 'an'
            else:
                article = 'a'

            text_cult = 'A photo of ' + article + ' ' + culture_text + ' person'

            

            cult_and_language = [text_lang, text_cult]

            text = tokenizer(cult_and_language).to(device)

            image = Image.open(prompt)

            inputs = preprocess(image).unsqueeze(0).to(device)

            clipmodel = clipmodel.to(device)

            with torch.no_grad(), torch.cuda.amp.autocast():
                image_features = clipmodel.encode_image(inputs)
                text_features = clipmodel.encode_text(text)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)

                text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

            

            row = {}
            cols = ['Language', 'Culture']

            row = dict(zip(cols, text_probs.tolist()[0]))

            row['image'] = prompt

            all_rows.append(row)


df = pd.DataFrame.from_dict(all_rows) 

df.to_csv(f'{repo}/embeddings/clip_scores.csv', index=False)

