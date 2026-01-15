from PIL import Image
import requests
from transformers import AutoProcessor, AutoModel
import torch
import glob
import os
import pandas as pd
from tqdm import tqdm

repo = '<REPO PATH>'
cache_dir = repo + '/cache_dir'

emb_model = "google/siglip-so400m-patch14-384"  #'laion/CLIP-ViT-bigG-14-laion2B-39B-b160k' #"google/siglip-so400m-patch14-384" 

clipmodel = AutoModel.from_pretrained(emb_model, cache_dir=cache_dir).to('cuda:1')
processor = AutoProcessor.from_pretrained(emb_model, cache_dir=cache_dir)

df_old = pd.read_csv(f'{repo}/embeddings/siglip_embeddings.csv')


all_rows = []
for language in tqdm(os.listdir(f'{repo}/images/')):
    for model in os.listdir(f'{repo}/images/{language}'):
        for prompt in tqdm(glob.glob(f'{repo}/images/{language}/{model}/*.png')):
            row = {}
            row['model'] = model
            row['language'] = language
            row['subject'] = prompt.split('/')[7]
            try:

                image = Image.open(prompt)

                inputs = processor(images=image, return_tensors="pt").to('cuda:1')

                image_features = clipmodel.get_image_features(**inputs)

                row[emb_model] = image_features.cpu().detach().numpy().tolist()[0]
                all_rows.append(row)
            except:
                print(prompt)




df = pd.DataFrame.from_dict(all_rows) 
df = pd.concat([df,df_old])
df.to_csv(f'{repo}/embeddings/siglip_embeddings.csvg',index=False)

