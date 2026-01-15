import pandas as pd
import cv2
import scipy.cluster
import numpy as np
import math
from tqdm import tqdm
import os
import glob
import json
import requests
import matplotlib.pyplot as plt
import time
import random


repo = '<REPO_PATH>'

for language in tqdm(os.listdir(f'{repo}/images/')):
    for model in  os.listdir(f'{repo}/images/{language}'):
        print(model)
        full_list_dataset = []

        if os.path.exists(f'{repo}/image_prom_color/{language}_{model}.csv'):
            old_df = pd.read_csv(f'{repo}/image_prom_color/{language}_{model}.csv')
        else: 
            old_df = pd.DataFrame() 

        for num_clusters in ['5', '6', '7', '8']: 
            print('Running for cluster ' + num_clusters)
            for prompt in tqdm(glob.glob(f'{repo}/images/{language}/{model}/*_Person_*.png')):
                row = {}
                img_name = prompt.split('/')[-1]

                # check if output path exists, otherwise create it
                if len(old_df) > 0:
                    sub = old_df[
                        (old_df['Language'] == language) & 
                        (old_df['Model'] == model)  &
                        (old_df['Total Clusters'] == int(num_clusters))
                        ]

                    if img_name in sub.Concept.unique():
                        continue

                #print(prompt)
                row['Language'] = language
                row['Model'] = model
                row['Concept'] = img_name
                row['Total Clusters'] = num_clusters

                
                IMAGE_URL = f'{'<GITHUB_STORAGE>'}/refs/heads/main/{language}/{model}/{img_name}'
                PRECISION = 'vlow'
                CLUSTERS = num_clusters

                

                response = requests.get(f'http://mkweb.bcgsc.ca/color-summarizer/?url={IMAGE_URL}&precision={PRECISION}&num_clusters={CLUSTERS}&json=1')


                if response.status_code != 200:
                    print('Server error')
                    time.sleep(5) # Sleep for 10 seconds
                    response = requests.get(f'http://mkweb.bcgsc.ca/color-summarizer/?url={IMAGE_URL}&precision={PRECISION}&num_clusters={CLUSTERS}&json=1')

                if response.status_code != 200:
                    print('Server error')
                    time.sleep(5) # Sleep for 10 seconds
                    response = requests.get(f'http://mkweb.bcgsc.ca/color-summarizer/?url={IMAGE_URL}&precision={PRECISION}&num_clusters={CLUSTERS}&json=1')

                test = response.text
                #print(test)

                try:
                    json_object = json.loads(test)
                except:
                    df = pd.DataFrame.from_dict(full_list_dataset) 
                    old_df = pd.concat([old_df, df])
                    old_df.to_csv(f'{repo}/image_prom_color/{language}_{model}.csv', index = False)
                    continue


                for cluster in range(0, int(CLUSTERS)):
                    cluster = str(cluster)
                    l = float(json_object['clusters'][cluster]['f'])
                    new_row = row.copy()
                    new_row['Cluster'] = cluster
                    new_row['Percentage'] = l
                    new_row['Hex'] = json_object['clusters'][cluster]['hex']
                    new_row['RGB'] = json_object['clusters'][cluster]['rgb']

                    full_list_dataset.append(new_row)

                time.sleep(random.randint(2, 7)) # Sleep for 2 seconds


            df = pd.DataFrame.from_dict(full_list_dataset) 
            old_df = pd.concat([old_df, df])
            old_df.to_csv(f'{repo}/image_prom_color/{language}_{model}.csv', index = False)

