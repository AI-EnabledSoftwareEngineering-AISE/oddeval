#!/usr/bin/env python
# coding: utf-8

# # Run from notebook

# In[1]:


import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from lavis.models import load_model_and_preprocess


# In[2]:


id2label = {
    0: 'bicycle',
    1: 'bus',
    2: 'car',
    3: 'motorcycle',
    4: 'pedestrian',
    5: 'rider',
    6: 'truck'
}
# Define paths
response_dir = '/home/user/projects/datasets/prompt_datasets/bdd100k/blip_response'
root_path = '/home/user/projects/datasets/prompt_datasets/bdd100k/images/'
json_file = '/home/user/projects/datasets/prompt_datasets/bdd100k/image_labels.json'

# Read the json file
with open(json_file, 'r') as f:
    data = json.load(f)

# For each image in data, join the path in root_path and the raw_file and save it in the new key fpath
for i, item in enumerate(data):
    file_path = item['raw_file']
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    data[i]['fpath'] = os.path.join(root_path, item['raw_file'])
    data[i]['prediction_path'] = os.path.join(response_dir, f'{file_name}.json')

# Setup device to use
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"



# In[4]:


# Load model
model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=True, device=device
)


# In[29]:


# Define roadway context
roadway_context = {
    'event': 'roadway users',
    'categories': [],
    'prompts': [
        "Do you see any bicycle in the image?",
        "Do you see any bus in the image?",
        "Do you see any car in the image?",
        "Do you see any motorcycle in the image?",
        "Do you see any pedestrian in the image?",
        "Do you see any rider in the image?",
        "Do you see any truck in the image?"
    ]
}

events = [roadway_context]


# In[30]:


data[0]


# In[2]:


# Function to process each image and save the results to the prediction_path
def get_answer(image, events, prediction_path, name='events', v='v1'):
    image_dict = {}
    image_dict['events'] = []
    for event in events:
        qa_dict = dict()
        qa_dict['event'] = event['event']
        qa_dict['prompts'] = event['prompts']
        qa_dict['answers'] = []

        for prompt in event['prompts']:
            ans = model.generate({"image": image, "prompt": f"{prompt}, answer in yes/no."})
            qa_dict['answers'].append(ans[0])

        image_dict['events'].append(qa_dict)
    
    # Save the result to the provided prediction_path
    with open(prediction_path, 'w') as f:
        json.dump(image_dict, f, indent=4)
    print(f'File {prediction_path} has been created')


# In[ ]:


# Process each image in the data object using its fpath and prediction_path
for i, item in enumerate(tqdm(data, desc="Processing Images")):
    image_path = item['fpath']
    prediction_path = item['prediction_path']

    # Load and process the image
    raw_image = Image.open(image_path)
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

    # Get answers and save to prediction_path
    get_answer(image, events, prediction_path, name='lane_markings', v='v1')
    # break

