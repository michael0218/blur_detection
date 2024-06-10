#%%
from openai import OpenAI
import re, os
import base64
import requests
import pandas as pd
import time 
from tqdm import tqdm

OPENAI_API_KEY= 'your-api-key'

dataset_root = '../dataset'

# use os.work to recursively list all files in the directory
def list_all_files(root):
    all_files = []
    dir2label = {"defocused_blurred": 0, "motion_blurred": 0, "sharp": 1}
    for root, dirs, files in os.walk(root):
        for file in files:
            if file.endswith('.JPG') or file.endswith('.jpg'):
                label = dir2label[root.split('/')[-1]]
                all_files.append({'image_path':os.path.join(root, file),
                                  'label':label})
    return all_files

dataset_root_dict = list_all_files(dataset_root)
# %%

pd_data = pd.DataFrame(dataset_root_dict)
# %%
pd_data.describe()
pd_data.label.value_counts()
# %%
# add new columns: laplacian_variance and fft_high_freq to store the features
pd_data['laplacian_variance'] = 0.0
pd_data['fft_high_freq'] = 0.0
pd_data['score'] = 0.0

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


class FindLLMScore:
    def __init__(self):
        self.PATTERN_0_10: re.Pattern = re.compile(r"\b([0-9]|10)(?=\D*$|\s*\.)")
    
    def check_rating(self, v):
        if not (0 <= v <= 10):
            raise ValueError('Rating must be between 0 and 10')
        return v
    
    def re_0_10_rating(self, str_val: str) -> int:
        """Extract 0-10 rating from a string.
        If the string does not match, returns -10 instead."""

        matches = self.PATTERN_0_10.fullmatch(str_val)
        if not matches:
            # Try soft match
            matches = re.search(r'([0-9]+)(?=\D*$)', str_val)
            if not matches:
                print(f"Warning: 0-10 rating regex failed to match on: '{str_val}'")
                return -10  # so this will be reported as -1 after division by 10

        rating = self.check_rating(int(matches.group()))
        return rating
    
#%%

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X = pd_data[['image_path']]
y = pd_data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%%
client = OpenAI()



# estimate excution time
start = time.time()

# Path to your image
y_pred = []


for i in tqdm(X_test.index):
  image_path = X_test.loc[i]['image_path']
  # Getting the base64 string
  base64_image = encode_image(image_path)
  headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {OPENAI_API_KEY}"
  }

  payload = {
    "model": "gpt-4o",
    "messages": [
      { 
        "role":"system",
        "content":[
          {
            "type":"text",
            "text":"This is an image evaluation model. Please answer using only numbers between 0 and 10, without providing detailed explanations."
          }
        ]},
        {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Output a number between 0 and 10 indicating the blurriness or sharpness of this image. 0 means blurry, and 10 means sharp."
          },
          {
            "type": "image_url",
            "image_url": {
              "url": f"data:image/jpeg;base64,{base64_image}"
            }
          }
        ]
      }
    ],
    "max_tokens": 300
  }

  response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

  print(response.json())
  try :
    response_str = response.json()['choices'][0]['message']['content']

    score_extractor = FindLLMScore()
    score = score_extractor.re_0_10_rating(response_str)/10
    print(score)
    pd_data.loc[i, 'score'] = score
    y_pred.append(score)
  except:
    print(f"Error: {response.json()}")
    y_pred.append(-1)
  
endtime = time.time()
pd_data.to_csv('pd_data.csv', index=False)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# save the pd_data to a csv file



print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")
# print the excution time
print(f"Execution time: {endtime - start}s")

# save the Mean Squared Error and R^2 Score and excution time to a txt file as log
with open('log.txt', 'w') as f:
    f.write(f"Mean Squared Error: {mse}\n")
    f.write(f"R^2 Score: {r2}\n")
    f.write(f"Execution time: {endtime - start}s\n")
# %%



#  82%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                        | 318/387 [48:23<10:13,  8.90s/it]{'error': {'message': 'Your input image may contain content that is not allowed by our safety system.', 'type': 'invalid_request_error', 'param': None, 'code': 'content_policy_violation'}}
#  82%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                        | 318/387 [48:30<10:31,  9.15s/it]
# Traceback (most recent call last):
#   File "/Users/michael/git/blur_detection/openai_api/02openai_vision_img_evaluation.py", line 131, in <module>
#     response_str = response.json()['choices'][0]['message']['content']
# KeyError: 'choices'