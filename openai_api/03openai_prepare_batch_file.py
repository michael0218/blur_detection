# batch file example

# {"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": { "model": "gpt-4o",
#   "messages": [
#     {
#       "role": "user",
#       "content": [
#         {
#           "type": "text",
#           "text": "Output a number between 0 and 10 indicating the blurriness or sharpness of this image. 0 means blurry, and 10 means sharp."
#         },
#         {
#           "type": "image_url",
#           "image_url": {
#             "url": f"data:image/jpeg;base64,{base64_image}"
#           }
#         }
#       ]
#     }
#   ],
#   "max_tokens": 300}}




from openai import OpenAI
import re, os
import json
import base64
import requests

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def list_all_files(root):
    all_files = []
    dir2label = {"defocused_blurred": 0, "motion_blurred": 0, "sharp": 1}
    for root, dirs, files in os.walk(root):
        for file in files:
            if file.endswith('.JPG') or file.endswith('.jpg'):
                label = dir2label[root.split('/')[-1]]
                all_files.append(os.path.join(root, file))
    return all_files


dataset_root = '../sample_images/blur_dataset_scaled'
list_all_image_path = list_all_files(dataset_root)

# dump json into jsonl file

jsonl_file = open('batch_files.jsonl', 'w')

for i in range(10):
    image_path = list_all_image_path[i]
    base64_image = encode_image(image_path)

    d = {"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": { "model": "gpt-4o",
        "messages": [
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
        "max_tokens": 300}}
    jsonl_file.write(json.dumps(d)+'\n')
jsonl_file.close()