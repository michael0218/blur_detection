from openai import OpenAI
import re
import base64
import requests

OPENAI_API_KEY= 'your-api-key'



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
    


client = OpenAI()

# Path to your image
image_path = "../sample_images/blur_dataset_scaled/sharp/0_IPHONE-SE_S.JPG"
# image_path = "../sample_images/blur_dataset_scaled/defocused_blurred/0_IPHONE-SE_F.JPG"
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
response_str = response.json()['choices'][0]['message']['content']

score_extractor = FindLLMScore()
score = score_extractor.re_0_10_rating(response_str)/10
print(score)