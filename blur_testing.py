
import torch, cv2
import numpy as np
from urllib.request import urlopen
from torchvision import transforms
import torch.nn as nn
# test single image and print class
def url_to_image(url, readFlag=cv2.IMREAD_COLOR):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp = urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, readFlag)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # return the image
    return image

manual_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.CenterCrop(224), # 1. Reshape all images to 224x224 (though some models may require different sizes)
    transforms.ToTensor(), # 2. Turn image values to between 0 & 1 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], # 3. A mean of [0.485, 0.456, 0.406] (across each colour channel)
                         std=[0.229, 0.224, 0.225]) # 4. A standard deviation of [0.229, 0.224, 0.225] (across each colour channel),
])

# img_url_path = "https://storage.mds.yandex.net/get-mturk/1350764/c55c766b-fea1-4cf0-9d7c-17bc90a050e9"
# img_url_path = "https://thumbs.dreamstime.com/z/abstract-blur-city-park-bokeh-background-abstract-blur-city-park-bokeh-background-blur-image-trees-park-evening-123059460.jpg?ct=jpeg"
img_url_path = "https://thumbs.dreamstime.com/z/green-park-tree-garden-under-sunset-background-exercise-relax-high-quality-photo-297933668.jpg?ct=jpeg"

class blurDetectionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3,32,kernel_size=3,stride=2),
            torch.nn.BatchNorm2d(32),
        )
        self.act1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(32,64,kernel_size=5),
            torch.nn.BatchNorm2d(64),
        )
        self.act2 = torch.nn.ReLU()
        
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(64,128,kernel_size=5,stride=1),
            torch.nn.BatchNorm2d(128),
        )
        self.act3 = torch.nn.ReLU()
        self.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=1)
        self.head = torch.nn.Sequential(
                        torch.nn.Flatten(),
                        torch.nn.Dropout(p=0.3, inplace=True), 
                        torch.nn.Linear(in_features=128, # checked by running without head
                                        out_features=2, # same number of output units as our number of classes
                                        bias=True))
    def forward(self,x):
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.act3(self.conv3(x))
        x = self.avgpool(x)
        x = self.head(x)
        return x
    

model_small = torch.load('blur_detection_small.pth', map_location=torch.device('cpu'))

img = url_to_image(img_url_path)

model_small.eval()

img_trans = manual_transforms(img)

img_trans_batch = img_trans.unsqueeze(0)
result = model_small(img_trans_batch)
_, class_id = torch.max(result, 1)

print('*'*50)
print("The image : ",img_url_path)
if int(class_id[0]) == 0:
    print('Blurry')
else:
    print('Sharp')
