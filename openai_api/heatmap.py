import os

import torch
import urllib
from PIL import Image
from torchvision import transforms

import numpy as np
import cv2

# load the model
model = torch.hub.load('pytorch/vision:v0.9.0', 'densenet121', pretrained=True)
# or any of these variants
# model = torch.hub.load('pytorch/vision:v0.9.0', 'densenet169', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.9.0', 'densenet201', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.9.0', 'densenet161', pretrained=True)
model.eval()

filename = 'dog.jpg'

# get input image and apply preprocessing
input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)

# output = model(input_batch)
# print(output.shape)

# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
# probabilities = torch.nn.functional.softmax(output[0], dim=0)

# print(probabilities)

named_layers = dict(model.named_modules())


# for layer in named_layers:
#     print(layer)
# print(named_layers['features.denseblock4.denselayer16'])


class densenet_last_layer(torch.nn.Module):
    def __init__(self, model):
        super(densenet_last_layer, self).__init__()
        self.features = torch.nn.Sequential(
            *list(model.children())[:-1]
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.nn.functional.relu(x, inplace=True)
        return x


conv_model = densenet_last_layer(model)
# print(conv_model)
input_batch = torch.autograd.Variable(input_batch)
conv_output = conv_model(input_batch)

# print(conv_output.shape)

conv_output = conv_output.cpu().data.numpy()
# conv_output = np.squeeze(conv_output)

# for state in model.state_dict():
#     print(state)

weights = model.state_dict()['classifier.weight']
weights = weights.cpu().numpy()

bias = model.state_dict()['classifier.bias']
bias = bias.cpu().numpy()

# print(conv_output.shape)
heatmap = None

for i in range(0, len(weights)):
    map = conv_output[0, i, :, :]
    if i == 0:
        heatmap = weights[i] * map
    else:
        heatmap += weights[i] * map

# ---- Blend original and heatmap
npHeatmap = heatmap.cpu().data.numpy()

transCrop = 224

imgOriginal = cv2.imread(filename, 1)
imgOriginal = cv2.resize(imgOriginal, (transCrop, transCrop))

cam = npHeatmap / np.max(npHeatmap)
cam = cv2.resize(cam, (transCrop, transCrop))
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

img = heatmap * 0.5 + imgOriginal

cv2.imwrite(os.getcwd(), img)

print(model.state_dict()['classifier.weight'])