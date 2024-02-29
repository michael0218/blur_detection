#%%
import numpy as np
import os
import pandas as pd 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import cv2
from PIL import Image
from tempfile import TemporaryDirectory

# cudnn.benchmark = True
# plt.ion()   # interactive mode
# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
device
# %%
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
# from rembg import remove
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image


root_dir = '/Users/michael/Downloads/archive/blur_dataset_scaled'

class BlurDataset(Dataset):
    def __init__(self, dataset, class_indices):
        self.dataset = dataset
        self.class_indices = class_indices

    def __getitem__(self, index):
        img, label = self.dataset[index]
        if label in self.class_indices.keys():
            return img, self.class_indices[label]

    def __len__(self):
        return len(self.dataset)
# %%
manual_transforms = transforms.Compose([
#     transforms.ToPILImage(),
    transforms.Resize((224, 224)), # 1. Reshape all images to 224x224 (though some models may require different sizes)
    transforms.ToTensor(), # 2. Turn image values to between 0 & 1 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], # 3. A mean of [0.485, 0.456, 0.406] (across each colour channel)
                         std=[0.229, 0.224, 0.225]) # 4. A standard deviation of [0.229, 0.224, 0.225] (across each colour channel),
])
# %%
class_indices = {0:0, 1:0,2:1}

# train_dataset = datasets.ImageFolder(root=root_dir, transform=manual_transforms)
# custom_dataset = BlurDataset(dataset, class_indices)
# %%
train_folder = datasets.ImageFolder(root=root_dir, transform=manual_transforms)

valid_folder = datasets.ImageFolder(root=root_dir, transform=manual_transforms)

train_dataset = BlurDataset(train_folder, class_indices)

valid_dataset = BlurDataset(valid_folder, class_indices)

num_train = len(train_dataset)
indices = list(range(num_train))
split = int(np.floor(0.3 * num_train))

np.random.seed(40)
np.random.shuffle(indices)
# %%
train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)
# %%
batch_size = 32 

train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler)
test_dataloader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=batch_size, sampler=valid_sampler)
# %%
im,label = next(iter(train_dataloader))
# %%
label
# %%
print(label[2].detach().numpy())

# %%
from random import sample


s = 5
idxs = sample(list(range(im.shape[0])),s)
fig,ax = plt.subplots(1,s,figsize=(30,30))
for i,idx in enumerate(idxs):
    ax[i].set_title(label[idx].detach().numpy())
    ax[i].imshow(im[idx,...].permute(1,2,0).detach().numpy())
# %%
model = torchvision.models.efficientnet_b0()
# %%
# Get a set of pretrained model weights
weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT # .DEFAULT = best available weights from pretraining on ImageNet
model = torchvision.models.efficientnet_b0(weights=weights).to(device)

# Get the transforms used to create our pretrained weights
auto_transforms = weights.transforms()
auto_transforms


# %%
# Freeze all base layers in the "features" section of the model (the feature extractor) by setting requires_grad=False
for param in model.features.parameters():
    param.requires_grad = False
# %%
# Set the manual seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Get the length of class_names (one output unit for each class)
output_shape = 2

# Recreate the classifier layer and seed it to the target device
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True), 
    torch.nn.Linear(in_features=1280, 
                    out_features=2, # same number of output units as our number of classes
                    bias=True)).to(device)
# %%
# Train

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# %%
loss_fn(torch.tensor([0.3,0.2,0.9]), torch.tensor([0.0,0.0,1.0]))
# %%
"""
Contains functions for training and testing a PyTorch model.
"""
import torch

from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). For example:

    (0.1112, 0.8743)
    """
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss

        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (test_loss, test_accuracy). For example:

    (0.0223, 0.8985)
    """
    # Put model in eval mode
    model.eval() 

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss

            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device) -> Dict[str, List]:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for 
    each epoch.
    In the form: {train_loss: [...],
              train_acc: [...],
              test_loss: [...],
              test_acc: [...]} 
    For example if training for epochs=2: 
             {train_loss: [2.0616, 1.0537],
              train_acc: [0.3945, 0.3945],
              test_loss: [1.2641, 1.5706],
              test_acc: [0.3400, 0.2973]} 
    """
    # Create empty results dictionary
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []
    }
    
    # Make sure model on target device
    model.to(device)

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          device=device)
        test_loss, test_acc = test_step(model=model,
          dataloader=test_dataloader,
          loss_fn=loss_fn,
          device=device)

        # Print out what's happening
        print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc:.4f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    # Return the filled results at the end of the epochs
    return results
# %%
# Set the random seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Start the timer
from timeit import default_timer as timer 
start_time = timer()

# Setup training and save the results
results = train(model=model,
                       train_dataloader=train_dataloader,
                       test_dataloader=test_dataloader,
                       optimizer=optimizer,
                       loss_fn=loss_fn,
                       epochs=23,
                       device=device)

# End the timer and print out how long it took
end_time = timer()
print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")
# %%
torch.save(model, 'blur_detection_big.pth')

#%%
import cv2
from urllib.request import urlopen
def url_to_image(url, readFlag=cv2.IMREAD_COLOR):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp = urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, readFlag)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # return the image
    return image
#%%
img = url_to_image('https://upload.wikimedia.org/wikipedia/commons/thumb/7/75/Mount_Ararat_and_the_Yerevan_skyline_in_spring_%2850mm%29.jpg\
/310px-Mount_Ararat_and_the_Yerevan_skyline_in_spring_%2850mm%29.jpg')
# img = torch.from_numpy(img)
#
conv2d = torch.nn.Conv2d(in_channels=3,
                         out_channels=10,
                         kernel_size=3,
                         
                         )

manual_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.CenterCrop(224), # 1. Reshape all images to 224x224 (though some models may require different sizes)
    transforms.ToTensor(), # 2. Turn image values to between 0 & 1 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], # 3. A mean of [0.485, 0.456, 0.406] (across each colour channel)
                         std=[0.229, 0.224, 0.225]) # 4. A standard deviation of [0.229, 0.224, 0.225] (across each colour channel),
])
#%%
img_t = manual_transforms(img)
fix,ax = plt.subplots(2,1,figsize=(20, 20))
ax[1].imshow(img_t.permute(1,2,0))
ax[0].imshow(img)
# %%
tensor = conv2d(img_t)
# %%
s = 5
fig,ax = plt.subplots(1,s,figsize=(30,30))
for i in range(s):
    ax[i].imshow(tensor[i,...].detach().numpy())
# %%

# Simpler model
# Get a set of pretrained model weights
weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT # .DEFAULT = best available weights from pretraining on ImageNet
model_small = torchvision.models.efficientnet_b0(weights=weights).to(device)

model_small.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True), 
    torch.nn.Linear(in_features=1280, 
                    out_features=2, # same number of output units as our number of classes
                    bias=True)).to(device)
# %%
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
    
# %%
model_small.features = model_small.features[:5]
model_small.classifier[1] = torch.nn.Linear(80,2)
# %%
model_small = blurDetectionCNN()



# Define loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_small.parameters(), lr=0.001)
# %%
# Freeze all base layers in the "features" section of the model (the feature extractor) by setting requires_grad=False
for param in model_small.features.parameters():
    param.requires_grad = False
# %%
torch.manual_seed(40)
torch.cuda.manual_seed(40)

# Start the timer
from timeit import default_timer as timer 
start_time = timer()

# Setup training and save the results
results = train(model=model_small,
                       train_dataloader=train_dataloader,
                       test_dataloader=test_dataloader,
                       optimizer=optimizer,
                       loss_fn=loss_fn,
                       epochs=17,
                       device=device)

# End the timer and print out how long it took
end_time = timer()
print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")
# %%
plt.plot(results['train_loss'],label='train')
plt.plot(results['test_loss'],label='test')
plt.legend()
# %%
torch.save(model_small, 'blur_detection_small.pth')