from model.ResNet50 import ResNet50Features
from PIL import Image
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import json
import torch


DataBase_DIR = './database/'
feature_save = './features/features.json'
FEATURES_LEN = 1000
IMG_SIZE = 224


class ImageDataset(data.Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.path[index]
        image = Image.open(DataBase_DIR+img_path).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
        if self.transform is not None:
            image = self.transform(image)
        return image, img_path

    def __len__(self):
        return len(self.path)


data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


model = ResNet50Features()
img_list = [f for f in os.listdir(DataBase_DIR) if f != ".DS_Store"]
features = {}

dataset = ImageDataset(img_list, transform=data_transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
for d in dataloader:
    input, image_name = d[0], d[1]
    output_feature = list(model(input)[0].data.numpy().astype(float))
    features[image_name[0]] = output_feature


with open(feature_save, 'w') as f:
    json.dump(features, f)

