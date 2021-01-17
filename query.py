from model.ResNet50 import ResNet50Features
from PIL import Image
from utils.similarity_measurement import SimilarityMeasurement
import torchvision.transforms as transforms
import numpy as np
import json
from utils import get_name as gn


DATABASE_FEATURES = './features/features.json'
QUERY_IMAGE = './query/t4.png'
peoples = {"ym": '杨幂', 'fbb': '范冰冰', 'lxl': '林心如', 'lzl': "林志玲"}
FEATURES_LEN = 1000
IMG_SIZE = 224

data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
image = Image.open(QUERY_IMAGE).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
image = data_transform(image).unsqueeze(dim=0)

model = ResNet50Features()

with open(DATABASE_FEATURES, 'r') as f:
    database = json.load(f)

output_feature = list(model(image)[0].data.numpy().astype(float))

ids = []
sims = []
for g_k in database.keys():
    g_value = database[g_k]
    sm = SimilarityMeasurement(output_feature, g_value)
    sims.append(sm.two_D())
    ids.append(g_k)
ids = np.array(ids)
sims = np.array(sims)
index = np.argsort(sims)
ids = ids[index]
print([peoples[gn.correction_name(id[:-4])] for id in ids])
sims = sims[index]
print(sims)

print("此人为: ", peoples[gn.correction_name(ids[0][:-4])])
