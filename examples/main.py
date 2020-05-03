from biometric_system import BiometricSystem
from inception_resnet_v1 import InceptionResnetV1_LWF
from utils import load_orig_task_weights
import time
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from VGGFace2Dataset import VGGFace2Dataset
from facenet_pytorch import fixed_image_standardization
from generate_original_embeddings import generate_original_embeddings
import math
import numpy as np
import os


#ASSUMPTION: All support classes are first k classes of all_classes

num_query = 2000
fraud_ratio = 0.1
num_day = 2 

num_query_total = num_query*num_day
tp = fp = tn = fn = 0

batch_size = 500
pretrained_path = './saved_models_first_pretrain/epoch_29.pt'
dataset_path = '../data/VGGFace2/train_cropped_split'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))
    
workers = 0 if os.name == 'nt' else 24

resnet = InceptionResnetV1_LWF(
    classify=False,
    pretrained=None,
    num_classes=5631
)
resnet = nn.DataParallel(resnet)
resnet = load_orig_task_weights(resnet,pretrained_path)
resnet.eval().to(device)

def collate_fn(x):
    return x[0]

print("Loading Dataset")
trans = transforms.Compose([
    np.float32,
    transforms.ToTensor(),
    fixed_image_standardization
])
dataset = VGGFace2Dataset(dataset_path, trans)
dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
labels = np.array([j for i,j in dataset.imgs])
dataset.class_to_instances = {class_idx : np.where(labels == class_idx)[0] for class_idx in dataset.idx_to_class.keys()}

print("Creating Dataloader")
loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)

pretrained_embeddings_path = 'pretrained_embeddings.npy'
if os.path.isfile(pretrained_embeddings_path):
    print("Loading Saved Pretrained Embeddings")
    pretrained_embeddings_list = np.load(pretrained_embeddings_path)
else:
    print("Generating Embeddings from pretrained weights")
    pretrained_embeddings_list = generate_original_embeddings(loader, dataset_path, pretrained_path, pretrained_embeddings_path, batch_size, workers, device)

initial_database = {}
labels = np.array([j for i,j in dataset.imgs])
for _class in dataset.classes[:2000]:
    initial_database[dataset.class_to_idx[_class]] = np.where(labels==dataset.class_to_idx[_class])[0][0]

all_classes = np.array(list(dataset.idx_to_class.keys()))
imp_classes = np.array(list(initial_database.keys()))
fraud_classes = np.array(all_classes[len(imp_classes):])

time_taken = []

print("Initializing Biometric System...")
biometricSystem = BiometricSystem(database=initial_database, model=resnet, vgg_dataset=dataset)

for i in range(num_day):
    print("Day ",i)
    
    fraud = np.random.rand(num_query) < fraud_ratio
    labels = np.random.choice(imp_classes, num_query)
    labels[fraud] = np.random.choice(fraud_classes, len(labels[fraud]))

    query_ids = [np.random.choice(dataset.class_to_instances[label]) for label in labels]
    query_ids = np.array(query_ids)
    
    t = time.process_time()
    print("Checking Today's Faces ")

    pred = biometricSystem.checkfaces(query_ids)
    elapsed_time = time.process_time() - t
    time_taken.append(elapsed_time)
    
    _tp = np.logical_and(pred == labels, pred >= 0)
    _fp = np.logical_and(pred != labels, pred >= 0)
    _tn = np.logical_and(labels > len(imp_classes)-1, pred < 0)
    _fn = np.logical_and(labels <= len(imp_classes)-1, pred < 0)
    tp += np.count_nonzero(_tp)
    fp += np.count_nonzero(_fp)
    tn += np.count_nonzero(_tn)
    fn += np.count_nonzero(_fn)
    
print(f"Average Time taken per day = {np.array(time_taken).mean()}")
print(f"tp = {100*tp/num_query_total}%")
print(f"fp = {100*fp/num_query_total}%")
print(f"tn = {100*tn/num_query_total}%")
print(f"fn = {100*fn/num_query_total}%")

