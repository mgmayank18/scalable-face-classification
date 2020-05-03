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
from facenet_pytorch import fixed_image_standardization, InceptionResnetV1
from generate_original_embeddings import generate_original_embeddings
import math
import numpy as np
import os

import visdom
vis = visdom.Visdom(server='gpu10.int.autonlab.org',port='8097')

#ASSUMPTION: All support classes are first k classes of all_classes

num_query = 2000
fraud_ratio = 0.1
num_day = 10

num_query_total = num_query*num_day
tp = fp = tn = fn = 0
tp_list = np.zeros(num_day)
fp_list = np.zeros(num_day)
tn_list = np.zeros(num_day)
fn_list = np.zeros(num_day)

batch_size = 200
pretrained_path = './saved_models_attempt2/lr_0.0001/epoch_7.pt'
dataset_path = '../data/VGGFace2/train_cropped_split'
num_imp_classes = 450

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

#resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

print("Loading Dataset")
trans = transforms.Compose([
    np.float32,
    transforms.ToTensor(),
    fixed_image_standardization
])

dataset = datasets.ImageFolder(dataset_path, trans)
dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
labels = np.array([j for i,j in dataset.imgs])
dataset.class_to_instances = {class_idx : np.where(labels == class_idx)[0] for class_idx in dataset.idx_to_class.keys()}

print("Creating Dataloader")
loader = DataLoader(dataset, num_workers=workers, batch_size=batch_size)

pretrained_embeddings_path = 'pretrained_embeddings.npy'
if os.path.isfile(pretrained_embeddings_path):
    print("Loading Saved Pretrained Embeddings")
    pretrained_embeddings_list = np.load(pretrained_embeddings_path)
else:
    print("Generating Embeddings from pretrained weights")
    dataset = VGGFace2Dataset(dataset_path, trans)
    loader = DataLoader(dataset, num_workers=workers, batch_size=batch_size)
    pretrained_embeddings_list = generate_original_embeddings(loader, dataset_path, pretrained_path, pretrained_embeddings_path, batch_size, workers, device)

initial_database = {}
labels = np.array([j for i,j in dataset.imgs])
for _class in dataset.classes[:num_imp_classes]:
    initial_database[dataset.class_to_idx[_class]] = np.where(labels==dataset.class_to_idx[_class])[0][0]

all_classes = np.array(list(dataset.idx_to_class.keys()))
imp_classes = np.array(list(initial_database.keys()))
fraud_classes = np.array(all_classes[len(imp_classes):])

time_taken = []

print("Initializing Biometric System...")
biometricSystem = BiometricSystem(database=initial_database, model=resnet, vgg_dataset=dataset, orig_target_dict=pretrained_embeddings_path, batch_size=batch_size)

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
    tp_list[i] = 100*np.count_nonzero(_tp)/num_query
    fp_list[i] = 100*np.count_nonzero(_fp)/num_query
    tn_list[i] = 100*np.count_nonzero(_tn)/num_query
    fn_list[i] = 100*np.count_nonzero(_fn)/num_query
    
print(f"Average Time taken per day = {np.array(time_taken).mean()}")
print(f"tp = {100*tp/num_query_total}%")
print(f"fp = {100*fp/num_query_total}%")
print(f"tn = {100*tn/num_query_total}%")
print(f"fn = {100*fn/num_query_total}%")

print(f"tp_list = {tp_list}")
print(f"fp_list = {fp_list}")
print(f"tn_list = {tn_list}")
print(f"fn_list = {fn_list}")

# Plot them in same one later?

vis.line(Y= tp_list, X=np.arange(num_day), opts={
    "title" : "True Positive", 
    "ytickmin" : 0,
    "ytickmax" : 100,
    "linecolor":np.array([(0,0,255)])
    })

vis.line(Y= fp_list, X=np.arange(num_day), opts={
    "title" : "False Positive", 
    "ytickmin" : 0,
    "ytickmax" : 100,
    "linecolor":np.array([(255,0,0)])
    })

vis.line(Y= tn_list, X=np.arange(num_day), opts={
    "title" : "True Negative", 
    "ytickmin" : 0,
    "ytickmax" : 100,
    "linecolor":np.array([(0,0,255)])
    })

vis.line(Y= fn_list, X=np.arange(num_day), opts={
    "title" : "False Negative", 
    "ytickmin" : 0,
    "ytickmax" : 100,
    "linecolor":np.array([(255,0,0)])
    })
