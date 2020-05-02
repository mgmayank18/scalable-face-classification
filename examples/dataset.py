from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import math

import numpy as np
import os

from sklearn.neighbors import NearestNeighbors

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

def collate_fn(x):
    return x[0]
workers = 0 if os.name == 'nt' else 4

toy_dataset = datasets.ImageFolder('../data/test_images')
toy_dataset.idx_to_class = {i:c for c, i in toy_dataset.class_to_idx.items()}
toy_loader = DataLoader(toy_dataset, collate_fn=collate_fn, num_workers=workers)

dataset = datasets.ImageFolder('../data/VGGFace2/train_cropped_split')
dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
labels = np.array([j for i,j in dataset.imgs])
dataset.class_to_instances = {class_idx : np.where(labels == class_idx)[0] for class_idx in dataset.idx_to_class.keys()}
loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)