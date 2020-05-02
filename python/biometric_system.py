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

initial_database = {}
labels = np.array([j for i,j in dataset.imgs])
for _class in dataset.classes[:2000]:
    initial_database[dataset.class_to_idx[_class]] = np.where(labels==dataset.class_to_idx[_class])[0][0]

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
all_classes = np.array(list(dataset.idx_to_class.keys()))
imp_classes = np.array(list(initial_database.keys()))
fraud_classes = np.array(all_classes[len(imp_classes):])

class BiometricSystem():
    def __init__(self, database, vgg_dataset, model=None, mtcnn=None, threshold=0.5):
        ''' Database format:
            Dictionary from class_idx to the sample index in vgg_dataset
        '''
        self.database = database
        self.classes = database.keys()
        self.vgg_dataset = vgg_dataset
#         if vgg_dataset==None:
#             self.vgg_dataset = datasets.ImageFolder('../data/VGGFace2/train_cropped')
#         else:
#             self.vgg_dataset = vgg_dataset
#         if mtcnn:
#             self.mtcnn = mtcnn
#         else:
#             self.mtcnn = MTCNN(
#                     image_size=160, margin=0, min_face_size=20,
#                     thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
#                     device=device
        if model:
            self.model = model
        else:
            self.model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        self.trans = transforms.Compose([
            np.float32,
            transforms.ToTensor(),
            fixed_image_standardization
        ])
        self.threshold = threshold
        
    def checkfaces(self, query_refs, thresh=0.8):
        ''' List of queries for one day
        Get a query with the vgg_sample_idx query_ids'''
        query_embeddings, support_embeddings = self.get_embeddings(query_refs)
        neigh = NearestNeighbors(1, 1)
        neigh.fit(support_embeddings)
        dists, neighs = neigh.kneighbors(query_embeddings, 1)
        neighs[dists>thresh] = -1
        return neighs.flatten()
                
    def get_embeddings(self, query_refs):
        ''' List of queries for one day
        Get a query with the vgg_sample_idx query_ids'''
        aligned = []
        classes = []
        
        n = len(query_refs)
        for query_ref in query_refs:
            img = self.trans(self.vgg_dataset.__getitem__(query_ref)[0])
            aligned.append(img)
            
        for class_id, img_ref in self.database.items():
            img = self.trans(self.vgg_dataset.__getitem__(img_ref)[0])
            aligned.append(img)
            classes.append(class_id)

        aligned = torch.stack(aligned).to(device)
        embeddings = np.zeros((len(aligned), 512))
        for i in range(0, math.ceil(len(aligned)/32)):
            start = 32*i
            end = min(32*(i+1), len(aligned))
            embeddings[start:end] = resnet(aligned[start:end]).detach().cpu()
            
        embeddings = embeddings / np.linalg.norm(embeddings, axis=-1)[:, np.newaxis]
        query_embeddings = embeddings[:n]
        support_embeddings = embeddings[n:]
        return query_embeddings, support_embeddings
