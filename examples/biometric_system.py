from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import math
import numpy as np
import time
import os

from sklearn.neighbors import NearestNeighbors

class BiometricSystem():
    def __init__(self, database, vgg_dataset, model=None, mtcnn=None, threshold=0.5, finetune_flag=True, batch_size=100):
        ''' Database format:
            Dictionary from class_idx to the sample index in vgg_dataset
        '''
        self.database = database
        self.classes = database.keys()
        self.vgg_dataset = vgg_dataset
        self.finetune_flag = finetune_flag
        self.batch_size = batch_size

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
        
    def checkfaces(self, query_refs, thresh=0.7):
        ''' List of queries for one day
        Get a query with the vgg_sample_idx query_ids'''
        a = time.time()
        query_embeddings, support_embeddings = self.get_embeddings(query_refs)
        b = time.time()
        print("Calculated Embeddings in ", b-a, " seconds")
        neigh = NearestNeighbors(1, 1)
        neigh.fit(support_embeddings)
        dists, neighs = neigh.kneighbors(query_embeddings, 1)
        neighs[dists>thresh] = -1
        c = time.time()
        print("Calculated NN in ", c-b, " seconds")
        pred = neighs.flatten()

        for query_ref, pred_val in zip(query_refs, pred):
            # TODO: update accordingly using Sanil's updated internal database structure
            supportDataset = SupportDataset(self.img_idxs, self.labels, self.vgg_dataset)
            balancedBatchSampler = BalancedBatchSampler(self.labels, 4, 4)
            supportTrainLoader = DataLoader(supportDataset, batch_sampler=balancedBatchSampler)
            finetune_on_support(self.model, supportTrainLoader, self.orig_target_dict)

        return pred
                
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
            start = self.batch_size*i
            end = min(self.batch_size*(i+1), len(aligned))
            embeddings[start:end] = resnet(aligned[start:end]).detach().cpu()
        
        embeddings = embeddings / np.linalg.norm(embeddings, axis=-1)[:, np.newaxis]
        query_embeddings = embeddings[:n]
        support_embeddings = embeddings[n:]
        return query_embeddings, support_embeddings
