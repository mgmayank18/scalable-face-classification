from facenet_pytorch import MTCNN, fixed_image_standardization, training
from BalancedBatchSampler import BalancedBatchSampler, SupportDataset
from finetune_on_support import finetune_on_support
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

class SupportDatabase():
    '''
    SUPPORT DATASET and SUPPORT DATABASE ARE DIFFERENT
    Apologies for the very confusing names
    Initially this class will store all classes and their first example
    Eventually new items will be added and then:

    This class will store all the support image_idx's and their corresponding class_idxs (according to the system)
    self.unique_class_ids: contains the UNIQUE support class_ids
    self.prototypes:       contain the prototype embedding for each class_id (in the order of class_ids)
    self.img_idxs:         contains ALL image_idxs in the support dataset even the online added
    self.labels:           contains labels corresponding to the img_idxs
    self.embeddings:       contains labels corresponding to the img_idxs


    Proposed alternate representation NOT IMPLEMENTED
    self.database: Dictionary from label name to { 1) img_idxs, 2) embeddings and the 3) prototype_embedding}
    '''
    def __init__(self, database, model, vgg_dataset, batch_size):
        # I have called this unique_class_ids to avoid confusion with labels
        self.unique_class_ids = np.zeros(len(database),dtype=int)
        self.prototypes = np.zeros((len(database), 512))
        self.img_idxs = np.zeros(len(database),dtype=int)
        self.labels = np.zeros(len(database),dtype=int)
        self.embeddings = np.zeros((len(database), 512))
        self.model = model
        #self.trans = trans
        self.vgg_dataset = vgg_dataset
        self.batch_size = batch_size
        # self.database = {}

        # Get embeddings for initial database items
        aligned = []
        for class_id, img_ref in database.items():
            img = self.vgg_dataset[img_ref][0]
            aligned.append(img)

        aligned = torch.stack(aligned).cuda()
        embeddings = np.zeros((len(aligned), 512))
        self.model.eval()
        for i in range(0, math.ceil(len(aligned)/self.batch_size)):
            start = self.batch_size*i
            end = min(self.batch_size*(i+1), len(aligned))
            embeddings[start:end] = self.model(aligned[start:end]).detach().cpu()

        embeddings = embeddings / (np.linalg.norm(embeddings, axis=-1)[:, np.newaxis])

        for i, item in enumerate(database.items()):
            label, img_idx = item
            self.unique_class_ids[i] = label
            self.prototypes[i] = embeddings[i]
            self.img_idxs[i] = img_idx
            self.labels[i] = label
            self.embeddings[i] = embeddings[i]
            
        #     self.database[label] = {
        #         'img_idx': [img_idx]
        #         'embeddings': [embeddings[i]]
        #         'prototype': [embeddings[i]]
        #     }

    def update_db(self, img_idxs, labels, embeddings):
        self.img_idxs = np.append(self.img_idxs, img_idxs).astype(int)
        self.labels = np.append(self.labels, labels).astype(int)
        self.embeddings = np.vstack((self.embeddings, embeddings))

        for i, label in enumerate(self.unique_class_ids):
            # print("before", self.prototypes[i])
            self.prototypes[i] = np.mean(self.embeddings[self.labels == label], axis=0)
            self.prototypes[i] = self.prototypes[i] / np.linalg.norm(self.prototypes[i])
            # print("after", self.prototypes[i])
    
    def update_model(self, model):
        self.model = model
        # Update embeddings and prototypes

        n = len(self.img_idxs)
        embeddings = np.zeros((n, 512))
        for i in range(0, math.ceil(n/self.batch_size)):
            start = self.batch_size*i
            end = min(self.batch_size*(i+1), n)
            aligned = []
            for img_idx in self.img_idxs[start:end]:
                img = self.vgg_dataset[img_idx][0]
                aligned.append(img)
            aligned = torch.stack(aligned).cuda()
            self.model.eval()
            embeddings[start:end] = self.model(aligned).detach().cpu()

        embeddings = embeddings / (np.linalg.norm(embeddings, axis=-1)[:, np.newaxis])
        self.embeddings = embeddings
        for i, label in enumerate(self.unique_class_ids):
            self.prototypes[i] = np.mean(self.embeddings[self.labels == label], axis=0)
            self.prototypes[i] = self.prototypes[i] / np.linalg.norm(self.prototypes[i])

    def __len__(self):
        return len(self.class_ids)

class BiometricSystem():
    def __init__(self, database, vgg_dataset, model, orig_target_dict, mtcnn=None, threshold=0.5, finetune_flag=True, batch_size=100, finetune_every = 10):
        ''' Database format:
            Dictionary from class_idx to the sample index in vgg_dataset
        '''
        self.finetune_every = finetune_every
        self.days = 0
        self.orig_target_dict = orig_target_dict
        self.classes = database.keys()
        self.vgg_dataset = vgg_dataset
        self.finetune_flag = finetune_flag
        self.batch_size = batch_size

        self.model = model
        #self.trans = transforms.Compose([
        #    np.float32,
        #    transforms.ToTensor(),
        #    fixed_image_standardization
        #])
        self.threshold = threshold
        self.supportDatabase = SupportDatabase(database, model, vgg_dataset, batch_size)
        
    def checkfaces(self, query_refs, thresh=self.threshold):
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
        mask = pred>0

        self.supportDatabase.update_db(query_refs[mask], pred[mask], query_embeddings[mask])     
        self.days +=1
        
        if(self.days%self.finetune_every == 0) and self.finetune_flag:
             # Do the finetuning
             # SUPPORT DATASET and SUPPORT DATABASE ARE DIFFERENT
             print("Finetuning on support database")
             supportDataset = SupportDataset(self.supportDatabase.img_idxs, self.supportDatabase.labels, self.vgg_dataset)
             np.save('img_idx.npy',self.supportDatabase.img_idxs)
             np.save('supportlabels.npy',self.supportDatabase.labels)
             balancedBatchSampler = BalancedBatchSampler(self.supportDatabase.labels, int(self.batch_size/10), 10)
             supportTrainLoader = DataLoader(supportDataset, batch_sampler=balancedBatchSampler)
             finetune_on_support(self.model, supportTrainLoader, self.orig_target_dict)
             self.supportDatabase.update_model(self.model)
        return pred
                
    def get_embeddings(self, query_refs):
        ''' List of queries for one day
        Get a query with the vgg_sample_idx query_ids'''
        classes = []
        
        n = len(query_refs)

        embeddings = np.zeros((n, 512))
        for i in range(0, math.ceil(len(query_refs)/self.batch_size)):
            start = self.batch_size*i
            end = min(self.batch_size*(i+1), n)
            aligned = []
            for query_ref in query_refs[start:end]:
                img = self.vgg_dataset[query_ref][0]
                aligned.append(img)
            aligned = torch.stack(aligned).cuda()
            self.model.eval()
            embeddings[start:end] = self.model(aligned).detach().cpu()

        embeddings = embeddings / (np.linalg.norm(embeddings, axis=-1)[:, np.newaxis])
        query_embeddings = embeddings
        support_embeddings = self.supportDatabase.prototypes
        return query_embeddings, support_embeddings
