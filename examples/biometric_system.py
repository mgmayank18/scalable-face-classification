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

from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import visdom
vis = visdom.Visdom()

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
        self.vgg_dataset = vgg_dataset
        self.batch_size = batch_size
        self.gt_labels = np.zeros(len(database),dtype=int)

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
            
            self.gt_labels[i] = label
            
        #     self.database[label] = {
        #         'img_idx': [img_idx]
        #         'embeddings': [embeddings[i]]
        #         'prototype': [embeddings[i]]
        #     }

        self.tsne_X = None
        self.dark_index = np.amax(self.labels) + 1
        self.color = None




    def update_db(self, img_idxs, labels, embeddings, gt_labels):
        self.img_idxs = np.append(self.img_idxs, img_idxs).astype(int)
        self.labels = np.append(self.labels, labels).astype(int)
        self.embeddings = np.vstack((self.embeddings, embeddings))
        self.gt_labels = np.append(self.gt_labels, gt_labels).astype(int)


        self.update_prototypes()

        # for i, label in enumerate(self.unique_class_ids):
        #     self.prototypes[i] = np.mean(self.embeddings[self.labels == label], axis=0)
        #     self.prototypes[i] = self.prototypes[i] / np.linalg.norm(self.prototypes[i])
        
        # self.tsne_X = TSNE(n_components=2).fit_transform(self.prototypes)
        color_num = np.unique(self.gt_labels).shape[0]
        self.color = (np.random.random_sample((color_num, 3))*200).astype(int) + 50
        self.color[-1,:] = 0    # 501th colour is fraud / msg of darkness
        print("updating")
        self.tsne_X = TSNE(n_components=2).fit_transform(self.embeddings)
        self.plot_outliers()
        print("Shape of TSNE X ", self.tsne_X.shape)
        
    
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
        self.update_prototypes()
        # for i, label in enumerate(self.unique_class_ids):
        #     self.prototypes[i] = np.mean(self.embeddings[self.labels == label], axis=0)
        #     self.prototypes[i] = self.prototypes[i] / np.linalg.norm(self.prototypes[i])

    def update_prototypes(self):
        n = len(self.unique_class_ids)
        kmeans = KMeans(n, init=self.embeddings[:n])
        kmeans.fit(self.embeddings)
        self.prototypes = kmeans.cluster_centers_
        self.labels[n:] = kmeans.predict(self.embeddings[n:])

    
    def plot_outliers(self):
        tsne_outlier_thresh = 1.414 * 50    # magic-est number
        invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                std = [ 1., 1., 1. ]),
                                                ])

        radius = np.linalg.norm(self.tsne_X, axis=1)
        print(radius)
        out_ind = np.where(radius > tsne_outlier_thresh)
        outlier_idxs = self.img_idxs[out_ind]
        for idx in outlier_idxs:
            img = self.vgg_dataset[idx][0]
            img = invTrans(img)
            print("plot")
            vis.image(img)



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
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001)
        self.threshold = threshold
        self.supportDatabase = SupportDatabase(database, model, vgg_dataset, batch_size)
    
    def checkfaces(self, query_refs):
        ''' List of queries for one day
        Get a query with the vgg_sample_idx query_ids'''

        thresh = self.threshold
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

        gt_labels = self.get_gt_labels(query_refs)

        self.supportDatabase.update_db(query_refs[mask], pred[mask], query_embeddings[mask], gt_labels[mask])     
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
             L_old, L_new = finetune_on_support(self.model, supportTrainLoader, self.orig_target_dict, self.optimizer)
             self.supportDatabase.update_model(self.model)
             torch.save(self.model.state_dict(),"Model_MSE_"+str(L_old.item())+"_Trip_"+str(L_new.item())+".pt")
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

    def get_gt_labels(self, query_refs):
        ''' Get the ground truth labels
        '''
        n = len(query_refs)
        
        gt_labels = []
        for i in range(0, math.ceil(n/self.batch_size)):
            start = self.batch_size*i
            end = min(self.batch_size*(i+1), n)
            
            for query_ref in query_refs[start:end]:
                label = self.vgg_dataset[query_ref][1]
                if len(np.where(self.supportDatabase.labels == label)[0]) == 0:
                    label = self.supportDatabase.dark_index
                gt_labels.append(label)
            
        gt_labels = np.array(gt_labels)

        return gt_labels
