from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from VGGFace2Dataset import VGGFace2Dataset
import numpy as np
import torch.nn as nn
import os 
from torchvision import transforms
import sys

def generate_original_embeddings(loader, dataset_path, pretrained_path, pretrained_embeddings_path, batch_size, workers, device):
    data_dir = dataset_path
    weights = pretrained_path

    resnet = InceptionResnetV1(
        classify=False,
        pretrained=None,
        num_classes=5631
    )

    resnet = nn.DataParallel(resnet.to(device))
    trained_weights = torch.load(weights)
    resnet.load_state_dict(trained_weights)

    vgg_data_loader = loader
    
    resnet.eval().to(device)

    emb_dict = []

    for i_batch, (x, y, index) in enumerate(vgg_data_loader):
        sys.stdout.write('\r - Processing Batch %d or %d.' % (i_batch,len(vgg_data_loader)))
        sys.stdout.flush()
        x = x.to(device)
        y = y.to(device)
        y_pred = resnet(x)
        emb_dict.append(y_pred.cpu().detach().numpy())
        
    emb_dict = np.vstack(emb_dict)
    print(emb_dict.shape)
    np.save('pretrained_embeddings.npy',emb_dict)
    return emb_dict
        
