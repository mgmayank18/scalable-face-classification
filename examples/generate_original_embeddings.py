from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from VGGFace2Dataset import VGGFace2Dataset
import numpy as np
import torch.nn as nn
import os 
from torchvision import transforms

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

<<<<<<< 88b820bc002f9a6f259d855fa30cbb78c1dd752e
    for i_batch, (x, y, index) in enumerate(vgg_data_loader):
=======
    for i_batch, (x, y, index) in enumerate(dataloader):
        # print(x.shape)
>>>>>>> Prototype baseline
        print(i_batch)
        x = x.to(device)
        y = y.to(device)
        y_pred = resnet(x)
        emb_dict.append(y_pred.cpu().detach().numpy())
        # print(len(dataloader))
        # print(len(emb_dict))

    emb_dict = np.vstack(emb_dict)
<<<<<<< 88b820bc002f9a6f259d855fa30cbb78c1dd752e
    print(emb_dict.shape)
    np.save('pretrained_embeddings.npy',emb_dict)
=======
    print(f"Saving to {output_path}")
    np.save(output_path, emb_dict)
>>>>>>> Prototype baseline
    return emb_dict
        
