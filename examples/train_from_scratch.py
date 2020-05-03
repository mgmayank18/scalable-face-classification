from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import numpy as np
import os
import torch.nn as nn
from utils import xavier_init

data_dir = '../data/VGGFace2/train'

batch_size = 500
epochs = 10
workers = 0 if os.name == 'nt' else 24

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))
'''
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

dataset = datasets.ImageFolder(data_dir, transform=transforms.Resize((512, 512)))
dataset.samples = [
    (p, p.replace(data_dir, data_dir + '_cropped'))
        for p, _ in dataset.samples
]
        
loader = DataLoader(
    dataset,
    num_workers=workers,
    batch_size=batch_size,
    collate_fn=training.collate_pil
)

for i, (x, y) in enumerate(loader):
    mtcnn(x, save_path=y)
    print('\rBatch {} of {}'.format(i + 1, len(loader)), end='')
    
# Remove mtcnn to reduce GPU memory usage
del mtcnn
"Done MTCNN"
'''

resnet = InceptionResnetV1(
    classify=True,
    pretrained=None,
    num_classes=5631
)

print("Xavier Init...")
resnet = xavier_init(resnet)
print("Done...")

trans = transforms.Compose([
    np.float32,
    transforms.ToTensor(),
    fixed_image_standardization
])
print("Creating Dataset (Cropped Images)")
dataset = datasets.ImageFolder(data_dir + '_cropped', transform=trans)
print("Done.")


resnet = nn.DataParallel(resnet.to(device))
#weights = torch.load('./saved_models/lr_0.1/best_lr_0.1.pt')
#resnet.load_state_dict(weights)

optimizer = optim.AdamW(resnet.parameters(), lr=0.1)

#optimizer = optim.SGD(resnet.parameters(), lr=0.1)
scheduler = MultiStepLR(optimizer, [10, 15])
#scheduler = None

img_inds = np.arange(len(dataset))
np.random.seed(0)
np.random.shuffle(img_inds)
train_inds = img_inds[:int(0.96 * len(img_inds))]
val_inds = img_inds[int(0.96 * len(img_inds)):]

train_loader = DataLoader(
    dataset,
    num_workers=workers,
    batch_size=batch_size,
    sampler=SubsetRandomSampler(train_inds)
)
val_loader = DataLoader(
    dataset,
    num_workers=workers,
    batch_size=batch_size,
    sampler=SubsetRandomSampler(val_inds)
)

loss_fn = torch.nn.CrossEntropyLoss()
metrics = {
    'fps': training.BatchTimer(),
    'acc': training.accuracy
}

writer = SummaryWriter()
writer.iteration, writer.interval = 0, 10

print('\n\nInitial')
print('-' * 10)

#print("Loading Weights...")
#weights=torch.load('./epoch_59.pt')
#resnet.load_state_dict(weights)

resnet.eval()
training.pass_epoch(
    resnet, loss_fn, val_loader,
    batch_metrics=metrics, show_running=True, device=device,
    writer=writer
)

for epoch in range(epochs):
    print('\nEpoch {}/{}'.format(epoch + 1, epochs))
    print('-' * 10)

    resnet.train()
    training.pass_epoch(
        resnet, loss_fn, train_loader, optimizer, scheduler,
        batch_metrics=metrics, show_running=True, device=device,
        writer=writer
    )
    
    print("Saving Model...")
    torch.save(resnet.state_dict(), './saved_models/epoch_'+str(epoch+1)+'.pt')

    resnet.eval()
    training.pass_epoch(
        resnet, loss_fn, val_loader,
        batch_metrics=metrics, show_running=True, device=device,
        writer=writer
    )

writer.close()
