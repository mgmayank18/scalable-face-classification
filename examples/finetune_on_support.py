from inception_resnet_v1 import InceptionResnetV1_LWF
from utils import load_orig_task_weights
import torch

#target_dict = load_target_dict()

model = InceptionResnetV1_LWF(
    classify=False,
    pretrained=None,
    num_classes=8631
).to('cuda')

model = load_orig_task_weights(model)


