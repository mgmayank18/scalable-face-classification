from triplet_loss import triplet_loss
import torch

def finetune_on_support(model, Dataloader, orig_target_dict):
    for batch_id, data in enumerate(Dataloader):
        target_old_task = orig_target_dict(data['paths'])
        orig_feats, new_feats = model(data['data'])
        labels = data['labels']

        L_old = MSE(orig_feats,target_old_task)
        L_new = triplet_loss(new_feats, labels)
        L_total = L_old+L_new
