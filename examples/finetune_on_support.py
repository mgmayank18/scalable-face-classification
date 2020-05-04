from triplet_loss import batch_hard_triplet_loss
import torch.nn as nn
import torch
import numpy as np
import pdb
import sys

def finetune_on_support(model, Dataloader, orig_target_dict, epochs=10, lr=0.00001, logger=None):
    MSE = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    model.train()

    for epoch in range(epochs):
        print("Epoch : ", epoch)
        for batch_id, data in enumerate(Dataloader):
            imgs = data[0].cuda()
            labels = data[1].cuda()
            idxs = data[2]
            gt_label = data[3]

            optimizer.zero_grad()
            
            target_old_task = torch.Tensor(orig_target_dict[idxs,:]).cuda()
            orig_feats, new_feats = model(imgs)

            L_old = MSE(orig_feats,target_old_task)
            L_new = batch_hard_triplet_loss(labels, new_feats, margin=0.1, squared=False, device='cuda')
            L_total = L_old + L_new
            sys.stdout.write("\rLoss Total : %f L_trip : %f L_old : %f"%(L_total.item(),L_new.item(), L_old.item()))
            sys.stdout.flush()
            print()

            if logger is not None:
                logger.add_scalar("loss/iter", L_total.item(), epoch*len(Dataloader)+batch_id)
                        
            L_total.backward()
            optimizer.step()

