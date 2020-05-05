from triplet_loss import batch_hard_triplet_loss, batch_all_triplet_loss
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn as nn
import torch
import numpy as np
import pdb
import sys

def finetune_on_support(model, Dataloader, orig_target_dict, epochs=12, lr=0.01, logger=None, margin=0.5, Lambda=10):
    MSE = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = MultiStepLR(optimizer, [4, 8])
    model.train()

    for epoch in range(epochs):
        print("\n \n"+("-" * 20))
        print("Epoch : ", epoch)
        print("-" * 20)
        for batch_id, data in enumerate(Dataloader):
            imgs = data[0].cuda()
            labels = data[1].cuda()
            idxs = data[2]
            gt_label = data[3]

            optimizer.zero_grad()
            
            target_old_task = torch.Tensor(orig_target_dict[idxs,:]).cuda()
            orig_feats, new_feats = model(imgs)

            L_old = MSE(orig_feats,target_old_task)
            L_new, frac_valid_trip = batch_all_triplet_loss(labels, new_feats, margin=margin, squared=False)
            L_total = Lambda*L_old + L_new
            # L_total = L_new
            sys.stdout.write("\rLoss Total : %f L_trip : %f L_old : %f, Frac Valid Trips: %f"%(L_total.item(),L_new.item(), L_old.item(),frac_valid_trip.item()))
            sys.stdout.flush()

            if logger is not None:
                logger.add_scalar("loss/iter", L_total.item(), epoch*len(Dataloader)+batch_id)
                        
            L_total.backward()
            optimizer.step()
        scheduler.step()
        return L_old, L_new