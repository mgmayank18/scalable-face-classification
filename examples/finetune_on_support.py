from triplet_loss import triplet_loss
import torch.nn as nn

def finetune_on_support(model, Dataloader, orig_target_dict, epochs=10, lr=0.001, logger=None):
    MSE = nn.MSELoss()
    optimizer = nn.optim.AdamW(model.parameters(), lr=lr)
    
    for epoch in range(epochs)
        for batch_id, data in enumerate(Dataloader):

            optimizer.zero_grad()

            target_old_task = orig_target_dict(data['paths'])
            orig_feats, new_feats = model(data['data'])
            labels = data['labels']

            L_old = MSE(orig_feats,target_old_task)
            L_new = triplet_loss(new_feats, labels)
            L_total = L_old + L_new
            
            if logger is not None:
                logger.add_scalar("loss/iter", L_total.item(), epoch*len(Dataloader)+batch_id)
            
            L_total.backward()
            optimizer.step()
    
