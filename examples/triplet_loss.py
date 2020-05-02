def triplet_loss(anchor, label, embeddings, database, criterion=None):
    """
    Given an anchor, finds P and N, calculates Triplet Loss and backward gradient
    
    Parameters:
        anchor: feature embedding for the anchor
        label:  label for anchor
        database: label to index
        embeddings: np array of normalized embeddings - assume same order as dataset
        criterion: triplet loss passed to the function or then instantiate everytime fn is called
    
    Returns:
        loss: calculated loss (gradients are backpropagated within the fn itself)
    """
    
    classes = database.keys()
    # if fn is a member of biometric class then use member var for classes and database
    
    pairwise_dist = np.linalg.norm(embeddings - anchor , axis =1)
    
    # assuming there is only one index for each class
    p_idx = database[label]
    positive = embeddings[p_idx]
    
    n_idx = np.argmin(pairwise_dist[:p_idx])
    if p_idx + 1 < len(embeddings):
        n2_idx = p_idx + 1 + np.argmin(pairwise_dist[p_idx+1:])
        if pairwise_dist[n2_idx] < pairwise_dist[n_idx]:
            n_idx = n2_idx
    
    negative = embeddings[n_idx]
    
    if criterion is None:
        criterion = nn.TripletMarginLoss(margin=1.0)
    
    loss = criterion(anchor, positive, negative)
    loss.backward()
    print("Loss: ", loss)
    
    return loss