def triplet_loss(query_embeddings, labels, embeddings, support_labels):
    """
    Given an anchor, finds P and N, calculates Triplet Loss and backward gradient
    
    Parameters:
        query_embeddings: feature embedding for the anchor
        labels:  label for anchor
        embeddings: np array of normalized embeddings - assume same order as dataset
        support_labels:  self explanatory
    
    Returns:
        loss: calculated loss (gradients are backpropagated within the fn itself)
    """    
    anchors = query_embeddings
    positive = np.zeros_like(anchors)
    negative = np.zeros_like(anchors)
    
#     support_labels = SOMETHING

    print(anchors.size)
    
    for i in range(anchors.shape[0]):
        # biometric -> get_embeddings -> query + base embeddings
        # get predicted labels from query using check_faces
        pairwise_dist = np.linalg.norm(embeddings - anchors[i] , axis =1)

        a_label = labels[i]
        
        print("alabale", a_label)

        pos_mask = np.where(support_labels == a_label, support_labels,  0)
        print("pos mask", pos_mask)
        pos_distances = pairwise_dist * pos_mask
        pos_index = np.argmax(pos_distances)
        print("pos index", pos_index)
        positive[i] = embeddings[pos_index]
        
        neg_mask = np.where(support_labels != a_label, support_labels,  0)
        print("neg mask", neg_mask)
        neg_distances = pairwise_dist * neg_mask
        neg_index = np.argmin(pos_distances)
        print("neg index", neg_index)
        negative[i] = embeddings[neg_index]
    
    criterion = nn.TripletMarginLoss(margin=1.0)
    print(positive)
    print(negative)
    print(anchors)
    loss = criterion(anchors, positive, negative)
#     loss.backward()
    print("Loss: ", loss.item())
    
    return loss
