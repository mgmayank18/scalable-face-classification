def triplet_loss(embeddings, support_labels):
    """
    Given embeddings and labels, calculates triplet loss
    Parameters:
        labels:  label for anchor
        embeddings: batch of embeddings with at least 2 examples per class - torch tensor
    """    
    positive = np.zeros_like(embeddings)
    negative = np.zeros_like(embeddings)
    
    for i in range(embeddings.size(0)):
        pairwise_dist = np.linalg.norm(embeddings - embeddings[i] , axis=1)
        a_label = support_labels[i]

        pos_mask = np.where(support_labels == a_label, support_labels,  0)
        pos_distances = pairwise_dist * pos_mask
        pos_index = np.argmax(pos_distances)
        positive[i] = embeddings[pos_index]
        
        neg_mask = np.where(support_labels != a_label, support_labels,  0)
        neg_distances = pairwise_dist * neg_mask
        neg_index = np.argmin(pos_distances)
        negative[i] = embeddings[neg_index]
    
    criterion = nn.TripletMarginLoss(margin=1.0)
#     print(positive)
#     print(negative)
#     print(anchors)
    loss = criterion(embeddings, positive, negative)
#     loss.backward()
    print("Triplet Loss: ", loss.item())
    
    return loss
