"""Define functions to create the triplet loss with online triplet mining."""
from tensorflow.keras import backend as K

import tensorflow as tf
import numpy as np
import gc


def cosine_similarity_two_classes(embeddings_class1, embeddings_class2):
    return tf.minimum(tf.matmul(embeddings_class1, tf.transpose(embeddings_class2)), 1.0)

def cosine_similarity_two_classes_without_tf(embeddings_class1, embeddings_class2):
    return np.minimum(np.matmul(embeddings_class1, np.transpose(embeddings_class2)), 1.0)

def get_similarity_vector(class1_emb, class2_emb, same_class=False, use_tf=True):
    if same_class:
        class1_emb = np.squeeze(class1_emb)

        if use_tf:
            cos_sim_intra = cosine_similarity_two_classes(class1_emb, class1_emb)
        else:
            cos_sim_intra = cosine_similarity_two_classes_without_tf(class1_emb, class1_emb)

        cos_sim_intra = np.triu(cos_sim_intra)
        np.fill_diagonal(cos_sim_intra, 0)
        cos_sim_intra = cos_sim_intra.flatten()

        cos_sim_intra = [i for i in cos_sim_intra if i != 0]

        return cos_sim_intra

    else:
        if len(class1_emb.shape) > 2:
            class1_emb = np.squeeze(class1_emb)
        if len(class2_emb.shape) > 2:
            class2_emb = np.squeeze(class2_emb)

        if use_tf:
            cos_sim_c1_c2 = cosine_similarity_two_classes(class1_emb, class2_emb)
            cos_sim_c1_c2 = list(cos_sim_c1_c2.numpy().flatten())

            K.clear_session()
            gc.collect()

        else:
            cos_sim_c1_c2 = cosine_similarity_two_classes_without_tf(class1_emb, class2_emb)
            cos_sim_c1_c2 = list(cos_sim_c1_c2.flatten())

        return cos_sim_c1_c2

def _cosine_similarity(embeddings):
    """Compute the 2D matrix of distances between all the embeddings.

    Args:
        embeddings: tensor of shape (batch_size, embed_dim)

    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    cosine_similarity = tf.matmul(embeddings, tf.transpose(embeddings))

    return cosine_similarity

def _get_triplet_mask(labels):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.

    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]

    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    """
    # Check that i, j and k are distinct

    labels = tf.reshape(labels, [tf.shape(labels)[0]])
    shape_labels = tf.shape(labels)
    indices_equal = tf.cast(tf.eye(shape_labels[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
    i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
    j_not_equal_k = tf.expand_dims(indices_not_equal, 0)

    distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)


    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    i_equal_j = tf.expand_dims(label_equal, 2)
    i_equal_k = tf.expand_dims(label_equal, 1)

    valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))

    # Combine the two masks
    mask = tf.logical_and(distinct_indices, valid_labels)

    return mask

def batch_all_cosine_triplet_loss(labels, embeddings, margin=0.1):
    """Build the triplet loss over a batch of embeddings.

    We generate all the valid triplets and average the loss over the positive ones.

    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    # labels = tf.Print(labels, [labels], 'labels')

    cosine_dist = _cosine_similarity(embeddings)

    # shape (batch_size, batch_size, 1)
    anchor_positive_dist = tf.expand_dims(cosine_dist, 2)
    assert anchor_positive_dist.shape[2] == 1, "{}".format(anchor_positive_dist.shape)
    # shape (batch_size, 1, batch_size)
    anchor_negative_dist = tf.expand_dims(cosine_dist, 1)
    assert anchor_negative_dist.shape[1] == 1, "{}".format(anchor_negative_dist.shape)

    # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
    # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
    # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
    # and the 2nd (batch_size, 1, batch_size)
    triplet_loss = anchor_negative_dist - anchor_positive_dist + margin

    # Put to zero the invalid triplets and count valid triplets
    # (where label(a) != label(p) or label(n) == label(a) or a == p)
    mask = _get_triplet_mask(labels)
    mask = tf.cast(mask, tf.float32)
    num_valid_triplets = tf.reduce_sum(mask)
    triplet_loss = tf.multiply(mask, triplet_loss)

    # Remove negative losses (i.e. the easy triplets)
    triplet_loss = tf.maximum(triplet_loss, 0.0)

    # Count number of positive triplets (where triplet_loss > 0)
    positives_triplets = tf.cast(tf.greater(triplet_loss, 1e-16), tf.float32)
    num_positive_triplets = tf.reduce_sum(positives_triplets)

    #num_valid_triplets = tf.Print(num_valid_triplets, [num_valid_triplets], 'num_valid_triplets')

    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = (tf.reduce_sum(triplet_loss) / (num_valid_triplets + 1e-16)) * 1000000

    return triplet_loss

def batch_all_cosine_accuracy(labels, embeddings, margin=0.1):
    # Get the pairwise distance matrix
    #labels = tf.Print(labels, [labels], 'labels')

    cosine_dist = _cosine_similarity(embeddings)

    # shape (batch_size, batch_size, 1)
    anchor_positive_dist = tf.expand_dims(cosine_dist, 2)
    assert anchor_positive_dist.shape[2] == 1, "{}".format(anchor_positive_dist.shape)
    # shape (batch_size, 1, batch_size)
    anchor_negative_dist = tf.expand_dims(cosine_dist, 1)
    assert anchor_negative_dist.shape[1] == 1, "{}".format(anchor_negative_dist.shape)

    # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
    # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
    # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
    # and the 2nd (batch_size, 1, batch_size)
    triplet_loss = anchor_negative_dist - anchor_positive_dist + margin

    # Put to zero the invalid triplets and count valid triplets
    # (where label(a) != label(p) or label(n) == label(a) or a == p)
    mask = _get_triplet_mask(labels)
    mask = tf.cast(mask, tf.float32)
    num_valid_triplets = tf.reduce_sum(mask)
    triplet_loss = tf.multiply(mask, triplet_loss)

    # Remove negative losses (i.e. the easy triplets)
    triplet_loss = tf.maximum(triplet_loss, 0.0)

    # Count number of positive triplets (where triplet_loss > 0)
    positives_triplets = tf.cast(tf.greater(triplet_loss, 1e-16), tf.float32)
    num_positive_triplets = tf.reduce_sum(positives_triplets)

    #num_valid_triplets = tf.Print(num_valid_triplets, [num_valid_triplets], 'num_valid_triplets')

    fraction_negative_triplets = (num_valid_triplets - num_positive_triplets) / (num_valid_triplets + 1e-16)

    return fraction_negative_triplets
