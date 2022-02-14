from utils.utils import k_fold_split_train_val_test

import random ## For generating test data

def test_k_fold_split():
    """
    Generate 10 random dataset sizes from range [5,1000)

    The number returned indices should add up to the dataset size

    Each set of indices should be unique and not contain duplicates - length of the union achieves this iff the first check passes
    """
    seed = random.randrange(1000000)
    dataset_sizes = random.choices(range(5, 1000), k=10)
    for dataset_size in dataset_sizes:
        for fold_num in range(5):
            train_inds, val_inds, test_inds = k_fold_split_train_val_test(dataset_size, fold_num, seed)
            # check correct number of inds returned
            assert(len(train_inds)+len(val_inds)+len(test_inds) == dataset_size)
            # check no overlapping or repeated inds
            s1, s2, s3 = set(train_inds), set(val_inds), set(test_inds)
            assert(len(set.union(s1, s2, s3)) == dataset_size)
