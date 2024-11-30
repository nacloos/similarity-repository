import numpy as np

from mouse_vision.core.constants import NUM_NS_STIM

def map_from_str(map_type):
    if map_type.lower() == "pls":
        from mouse_vision.neural_mappers import PLSNeuralMap
        return PLSNeuralMap
    elif map_type.lower() == "corr":
        from mouse_vision.neural_mappers import CorrelationNeuralMap
        return CorrelationNeuralMap
    elif map_type.lower() == "factored":
        from mouse_vision.neural_mappers import FactoredNeuralMap
        return FactoredNeuralMap
    elif map_type.lower() == "identity":
        from mouse_vision.neural_mappers import IdentityNeuralMap
        return IdentityNeuralMap
    else:
        raise ValueError(f"{map_type.lower()} is not supported.")

def generate_train_test_img_splits(num_splits=10, train_frac=0.5, num_imgs=NUM_NS_STIM):
    if train_frac > 0:
        train_test_splits = []
        for s in range(num_splits):
            rand_idx = np.random.RandomState(seed=s).permutation(num_imgs)
            num_train = (int)(np.ceil(train_frac*len(rand_idx)))
            train_idx = rand_idx[:num_train]
            test_idx = rand_idx[num_train:]
            curr_sp = {'train': train_idx, 'test': test_idx}
            train_test_splits.append(curr_sp)
    else:
        print("Train fraction is 0, make sure your map has no parameters!")
        # we apply no random permutation in this case as there is no training of parameters
        # (e.g. rsa)
        train_test_splits = [{'train': np.array([], dtype=int), 'test': np.arange(num_imgs)}]
    return train_test_splits

