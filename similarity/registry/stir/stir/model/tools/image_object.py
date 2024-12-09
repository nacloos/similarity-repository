import joblib, pickle


def save_object(index, image, label, path):
    obj = ImageObject(index, image, label)
    with open(path, 'wb') as handle:
        joblib.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_object(path):
    return joblib.load(path)


class ImageObject:

    def __init__(self, index, image, label):
        self.index = index
        self.image = image
        self.label = label
