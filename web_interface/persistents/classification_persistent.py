import pickle
import os


def classification_persistent(folder_name: str):
    images = pickle.load(open(os.path.join(folder_name, "images.pydump"), 'rb'))
    labels = pickle.load(open(os.path.join(folder_name, "labels.pydump"), 'rb'))
    return images, labels
