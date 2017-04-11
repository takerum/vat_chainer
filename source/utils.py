import os, errno, sys, pickle
from scipy import linalg
import numpy as np


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def load_npz_as_dict(path):
    data = np.load(path)
    return {key: data[key] for key in data}






