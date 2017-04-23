from collections import Counter
import numpy as np
import pdb

def generate_encoding(text):

    coding = {}
    chars = list(set(text.decode('unicode_escape')))
    coding['decoding'] = { i:ch for i,ch in enumerate(chars) }
    coding['encoding'] = { ch:i for i,ch in enumerate(chars) }

    return coding


def encode_text(text, encoding):

    encoded_text = []
    encoded_text = [encoding[char] for char in text]

    return encoded_text


def GroupDataByLength(X, y, itype='list'):

    N = []
    for sample in X:
        N.append(len(sample))

    N = np.array(N)
    vals = Counter(N)
    X_dict = {}
    y_dict = {}

    for val in vals:
        IDX = np.ravel(np.where(N==val))
        if itype is 'list':
            X_dict[val] = np.array([X[idx] for idx in IDX], dtype='float32')[:, :, None]
            y_dict[val] = np.array([y[idx] for idx in IDX], dtype='float32')[:, None]
        elif itype is 'array':
            raise NotImplementedError("Must always be an array")

    return X_dict, y_dict


