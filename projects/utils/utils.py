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
    for char in text:
        encoded_text.append(encoding[char])

	return encoded_text

def GroupDataByLength(X, y, type='list'):

    N = []
    for sample in X:
        N.append(len(sample))

    N = np.array(N)
    counts = Counter(N)
    X_dict = {}
    y_dict = {}

    for count in counts:
        IDX = np.ravel(np.where(N==count))
        if type is 'list':
            X_dict[count] = [X[idx] for idx in IDX]
            y_dict[count] = [y[idx] for idx in IDX]
        elif type is 'array':
            raise NotImplementedError

    return X_dict, y_dict


