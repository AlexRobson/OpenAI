from collections import Counter
import numpy as np
import pdb

def generate_encoding(text):

    coding = {}
    chars = list(set(text.decode('unicode_escape')))
    chars = [char for char in chars if ord(char) < 128]

    coding['decoding'] = { i:ch for i,ch in enumerate(chars)}
    coding['encoding'] = { ch:i for i,ch in enumerate(chars)}

    return coding


def encode_text(text, coding):

    encoded_text = []
    encoding = coding['encoding']
    encoded_text = [encoding[char] for char in text if char in encoding]

    assert len(encoded_text)>0


    return encoded_text

def byteify(input):
    if isinstance(input, dict):
        return {byteify(key): byteify(value)
                for key, value in input.iteritems()}
    elif isinstance(input, list):
        return [byteify(element) for element in input]
    elif isinstance(input, unicode):
        return input.encode('utf-8')
    else:
        return input

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


