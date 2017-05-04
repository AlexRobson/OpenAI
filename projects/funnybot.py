""" 
Repo to explore suggested exercises in Open AI
"""

"""
Notes

Want to explore the latent space from which we can sample text/jokes
To create this latent space we want a generative model.
The set-up could work something like this:

- Create a character level LSTM.
- This works by, for each input char, predicitng the next char
- I.e. it trains in a supervised fasion from the training corpus,
using the {i-1}_th character in the string, to predict the {i}th.
It can then be sampled by giving a seed string

Useful links: http://karpathy.github.io/2015/05/21/rnn-effectiveness/

"""

import pdb
from sklearn.model_selection import train_test_split
import json
import numpy as np
from utils import modelrunner
from networks.funnybotlstm import Scorer, Generator
import yaml
from utils.utils import GroupDataByLength, generate_encoding, encode_text, byteify

GENERATOR = True

def set_config(FBL):

    config = {}
    config['num_epochs'] = 10
    config['shuffle'] = False
    config['batch_size'] = 128
    config['validate'] = False
    config['test'] = False

    setattr(FBL, 'config', config)


def run():
    data, coding = load_data()

    # Initialise the funnybot class
    if GENERATOR:
        FBL = Generator()
    else:
        FBL = Scorer()
    set_config(FBL)
    FBL.coding = coding
    FBL.initialise()

    # TODO: Convert modelrunner into a class
    modelrunner.run(data, functions=FBL.functions, CONFIG=FBL.config)





def create_snippets(sample, sniplength):
    """
    This function receives some sample text of arbitrary length, and then snips it
    into X, y samples of sample[i:i+sniplength], sample[i+sniplength+1], for all i
    :param sample:
    :return: dataX, datay
    """
    seq_length = len(sample)
    dataX = []
    datay = []
    if seq_length<sniplength:
        return [sample[0:-1]], [sample[-1]]
    for i in range(0, seq_length-sniplength, 1):
        seqX = sample[i:i+sniplength]
        seqy = sample[i+sniplength]
        dataX.append(seqX)
        datay.append(seqy)

    return dataX, datay


def parsedata(data,coding):

    textfields = ['body']
    scorefields = ['score']
    titlefield = ['title']

    X = [(encode_text(d['title']+d['body'], coding), d['score']) for d in data[0:1000]]
    X, y = zip(*X)
    if GENERATOR:
        X_dash = []
        y_dash = []
        for x in X:
            pX, py = create_snippets(x, 20)
            X_dash.extend(pX)
            y_dash.extend(py)

        # Now relabel
        X = X_dash
        y = y_dash

    else:
        # Normalise the bins
        _, bins = np.histogram(y)
        y = np.digitize(y, bins)



    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42)

    # Percentage training
    pdata = {}
    X_train_dict, y_train_dict = GroupDataByLength(X_train, y_train)
    X_valid_dict, y_valid_dict = GroupDataByLength(X_valid, y_valid)
    X_test_dict, y_test_dict = GroupDataByLength(X_test, y_test)
    pdata['train'] = X_train_dict, y_train_dict
    pdata['valid'] = X_valid_dict, y_valid_dict
    pdata['test'] = X_test_dict, y_test_dict

    return pdata



def load_data():
    import json

    froot = '../data/joke-dataset/'
    files = ['test.json', 'reddit_jokes.json', 'stupidstuff.json', 'wocka.json']
    fpath = froot+files[1]

    with open(fpath) as f:
        coding = generate_encoding(f.read())

    print("Length of the encoding is {}".format(len(coding['decoding'])))
    with open(fpath) as f:
        data = json.loads(f.read())
        data = parsedata(data, coding)

    return data, coding

if __name__=='__main__':
    run()

