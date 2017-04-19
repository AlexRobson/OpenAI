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
from utils import modelrunner
from networks import funnybotlstm
from utils.utils import GroupDataByLength

def set_config(FBL):

    config = {}
    config['num_epochs'] = 50
    config['shuffle'] = False
    config['batch_size'] = 1

    setattr(FBL, 'config', config)


def run():
    data = load_data()
    FBL = funnybotlstm.funnybotlstm()

    set_config(FBL)

    return data
#    modelrunner.run(data, CONFIG=FBL.config)

def parsedata(data):

    textfields = ['body']
    scorefields = ['score']
    titlefield = ['title']

    X = [(d['body'], d['score']) for d in data]

    X, y = zip(*X)

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

    pdb.set_trace()
    return pdata



def load_data():
    import json

    froot = '../data/joke-dataset/'
    files = ['reddit_jokes.json', 'stupidstuff.json', 'wocka.json']

    fpath = froot+files[0]
    with open(fpath) as f:
        data = json.load(f)
        data = parsedata(data)

    return data

if __name__=='__main__':
    data = run()

