""" 
Repo to explore suggested exercises in Open AI
"""

import pdb
import json

def run():
    load_data()

def load_data():
    import json

    froot = '../data/joke-dataset/'
    files = ['reddit_jokes.json', 'stupidstuff.json', 'wocka.json']

    fpath = froot+files[0]
    with open(fpath) as f:
        data = json.load(f)
    
    pdb.set_trace()


if __name__=='__main__':
    run()

