import numpy as np

def load(fname):
    dat = np.load('/data17/grenache/staudt/desi/'+fname)
    return dat
