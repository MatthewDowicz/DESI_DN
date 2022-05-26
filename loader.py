import numpy as np

def load(fname):
    dat = np.load('./Data/'+fname)
    return dat
