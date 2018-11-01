import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import argparse
from textwrap import wrap

def get_data(fname, plot = False, ax = None):
    img = plt.imread(fname)
    if plot:
        if ax is None:
            plt.figure()
            ax = plt.gca()
        ax.imshow(img)
        ax.set_title(fname)
    img = img[:,:,:3]

    if img.dtype == 'uint8':
        return np.array(img,dtype = np.float32)/255
    else:
        return img
