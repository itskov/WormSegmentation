import pandas as pd

from scipy.io import loadmat
from sklearn.manifold.t_sne import TSNE

chris_data = loadmat('/home/itskov/Downloads/dataForEyal.mat')
tnse_manifold = TSNE().fit_transform(chris_data['PCs'])
pass