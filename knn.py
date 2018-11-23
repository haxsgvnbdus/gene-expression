import numpy as np
import pandas as pd
import matplotlib.pylab as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score


#read npy files
X_train, y_train, X_test, y_test = np.load('X_train.npy'), np.load('y_train.npy'), np.load('X_test.npy'), np.load('y_test.npy')


#downsampling 
# X_train, y_train, X_test, y_test = X_train, y_train, X_test, y_test[:20]


#standardizing should use normalizer/standardscaler
normalizer = Normalizer()
X_train = normalizer.fit_transform(X_train)
X_test = normalizer.fit_transform(X_test)

#PCA 
pca = PCA(n_components=450)
X_train = pca.fit_transform(X_train)
X_test = pca.fit_transform(X_test)
