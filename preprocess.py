import math
import pickle
import gzip
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


x = pd.read_csv('./data/x.csv')
y = pd.read_csv('./data/y.csv')


def check_100bins_5histones(x):
	return 

def aggragate_features(x, y):		#stretch out into 100X5 feature data 
	
	#raveling 100x5 for x to match with y shape
	x = x.iloc[:, 1:]
	y = y.iloc[:, 1:]
	size_x = x.shape[0] / 100
	x = np.split(x, size_x)
	# print(type(x[0]))	#pd df
	x = [i.values.ravel() for i in x]
	# print(len(x), x[0].shape, y.shape)


	X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=123)
	X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)
	print("Done preprocessing")
	print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

	return X_train, X_test, y_train, y_test

# X_train, X_test, y_train, y_test = aggragate_features(x.iloc[:800],y[:8])
X_train, X_test, y_train, y_test = aggragate_features(x,y)


#save as npy file for retrieval 
np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)

