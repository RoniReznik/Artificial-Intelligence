#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder

feature_dict ={i:label for i,label in zip(range(4), ('sepal length', 'sepal width', 'petal length', 'petal width', ))}

df = pd.io.parsers.read_csv(filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None,sep=',',)

df.columns = [l for i,l in sorted(feature_dict.items())] + ['class label']

X = df.iloc[:,0:4].values
y = df['class label'].values

enc = LabelEncoder()
label_encoder = enc.fit(y)
y = label_encoder.transform(y) + 1

def mean_vectors(X, y):
    class_labels = np.unique(y)
    n_classes = class_labels.shape[0]
    mean_vectors = []
    for class_ in class_labels:
        mean_vectors.append(np.mean(X[y==class_], axis=0))
    return mean_vectors

def scatter_within(X,y):
    class_labels = np.unique(y)
    n_classes = class_labels.shape[0]
    n_features = X.shape[1]
    mean_vectors_ = mean_vectors(X,y)
    S_W = np.zeros((n_features, n_features))
    for class_, mean_vector in zip(class_labels, mean_vectors_):
        class_scatter_matrix = np.zeros((n_features, n_features))
        for row in X[y==class_]:
            row, mean_vector = row.reshape(n_features, 1), mean_vector.reshape(n_features, 1)
            class_scatter_matrix += (row-mean_vector).dot((row-mean_vector).T)
        S_W += class_scatter_matrix
    return S_W

def scatter_between(X,y):
    overall_mean = np.mean(X, axis=0)
    n_features = X.shape[1]
    mean_vectors_ = mean_vectors(X,y)
    S_B = np.zeros((n_features, n_features))
    for i, mean_vector in enumerate(mean_vectors_):
        n = X[y==i+1,:].shape[0]
        mean_vector = mean_vector.reshape(n_features, 1)
        overall_mean = overall_mean.reshape(n_features,1)
        S_B += n * (mean_vector - overall_mean).dot((mean_vector - overall_mean).T)
    return S_B


def components(eig_vals, eig_vecs, n_comp=2):
    n_features = X.shape[1]
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
    W = np.hstack([eig_pairs[i][1].reshape(4, 1) for i in range(0, n_comp)])
    return W

S_W, S_B = scatter_within(X, y), scatter_between(X, y)
eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
W = components(eig_vals, eig_vecs, n_comp=2)
print('EigVals: %s\n\nEigVecs: %s' % (eig_vals, eig_vecs))
print('\nW: %s' % W)

X_lda = X.dot(W)
for label,marker,color in zip(
        np.unique(y),('^', 's', 'o'),('blue', 'red', 'green')):
    plt.scatter(X_lda[y==label, 0], X_lda[y==label, 1],
                color=color, marker=marker)

