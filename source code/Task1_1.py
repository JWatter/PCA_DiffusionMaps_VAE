#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import os

data_raw = np.loadtxt("./data/pca_dataset.txt", delimiter=' ')
samples, features = data_raw.shape

# Center data
data = data_raw - data_raw.mean(axis = 0)

#calculate SVD
U, S, V = np.linalg.svd(data)

#plt.scatter(data[:,0], data[:,1])


explained_variance = np.array((S**2) / 100) # or explained energy
data_mean = data.mean(axis=0)

#direction of the principal components
components = V

def draw_vector(v0, v1, ax=None):
    """
    Plots arrow annotation in a figure starting from Input v0 
    ending at v1
    This represents the direction of the PCs
    The length of the arrows represent the explained energy (in relation)
    """
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)

# plot data and PCAs
plt.scatter(data[:, 0], data[:, 1], alpha=0.2)
for length, vector in zip(explained_variance, components):
    print('length, vector:', length, vector)
    v = vector *2* np.sqrt(length)
    #print('v:', v)
    #print(data_mean, data_mean + v)
    draw_vector(data_mean, data_mean + v)
plt.show()

"""
print(explained_variance)
print(components)
for l,v in zip(explained_variance, components):
    print('l', l)
    print('v', v)
"""
