#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np

data_raw = np.loadtxt('data_DMAP_PCA_vadere.txt', delimiter=' ')

# Normalize data to mean = 0 and standard deviation = 1
data = (data_raw - data_raw.mean(axis=0)) / data_raw.std(axis=0)
print('Data mean: ' + str(data.mean()))
print('Data std: ' + str(data.std()))
#n = samples, p = features
n, p = data.shape

# Apply SVD
U, S, V = np.linalg.svd(data, full_matrices=False)
Smat = np.diag(S)

# Calculate explained variance
explained_variance = (S**2 / (n-1))
explained_variance_ratio = explained_variance / np.sum(explained_variance)
explained_variance_sum = np.cumsum(explained_variance_ratio)
#print(explained_variance_ratio)
#print(explained_variance_sum)

# Keep first 2 coordinates of dataset after transformation
newdata = U.dot(Smat)
newdata = newdata[:,:2]
     
# Retrieve dataset 
# new = U[:,:2]*Sig*V[:2,:]

# Plot first two axis of transformed dataset 
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(newdata[:,0],newdata[:,1], color = 'b')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Plot of first 2 axis of transformed coordinates')
plt.show()

# Plot of reconstructed dataset, similar to task1_2
S_red = Smat[:2, :2]
#print(S_red.shape)
V_red = V[:2, :]
#print(V_red.shape)
U_red = U[:, :2]
#print(U_red.shape)
data_new = np.dot(U_red, np.dot(S_red, V_red))
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(data_new[:,0], data_new[:,1])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Plot of first 2 axis of reconstructed dataset')
plt.show()


# Double check PCA using sklearn
from sklearn.decomposition import PCA
# project from 64 to 2 dimensions
pca = PCA(n_components=2)
projected = pca.fit_transform(data)

# Plot results
fig = plt.figure()
plt.scatter(projected[:, 0], projected[:, 1])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Plot of first 2 axis using sklearn')
plt.show()

"""
# Another plot using seaborn
pc_df = pd.DataFrame(data=projected, columns = ['PC1', 'PC2'])
pc_df.head()

import seaborn as sns
sns.lmplot( x="PC1", y="PC2",
  data=pc_df, 
  fit_reg=False, 
  legend=True,
  scatter_kws={"s": 80}) # specify the point size
ax = plt.gca()
ax.set_title("Plot Using seaborn")
"""