#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

data_raw = np.loadtxt('../data/data_DMAP_PCA_vadere.txt', delimiter=' ')

"""
Part 1 - Visualization of trajectories
"""
data_x1 = data_raw[:, :2]
data_x2 = data_raw[:, 2:4]

# Split data for the pedestrians in separate x and y arrays
x_1 = data_x1[:,0]
y_1 = data_x1[:,1]
x_2 = data_x2[:,0]
y_2 = data_x2[:,1]

fig, ax = plt.subplots()

line_1, = ax.plot(x_1, y_1, linewidth=0.5, color='k')
line_2, = ax.plot(x_2, y_2, linewidth=0.5, color='r')

def update(num, x, y, line):
    """
    update function for matplotlib animation
    Plots trajectory for inputs x and y with attributes given by line
    """
    line.set_data(x[:num], y[:num])
    line.axes.axis([0, 18, 10, 25])
    return line,

# Start animation
animation.FuncAnimation(fig, update, len(x_1),
                              fargs=[x_1, y_1, line_1],
                              interval=20, blit=True, repeat=False)

animation.FuncAnimation(fig, update, len(x_2), 
                              fargs=[x_2, y_2, line_2],
                              interval=20, blit=True, repeat=False)

plt.show()

"""
Part 2 - PCA
"""
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

explained_variance_sk = pca.explained_variance_ratio_
expl_var_cumsum_sk = np.cumsum(explained_variance_sk)

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