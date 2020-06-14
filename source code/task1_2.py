# In[0]:
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc

#convert to grayscale image
face_raw = scipy.misc.face(gray=True)

from skimage.transform import resize
#resize image to (249x185)
face = resize(face_raw, (185,249), anti_aliasing=True)

# Normalize data to mean = 0 and standard deviation = 1
face_normed = (face - face.mean(axis=0)) / face.std(axis=0)
face_normed_mean = face_normed.mean()
face_normed_std = face_normed.std()
n, p = face_normed.shape

# In[1]: Calculate SVD
U, S, V = np.linalg.svd(face_normed, full_matrices=False)
Smat = np.diag(S)
# throwaways dimensions we dont need
#data_reduced = U[:,:1]
#data_reduced

# In[2]: Calculate explained variance for each component in S
explained_variance_self = (S**2 / (n-1))
explained_variance_self_ratio = explained_variance_self / np.sum(explained_variance_self)
explained_variance_self_sum = np.cumsum(explained_variance_self_ratio)

# In[3]: Reconstruct image

def reconstruct(L):
    """
    L: Number of principal components
    Returns reconstructed image with given number L of PCs using
    X = U*S*V.T    
    """
    S_red = Smat[:L, :L]
    #print(S_red.shape)
    V_red = V[:L, :]
    #print(V_red.shape)
    U_red = U[:, :L]
    #print(U_red.shape)
    face_new = np.dot(U_red, np.dot(S_red, V_red))
    #plt.imshow(face_new)
    return face_new

# In[4]: Print reconstructed images

for i in [185,120,50,10]:
    plot_image = reconstruct(i)
    plt.figure()
    plt.title('Image reconstructed with ' + str(i) + ' Principal Components')
    plt.imshow(plot_image)


# In[5]: Check results using sklearn
"""      
from sklearn.preprocessing import StandardScaler
face = StandardScaler().fit_transform(face)
facemean = face.std()

from sklearn.decomposition import PCA
pca = PCA(n_components=185)
pca.fit(face)

#plt.imshow(face)
explained_variance = pca.explained_variance_ratio_
expl_var_cumsum = np.cumsum(explained_variance)
"""
