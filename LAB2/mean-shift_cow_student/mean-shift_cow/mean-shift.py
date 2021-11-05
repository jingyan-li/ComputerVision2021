import time
import os
import random
import math
import torch
import numpy as np

# run `pip install scikit-image` to install skimage, if you haven't done so
from skimage import io, color
from skimage.transform import rescale
#%%
def distance(x, X):
    # dist = torch.sqrt(torch.sum(torch.pow(X-x, 2), dim=1))
    dist = torch.linalg.norm(x - X, dim=-1)
    # raise NotImplementedError('distance function not implemented!')
    return dist

def distance_batch(x, X):
    dist_batch = torch.linalg.norm(x[:, None, :] - X[None, :, :], dim=-1)
    return dist_batch
    # raise NotImplementedError('distance_batch function not implemented!')

def gaussian(dist, bandwidth):
    val = (1 / (bandwidth * math.sqrt(2 * math.pi))) * torch.exp(-0.5 * ((dist / bandwidth)) ** 2)
    return val
    # raise NotImplementedError('gaussian function not implemented!')

def update_point(weight, X):
    denominator = torch.sum(weight)
    updated_x = torch.sum(torch.mul(torch.unsqueeze(weight, dim=1), X), dim=0) / denominator
    return updated_x
    # raise NotImplementedError('update_point function not implemented!')

def update_point_batch(weight, X):
    batch_denominator = torch.sum(weight, dim=1)
    updated_batch_x = torch.sum(torch.mul(torch.unsqueeze(weight, dim=2), X), dim=1) / torch.unsqueeze(batch_denominator, dim=1)
    # raise NotImplementedError('update_point_batch function not implemented!')
    return updated_batch_x

def meanshift_step(X, bandwidth=2.5):
    X_ = X.clone()
    for i, x in enumerate(X):
        dist = distance(x, X)
        weight = gaussian(dist, bandwidth)
        X_[i] = update_point(weight, X)
    return X_

def meanshift_step_batch(X, bandwidth=2.5):
    BATCH_SIZE = 400
    X_ = X.clone()
    for i in range(0, X.shape[0], BATCH_SIZE):
        end_idx = i+BATCH_SIZE if i+BATCH_SIZE < X.shape[0] else X.shape[0]-1
        batch_dist = distance_batch(X[i:end_idx], X)
        batch_weight = gaussian(batch_dist, bandwidth)
        X_[i:end_idx] = update_point_batch(batch_weight, X)
    return X_
    # raise NotImplementedError('meanshift_step_batch function not implemented!')

def meanshift(X):
    X = X.clone()
    for _ in range(20):
        # X = meanshift_step(X)   # slow implementation
        X = meanshift_step_batch(X)   # fast implementation
    return X
#%%
scale = 0.25    # downscale the image to run faster

# Load image and convert it to CIELAB space
image = rescale(io.imread('cow.jpg'), scale, multichannel=True)
image_lab = color.rgb2lab(image)
shape = image_lab.shape # record image shape
image_lab = image_lab.reshape([-1, 3])  # flatten the image
#%%
# Run your mean-shift algorithm
t = time.time()
X = meanshift(torch.from_numpy(image_lab)).detach().cpu().numpy()
# X = meanshift(torch.from_numpy(data).cuda()).detach().cpu().numpy()  # you can use GPU if you have one
t = time.time() - t
print ('Elapsed time for mean-shift: {}'.format(t))

# Load label colors and draw labels as an image
colors = np.load('colors.npz')['colors']
colors[colors > 1.0] = 1
colors[colors < 0.0] = 0

centroids, labels = np.unique((X / 4).round(), return_inverse=True, axis=0)

result_image = colors[labels].reshape(shape)
result_image = rescale(result_image, 1 / scale, order=0, multichannel=True)     # resize result image to original resolution
result_image = (result_image * 255).astype(np.uint8)
io.imsave('result_batch.png', result_image)
