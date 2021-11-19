import numpy as np

def BuildProjectionConstraintMatrix(points2D, points3D):

  # TODO
  # For each correspondence, build the two rows of the constraint matrix and stack them

  num_corrs = points2D.shape[0]
  constraint_matrix = np.zeros((num_corrs * 2, 12))

  for i in range(num_corrs):
    # TODO Add your code here
    x, y, z = points3D[i, 0], points3D[i, 1], points3D[i, 2]
    u, v = points2D[i, 0], points2D[i, 1]
    constraint_matrix[2*i] = [x, y, z, 1, 0, 0, 0, 0, -u * x, -u * y, -u * z, -u]
    constraint_matrix[2*i+1] = [0, 0, 0, 0, x, y, z, 1, -v * x, -v * y, -v * z, -v]
  return constraint_matrix