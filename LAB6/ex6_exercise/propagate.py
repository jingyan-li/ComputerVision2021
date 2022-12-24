import numpy as np

def propagate(particles, frame_height, frame_width, params):
    # stochastic components [k, 2]
    var_pos = np.random.normal(0, params["sigma_position"], (params["num_particles"], 2))
    var_vol = np.random.normal(0, params["sigma_velocity"], (params["num_particles"], 2))

    # get particles
    if params["model"] == 0:
        A = np.eye(2)
        particles = particles @ A + var_pos  # [k, 2]
    else:
        A = np.eye(4)
        A[0,2] = 1
        A[1,3] = 1
        particles = particles @ A + np.concatenate([var_pos, var_vol], axis=1)  # [k, 4]

    # Ensure particles inside image
    particles[:, 0] = np.minimum(particles[:, 0], frame_width - 1)
    particles[:, 0] = np.maximum(particles[:, 0], 0)
    particles[:, 1] = np.minimum(particles[:, 1], frame_height - 1)
    particles[:, 1] = np.maximum(particles[:, 1], 0)

    return particles