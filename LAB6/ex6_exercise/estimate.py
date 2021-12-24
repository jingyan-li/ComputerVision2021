import numpy as np


def estimate(particles, particles_w):
    mean_state = np.sum(particles * particles_w, axis=0)
    return mean_state