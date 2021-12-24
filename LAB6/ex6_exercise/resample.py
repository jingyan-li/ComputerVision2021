import numpy as np


def resample(particles, particles_w):
    K = particles.shape[0]
    sample_idx = np.random.choice(K, size=K, replace=True, p=particles_w.reshape(-1))
    new_particles = particles[sample_idx]
    new_particles_w = particles_w[sample_idx]
    # Normalization
    new_particles_w = new_particles_w/sum(new_particles_w)

    return new_particles, new_particles_w