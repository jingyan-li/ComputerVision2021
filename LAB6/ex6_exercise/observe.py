import numpy as np
from color_histogram import color_histogram
from chi2_cost import chi2_cost


def observe(particles, frame, bbox_height, bbox_width, hist_bins, hist, sigma_observe):
    K = particles.shape[0]
    particles_weight = np.zeros((K, 1))
    hist = hist.reshape(-1)

    for i in range(K):
        # For each particle
        min_x = particles[i, 0] - bbox_width/2
        max_x = particles[i, 0] + bbox_width/2
        min_y = particles[i, 1] - bbox_height/2
        max_y = particles[i, 1] + bbox_height/2
        # Color histogram of current particle
        cur_par_hist = color_histogram(min_x, min_y, max_x, max_y, frame, hist_bins)
        cur_par_hist = cur_par_hist.reshape(-1)
        # Chi2 distance
        dist = chi2_cost(hist, cur_par_hist)

        # Update weight
        particles_weight[i] = 1/(np.sqrt(2*np.pi)*sigma_observe) * np.exp(-(np.power(dist, 2))/(2*np.power(sigma_observe, 2)))

    # Normalization
    particles_weight_norm = particles_weight/np.sum(particles_weight)
    return particles_weight_norm