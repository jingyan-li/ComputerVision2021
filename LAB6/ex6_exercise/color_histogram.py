import numpy as np


def color_histogram(xmin, ymin, xmax, ymax, frame, hist_bin):
    """
    @param xmin: bounding box in WIDTH position[0]
    @param ymin: bounding box in HEIGHT position[1]
    @param xmax: bounding box
    @param ymax; bounding box
    @param frame: Frame to define bounding box; shape [H,W,3]
    @param hist_bin: color histogram number of bins;

    return hist [3, hist_bin]
    """
    H, W = frame.shape[0], frame.shape[1]

    # Cut the bbox if it goes out of index
    xmin = round(max(0, xmin))
    ymin = round(max(0, ymin))
    xmax = round(min(xmax, W-1))
    ymax = round(min(ymax, H-1))

    # Get bounding box
    bbox = frame[ymin:ymax, xmin:xmax]
    # Check if there exists some pixels
    # if bbox.shape[0] * bbox.shape[1] == 0:
    #     return np.zeros((3, hist_bin))

    # Generate hist
    hist_rgb = []
    for i in range(3):
        hist, _ = np.histogram(bbox[:, :, i].reshape(-1), bins=hist_bin, range=(0, 255))
        hist_rgb.append(hist)
    hist_rgb = np.asarray(hist_rgb)

    # Normalize
    hist_rgb = hist_rgb/np.sum(hist_rgb, axis=1)[:, None]
    # hist_rgb = hist_rgb/np.sum(hist_rgb)
    return hist_rgb