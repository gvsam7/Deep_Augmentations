import numpy as np


def mask(img, size=96, n_squares=1):
    h, w, channels = img.shape
    new_img = img
    for _ in range(n_squares):
        y = np.random.randint(h)
        x = np.random.randint(w)
        y1 = np.clip(y - size // 2, 0, h)
        y2 = np.clip(y + size // 2, 0, h)
        x1 = np.clip(x - size // 2, 0, w)
        x2 = np.clip(x + size // 2, 0, w)
        new_img[y1:y2, x1:x2, :] = 0
    return new_img





