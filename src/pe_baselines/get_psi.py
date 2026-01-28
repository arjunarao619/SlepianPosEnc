import numpy as np
from .get_psi_helper import *
import scipy.io


"""
Spherical needlet basis extraction.
Reference: arxiv 1508.05406

The pix2ang data files are precomputed from HEALPix and stored in pix2ang_data/.
For higher resolution grids, regenerate these using the MATLAB scripts.
"""
def get_psi(B, j, k, theta, phi):
    j_max = j
    l_max = int(np.floor(B ** (j_max + 1)))

    bl_vector = get_bl_vector(B, j_max, l_max)
    Nside = get_Nside(2, j)
    sqrt_lambda = np.sqrt(4 * np.pi / (12 * Nside**2))

    # Use healpy for pixel to angle conversion
    num = str(int(Nside))
    mat_contents = scipy.io.loadmat(f'pix2ang_data/pix2ang_result_{num}.mat')['result']
    tp = np.ravel(np.array(mat_contents))
    theta_xi, phi_xi = tp[k-1][0], tp[k-1][1]

    # Convert from colatitude to latitude convention
    theta_xi = np.pi/2 - theta_xi

    x_xi, y_xi, z_xi = sph2cart(phi_xi, theta_xi, 1)

    psi = []
    for phi_itm, theta_itm in zip(phi, theta):
        x, y, z = sph2cart(phi_itm, np.pi/2 - theta_itm, 1)

        n = 1
        dist = np.zeros(n)

        for i in range(n):
            dist[i] = np.dot([x[0], y[0], z[0]], [x_xi, y_xi, z_xi])

        P = p_polynomial_value(n, l_max, dist)
        psi_itm = spneedlet_eval_fast(2, j, bl_vector, P, dist, sqrt_lambda)
        psi.append(psi_itm)

    return np.array(psi)

# if __name__ == '__main__':
#     theta = 41.34
#     phi = 2.32
#     print("RESULT: ", get_psi(2, 1, 2, theta, phi))