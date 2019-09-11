"""
Featured-Based Image Metamorphosis
Beier 1992
"""
import numpy as np
from scipy import interpolate


class ImageMorpher:
    def __init__(self):
        pass

    def run(self, src, dst, pq1, pq2):
        h, w, c = src.shape
        trans_coord = np.meshgrid(range(h), range(w), indexing='ij')
        yy, xx = trans_coord[0].astype(np.float64), trans_coord[1].astype(np.float64)  # might need to switch

        xsum = xx * 0.
        ysum = yy * 0.
        wsum = xx * 0.
        for i in range(len(pq1) - 1):
            if i in {16, 21, 26, 30, 35, 47}:
                continue
            elif i is 41:
                j = 36
            elif i is 47:
                j = 42
            elif i is 59:
                j = 48
            elif i is 67:
                j = 60
            else:
                j = i + 1
            # Computes u, v
            p_x1, p_y1 = (pq1[i, 0], pq1[i, 1])
            q_x1, q_y1 = (pq1[j, 0], pq1[j, 1])
            qp_x1 = q_x1 - p_x1
            qp_y1 = q_y1 - p_y1
            qpnorm1 = (qp_x1 ** 2 + qp_y1 ** 2) ** 0.5

            u = ((xx - p_x1) * qp_x1 + (yy - p_y1) * qp_y1) / qpnorm1 ** 2
            v = ((xx - p_x1) * -qp_y1 + (yy - p_y1) * qp_x1) / qpnorm1

            # Computes x', y'
            p_x2, p_y2 = (pq2[i, 0], pq2[i, 1])
            q_x2, q_y2 = (pq2[j, 0], pq2[j, 1])
            qp_x2 = q_x2 - p_x2
            qp_y2 = q_y2 - p_y2
            qpnorm2 = (qp_x2 ** 2 + qp_y2 ** 2) ** 0.5

            x = p_x2 + u * (q_x2 - p_x2) + (v * -qp_y2) / qpnorm2  # X'(x)
            y = p_y2 + u * (q_y2 - p_y2) + (v * qp_x2) / qpnorm2  # X'(y)

            # Computes weights
            d1 = ((xx - q_x1) ** 2 + (yy - q_y1) ** 2) ** 0.5
            d2 = ((xx - p_x1) ** 2 + (yy - p_y1) ** 2) ** 0.5
            d = np.abs(v)
            d[u > 1] = d1[u > 1]
            d[u < 0] = d2[u < 0]
            W = (qpnorm1 ** 1 / (10 + d)) ** 1

            wsum += W
            xsum += W * x
            ysum += W * y

        x_m = xsum / wsum
        y_m = ysum / wsum
        vx = xx - x_m
        vy = yy - y_m
        vx[x_m < 1] = 0
        vx[x_m > w] = 0
        vy[y_m < 1] = 0
        vy[y_m > h] = 0

        vx = (vx + xx).astype(int)
        vy = (vy + yy).astype(int)
        vx[vx >= w] = w - 1
        vy[vy >= h] = h - 1

        warp = np.ones(src.shape)
        warp[yy.astype(int), xx.astype(int)] = src[vy, vx]
        # from skimage.io import imsave
        # imsave('output/transformed.jpg', warp / 255)

        return warp, vx, vy
