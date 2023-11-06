import numpy as np
import scipy
import scipy.spatial
import scipy.ndimage
import pathlib

def make_sure_folder_exists(path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
# the function to generate the density map, with provided points
# from https://github.com/TencentYoutuResearch/CrowdCounting-SASNet
def generate_density_map(shape=(5, 5), points=None, f_sz=15, sigma=4):
    """
    generate density map given head coordinations
    """
    im_density = np.zeros(shape[0:2])
    h, w = shape[0:2]
    if len(points) == 0:
        return im_density
    # iterate over all the points
    for j in range(len(points)):
        # create the gaussian kernel
        H = matlab_style_gauss2D((f_sz, f_sz), sigma)
        # limit the bound
        x = np.minimum(w, np.maximum(1, np.abs(np.int32(np.floor(points[j, 0])))))
        y = np.minimum(h, np.maximum(1, np.abs(np.int32(np.floor(points[j, 1])))))
        if x > w or y > h:
            continue
        # get the rect around each head
        x1 = x - np.int32(np.floor(f_sz / 2))
        y1 = y - np.int32(np.floor(f_sz / 2))
        x2 = x + np.int32(np.floor(f_sz / 2))
        y2 = y + np.int32(np.floor(f_sz / 2))
        dx1 = 0
        dy1 = 0
        dx2 = 0
        dy2 = 0
        change_H = False
        if x1 < 1:
            dx1 = np.abs(x1) + 1
            x1 = 1
            change_H = True
        if y1 < 1:
            dy1 = np.abs(y1) + 1
            y1 = 1
            change_H = True
        if x2 > w:
            dx2 = x2 - w
            x2 = w
            change_H = True
        if y2 > h:
            dy2 = y2 - h
            y2 = h
            change_H = True
        x1h = 1 + dx1
        y1h = 1 + dy1
        x2h = f_sz - dx2
        y2h = f_sz - dy2
        if change_H:
            H = matlab_style_gauss2D((y2h - y1h + 1, x2h - x1h + 1), sigma)
        # attach the gaussian kernel to the rect of this head
        im_density[y1 - 1 : y2, x1 - 1 : x2] = im_density[y1 - 1 : y2, x1 - 1 : x2] + H
    return im_density


def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]
    h = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

# modify from https://github.com/davideverona/deep-crowd-counting_crowdnet
def gaussian_filter_density(gt):
    print(gt.shape)
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density

    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    leafsize = 2048
    # build kdtree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(pts, k=4)

    print("generate density...")
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1], pt[0]] = 1.0
        if gt_count > 1:
            sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1
        else:
            sigma = np.average(np.array(gt.shape)) / 2.0 / 2.0  # case: 1 point
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode="constant")
    print("done.")
    return density