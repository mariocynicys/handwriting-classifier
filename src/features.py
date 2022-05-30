import cv2
import sys
import math
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog as skimghog


FEATURES = {
  'lbp',
  'hog',
  'glcm',
  'cold',
  'hinge',
  'slopes_and_curves',
  'chain_codes_and_pairs',
}

def glcm(image):
    props = ['contrast', 'homogeneity', 'energy', 'correlation', 'entropy']
    angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    distances = [1]
    features = []
    glcm = graycomatrix(image, distances=distances,
                        angles=angles, levels=2,
                        symmetric=False, normed=True)
    for prop in props:
        if prop == 'entropy':
            # Since graycoprops doesn't support calculating entropy.
            features.append([-np.sum(glcm[:, :, d, a] * np.log(glcm[:, :, d, a]))
                             for d in range(len(distances)) 
                             for a in range(len(angles))])
        else:
            features.append(graycoprops(glcm, prop).ravel())
    return np.hstack(features)

def lbp(image, n_points=16, radius=2):
    lbp = local_binary_pattern(image, n_points, radius, method='nri_uniform')
    # [n * (n - 1) + 2] for uniform bp and [1] more bin for non uniform bp.
    n_bins = n_points * (n_points - 1) + 2 + 1
    return np.histogram(lbp.ravel(), bins=np.arange(n_bins + 1), density=True)[0]

def hog(image, resize_factor=(2289, 1600), **kwargs):
    hog_kwargs = {
        'orientations': 9,
        'pixels_per_cell': (16, 16),
        'cells_per_block': (1, 1),
        'transform_sqrt': False,
    }
    hog_kwargs.update(kwargs)
    image = cv2.resize(image.astype('float'), resize_factor)
    features = skimghog(image, **hog_kwargs)
    return features.ravel() / np.sum(features)

def chain_codes_and_pairs(image):
    # Get the image contours.
    contours = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
    # A direction to index map.
    DIR8S = {(1, 0): 0, (1, 1): 1, (0, 1): 2, (-1, 1): 3, (-1, 0): 4, (-1, -1): 5, (0, -1): 6, (1, -1): 7}
    # Create an 8-bin histogram for the chain codes.
    chain_code_hist = np.zeros(8)
    # Create a 8x8 matrix for the chain code pairs.
    chain_code_pairs_hist = np.zeros((8, 8))
    for contour in contours:
        # Skip these contours to avoid errors. They carry very few information anyway.
        if len(contour) < 2: continue
        previous_point = contour[1]
        previous_direction = DIR8S[tuple((previous_point - contour[0])[0])]
        # Don't forget to account for previous direction.
        chain_code_hist[previous_direction] += 1
        for point in contour[2:]:
            direction = DIR8S[tuple((point - previous_point)[0])]
            chain_code_hist[direction] += 1
            chain_code_pairs_hist[previous_direction, direction] += 1
            previous_point = point
            previous_direction = direction
    # Normalize the histograms.
    chain_code_hist /= np.sum(chain_code_hist)
    chain_code_pairs_hist = chain_code_pairs_hist.ravel() / np.sum(chain_code_pairs_hist)
    return np.append(chain_code_hist, chain_code_pairs_hist)

def slopes_and_curves(image):
    # Get the image contours.
    contours = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)[0]
    def bound_angle(minimum: float, maximum: float):
        def _bounder(angle: float):
            if angle < minimum:
                return angle + 180
            if angle >= maximum:
                return angle - 180
            return angle
        return _bounder
    slopes = [] # [-90, 90)
    curves = [] # [0, 190)
    for contour in contours:
        # Skip these contours to avoid errors. They carry very few information anyway.
        if len(contour) < 2: continue
        previous_point = contour[1]
        previous_slope = math.degrees(math.atan2(*(previous_point - contour[0])[0]))
        # Don't forget to account for that previous slope.
        slopes.append(previous_slope)
        for point in contour:
            slope = math.degrees(math.atan2(*(point - previous_point)[0]))
            slopes.append(slope)
            curves.append(slope - previous_slope)
            previous_point = point
            previous_slope = slope
    slopes_hist = np.histogram([*map(bound_angle(-90, 90), slopes)],
                               bins=np.arange(-90, 90 + 1, 180 / 8),
                               density=True)[0]
    curves_hist = np.histogram([*map(bound_angle(0, 180), curves)],
                               bins=np.arange(0, 180 + 1, 180 / 8),
                               density=True)[0]
    return np.append(slopes_hist, curves_hist)

# Credits for the following 2 features: https://github.com/Swati707/hinge_and_cold_feature_extraction

def hinge(image, n_angles=12, leg_len=25):
    bin_size = 360 // n_angles
    hist = np.zeros((n_angles, n_angles))

    contours = sorted(
        cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0],
        key=cv2.contourArea, reverse=True)

    for contour in contours:
        n_pixels = len(contour)
        if n_pixels <= leg_len: continue

        points = np.array([point[0] for point in contour])
        xs, ys = points[:, 0], points[:, 1]

        point_1s = np.array([points[(i + leg_len) % n_pixels] for i in range(n_pixels)])
        point_2s = np.array([points[(i - leg_len) % n_pixels] for i in range(n_pixels)])

        x1s, y1s = point_1s[:, 0], point_1s[:, 1]
        x2s, y2s = point_2s[:, 0], point_2s[:, 1]

        phi_1s = np.degrees(np.arctan2(y1s - ys, x1s - xs) + np.pi)
        phi_2s = np.degrees(np.arctan2(y2s - ys, x2s - xs) + np.pi)

        indices = np.where(phi_2s > phi_1s)[0]

        for i in indices:
            phi1 = int(phi_1s[i] // bin_size) % n_angles
            phi2 = int(phi_2s[i] // bin_size) % n_angles
            hist[phi1, phi2] += 1

    hist /= np.sum(hist)
    return hist[np.triu_indices_from(hist, k=1)]

def cold(image, approx_poly_factor=0.01, n_rho=7,
         n_angles=12, ks=np.arange(3, 8), max_cnts=1000,
         r_inner=5.0, r_outer=35.0):
    bin_size = 360 // n_angles
    n_bins = n_rho * n_angles

    contours = list(cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0])
    np.random.shuffle(contours)
    contours = contours[:max_cnts]

    rho_bins_edges = np.log10(np.linspace(r_inner, r_outer, n_rho))
    feature_vectors = np.zeros((len(ks), n_bins))

    for k in ks:
        hist = np.zeros((n_rho, n_angles))
        for contour in contours:
            epsilon = approx_poly_factor * cv2.arcLength(contour, True)
            contour = cv2.approxPolyDP(contour, epsilon, True)

            n_pixels = len(contour)

            point_1s = np.array([point[0] for point in contour])
            x1s, y1s = point_1s[:, 0], point_1s[:, 1]
            point_2s = np.array([contour[(i + k) % n_pixels][0]
                                 for i in range(n_pixels)])
            x2s, y2s = point_2s[:, 0], point_2s[:, 1]

            thetas = np.degrees(np.arctan2(y2s - y1s, x2s - x1s) + np.pi)
            rhos = np.sqrt((y2s - y1s) * 2 + (x2s - x1s) * 2)
            rhos_log_space = np.log10(rhos)

            quantized_rhos = np.zeros(rhos.shape, dtype=int)
            for i in range(n_rho):
                quantized_rhos += (rhos_log_space < rho_bins_edges[i])

            for i, r_bin in enumerate(quantized_rhos):
                theta_bin = int(thetas[i] // bin_size) % n_angles
                hist[r_bin - 1, theta_bin] += 1

        hist /= hist.sum()
        feature_vectors[k-ks[0]] = hist.flatten()
    return feature_vectors.flatten()

def run_feature_extraction(image, bw_image, feature):
    assert feature in FEATURES, f"Unknown feature {feature}!"
    if feature in ['lbp', 'hog', 'glcm']:
        image = bw_image
    return sys.modules[__name__].__dict__[feature](image)
