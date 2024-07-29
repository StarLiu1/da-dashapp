import numpy as np
import sympy as sy
import numpy as np
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import time # timestamp
from scipy.special import comb
from scipy.optimize import minimize_scalar
# import bezier 
from scipy.optimize import minimize
import math

from scipy.spatial.distance import cdist #for perpendicular distances
from scipy.optimize import differential_evolution

def treatAll(x, uFP, uTP):
    """
    Expected value calculation for the option treat all
    
    Args: 
        x (float): probability
        uFP (float): utility of false positive
        uTP (float): utility of true positive
        
    Returns: 
        expected utility (float)
    
    """
    return x * uTP + (1-x) * uFP

def treatNone(x, uFN, uTN):
    """
    Expected value calculation for the option treat none
    
    Args: 
        x (float): probability
        uFN (float): utility of false negative
        uTN (float): utility of true negative
        
    Returns: 
        expected utility (float)
    
    """
    return x * uFN + (1-x) * uTN

def test(x, sensitivity, specificity, uTN, uTP, uFN, uFP, u):
    """
    Expected value calculation for the option test
    
    Args: 
        x (float): probability
        sensitivity (float): sensitivity of the test
        specificity (float): specificity of the test
        uTN (float): utility of true negative = 1
        uTP (float): utility of true positive, cost of false negative (misclassifying a minority class) 
                        is the cost of not getting the benefit of TP
        uFN (float): utility of false negative = 0, the cost of not getting the benefit of TP is uTP
        uFP (float): utility of false positive, harm = 1-uFP = cost of FP
        u: utility of the test itself
        
    Returns: 
        expected utility (float)
    
    """
    
    return x * sensitivity * uTP + x * (1-sensitivity) * uFN + (1-x) * (1-specificity) * uFP + (1-x) * specificity * uTN + u

def pLpUThresholds(sens, spec, uTN, uTP, uFN, uFP, u):
    """
    Identifies the lower and upper thresholds formed by the three utility lines
    
    Args: 
        sens (float): sensitivity of the test
        spec (float): specificity of the test
        uTN (float): utility of true negative
        uTP (float): utility of true positive
        uFN (float): utility of false negative
        uFP (float): utility of false positive
        u: utility of the test itself
        
    Returns: 
        a list of two thresholds (pL and pU): [pL, pU]
    
    """
    #initate a variable called x (prior)
    x = sy.symbols('x')
    
#     print(u)
    
    #solve for upper threshold formed by test and treat all
    pU = sy.solve(treatAll(x, uFP, uTP) - test(x, sens, spec, uTN, uTP, uFN, uFP, u), x)
    
    #solve for lower threshold formed by treat none and test
    pL = sy.solve(treatNone(x, uFN, uTN) - test(x, sens + 0.00000001, spec + 0.00000001, uTN, uTP, uFN, uFP, u), x)
    
    #placeholder values when there are not two thresholds formed
    pU = -1 if (len(pU) == 0) else float(pU[0])
    pU = 1 if (pU > 1) else pU
    pU = 0 if ((pU < 0) & (pU != -999)) else pU
    pL = -1 if (len(pL) == 0) else float(pL[0])
    pL = 1 if (pL > 1) else pL
    pL = 0 if ((pL < 0) & (pL != -999)) else pL

    return [pL, pU]

# # Schumaker spline

def cleanThresholds(array_thresholds):
    #replace inf and -inf
    for i, value in enumerate(array_thresholds):
        if value == math.inf or value > 1:
            array_thresholds[i] = 1
        elif value == -math.inf or value < 0:
            array_thresholds[i] = 0

    return array_thresholds

def compute_spline_coeffs(x, y):
    n = len(x) - 1
    h = np.diff(x)
    b = np.zeros(n + 1)

    # Calculate the first derivatives (slopes) at internal points using finite differences
    delta = np.diff(y) / (h + 0.01)
    
    # Calculate the rhs vector `b` (excluding the first and last point, which are zero by natural spline conditions)
    for i in range(1, n):
        b[i] = 6 * (delta[i] - delta[i-1])

    # Create the tridiagonal matrix `A`
    diagonal = np.ones(n+1) * 2
    lower = np.hstack([0, h[:-1]])
    upper = np.hstack([h[1:], 0])
    
    A = np.diag(lower, -1) + np.diag(diagonal, 0) + np.diag(upper, 1)

    # Boundary conditions for a natural spline
    A[0, 0] = 1
    A[0, 1] = 0
    A[-1, -1] = 1
    A[-1, -2] = 0

    # Solve the system
    m = np.linalg.solve(A, b)

    # Compute coefficients for each polynomial segment
    a = (m[1:] - m[:-1]) / (6 * h + 0.01)
    b = m[:-1] / 2
    c = delta - (h * (2 * m[:-1] + m[1:]) / 6)
    d = y[:-1] - (h**2 * m[:-1] / 6)

    return a, b, c, d, x[:-1]

def eval_spline(x_eval, coeffs, x):
    a, b, c, d, xj = coeffs
    idx = np.searchsorted(x, x_eval, side='right') - 1
    idx = np.clip(idx, 0, len(xj)-1)
    dx = x_eval - xj[idx]
    return a[idx]*dx**3 + b[idx]*dx**2 + c[idx]*dx + d[idx]

def slope(fprs, tprs):
    return np.round(np.diff(tprs)/np.diff(fprs), 5)

def derivSlope(slopes, fprs):
    return np.diff(slopes)/np.diff(fprs)[1:]


def schumaker_fit(false_positive_rate, true_positive_rate, n_int = 100):
    #schumaker spline fit
    coeffs = compute_spline_coeffs(false_positive_rate, true_positive_rate)
    fprs_vals = np.linspace(0, 1, n_int)
    tprs_vals = eval_spline(fprs_vals, coeffs, false_positive_rate)

    #clean the fit so that the slope is always decreasing, so that we dont have recurring slopes
    while all(element <= 0 for element in np.gradient(np.gradient(tprs_vals, fprs_vals))[1:]) is False:
        tprs_vals, fprs_vals = clean_schumaker_spline_slopes(tprs_vals, fprs_vals)

    tprs_vals = [1 if x > 1 else x for x in tprs_vals]
    fpr_finer = np.linspace(0, 1, n_int)
    tpr_finer = np.interp(fpr_finer, fprs_vals, tprs_vals)

    while all(element <= 0 for element in np.gradient(np.gradient(tpr_finer, fpr_finer))[1:]) is False:
        tpr_finer, fpr_finer = clean_schumaker_spline_slopes(tpr_finer, fpr_finer)

    tpr_finer = [1 if x > 1 else x for x in tpr_finer]
    
    return fpr_finer, tpr_finer

def optimalPointOnRoc(false_positive_rate, true_positive_rate, prob_cutoffs, desired_slope, fitted_curve_points, n_int = 100):
    """
    Find the optimal point on the ROC curve that maximizes patient utility, using np.diff
    
    Args: 
        false_positive_rate (array): fprs of the test
        sensitivity (array): tprs of the test
        desired_slope (float): slope of interest
        
    Returns: 
        the point on the ROC curve with the specified curve
    
    """
    # slope of interest
#     desired_slope = maxNNT * (1 - pDisease) / pDisease

    if(desired_slope == 0):
        closest_slope, original_fpr, original_tpr, closest_prob_cutoff = 0, 1.0, 1.0, 0.0
    else:


        # Use these indices to select unique TPRs and corresponding FPRs and thresholds
#         unique_tpr = sensitivity
#         unique_fpr = false_positive_rate
        
#         fpr_finer, tpr_finer = schumaker_fit(false_positive_rate, true_positive_rate)
#         # Calculate the slope between consecutive points on the fitted curve
#         slopes_fine = slope(fpr_finer, tpr_finer)

#         # Find the index of the point with the slope closest to the desired_slope
#         closest_slope_index = np.argmin(np.abs(slopes_fine - desired_slope))
#         closest_slope = slopes_fine[closest_slope_index]
        
        cutoff_rational = find_fpr_tpr_for_slope(fitted_curve_points, desired_slope)

        closest_fpr, closest_tpr = cutoff_rational[0], cutoff_rational[1]
        
        # Corresponding FPR and TPR values for the closest slope
#         closest_fpr = fpr_finer[closest_slope_index]
#         closest_tpr = tpr_finer[closest_slope_index]

        #find the closest pair of tpr and fpr from the original arrays
        original_tpr, original_fpr, index = find_closest_pair_separate(true_positive_rate, false_positive_rate, closest_tpr, closest_fpr)
    #     # Corresponding probability cutoff
    #     closest_prob_cutoff = prob_cutoffs_fine[closest_slope_index]
    #     print(len(prob_cutoffs))
    #     print(index)
        closest_prob_cutoff = prob_cutoffs[index]
        #     print(prob_cutoffs[index -1])
        #     print(prob_cutoffs[index +200])
        
        slopes_fine = slope(false_positive_rate, true_positive_rate)

        # Find the index of the point with the slope closest to the desired_slope
        closest_slope_index = np.argmin(np.abs(slopes_fine - desired_slope))
        closest_slope = slopes_fine[closest_slope_index]

    return closest_slope, original_fpr, original_tpr, closest_prob_cutoff


def slope_of_line(x1, y1, x2, y2):
    return (y2 - y1) / (x2 - x1)


#######05/08/24
# def max_relative_slopes(fprs, tprs):
#     n = len(tprs)
#     max_slopes = np.zeros(n)  # To store the maximum slope for each point
#     max_indices = np.zeros(n, dtype=int)  # To store the index of the other point for the max slope
#     # Calculate the maximum slope from each point to every other point
#     for i in np.arange(0, n):
        
#         max_slope = -np.inf
#         for j in np.arange(i+1, n):
#             if((fprs[j] > fprs[i]) | (tprs[j] > tprs[i])):
#                 slope = (tprs[j] - tprs[i]) / (fprs[j] - fprs[i])
#                 if slope >= max_slope:
#                     max_slope = slope
#                     max_indices[i] = j
#         max_slopes[i] = max_slope
        
#     return [max_slopes, max_indices]
        
# def clean_max_relative_slope_index(indices, len_of_tpr):
#     ordered = [0, indices[0]]
#     for i in np.arange(0, len(indices)):
#         if (np.all(indices[i] >= ordered)):
#             ordered = np.append(ordered, indices[i])
#             i = indices[i]
#     ordered = np.append(ordered, (len_of_tpr-1))
#     return ordered
#find all relative slopes for each point, find the largest one and its index
#######05/08/24

def max_relative_slopes(fprs, tprs):
    n = len(tprs)
    max_slopes = np.zeros(n)  # To store the maximum slope for each point
    max_indices = np.zeros(n, dtype=int)  # To store the index of the other point for the max slope
    # Calculate the maximum slope from each point to every other point
    for i in np.arange(0, n):
        
        max_slope = -np.inf
        for j in np.arange(i+1, n):
            if((fprs[j] > fprs[i]) | (tprs[j] > tprs[i])):
                slope = (tprs[j] - tprs[i]) / (fprs[j] - fprs[i])
                if slope >= max_slope:
                    max_slope = slope
                    max_indices[i] = j
        max_slopes[i] = max_slope
        
    return [max_slopes, max_indices]


def clean_max_relative_slope_index(indices, len_of_tpr):
    # Check if the indices array is empty using the size attribute of the numpy array
    if indices.size == 0:
        return [0, len_of_tpr - 1]

    ordered = [0]  # Start with the first index as 0 (assuming it's the starting index)

    # Start with the first value from indices
    max_val = indices[0]
    ordered.append(max_val)

    for i in indices[1:]:  # Start from the second element
        if i > max_val:
            max_val = i
            ordered.append(i)
        else:
            ordered.append(max_val)

    # Ensure the last index is included
    if ordered[-1] != len_of_tpr - 1:
        ordered.append(len_of_tpr - 1)

    return ordered


def find_lower_vertices(fprs, tprs):
    l_roc_tpr = [0]
    l_roc_fpr = [0]
    #for a pair of tpr and fpr
    idx = 1
    while((idx > 0) & (idx < len(tprs) - 1)):
        #if fpr increases, then add to lower roc
        if((fprs[idx] > fprs[idx - 1]) & (tprs[idx] == tprs[idx - 1])):
            l_roc_tpr = np.append(l_roc_tpr, tprs[idx])
            l_roc_fpr = np.append(l_roc_fpr, fprs[idx])
        idx += 1

    l_roc_tpr = np.append(l_roc_tpr, [1])
    l_roc_fpr = np.append(l_roc_fpr, [1])
    
    return l_roc_fpr, l_roc_tpr

def find_lower_roc_controls(fprs, tprs):
#     slopes = slope(fprs, tprs)
    n = len(tprs)
    
    indices = [0]
    #find where the slopes goes from either pos or inf to negative
    #first find all the slopes relative to the current point
    i = 0
    while i < n:
        slopes = []
        for j in np.arange(i+1, n):
            slope = (tprs[j] - tprs[i]) / (fprs[j] - fprs[i])
            slopes = np.append(slopes, slope)
        slope_of_slopes = np.diff(slopes)
        slope_idx = 0
        next_idx = i
        while slope_idx < (len(slope_of_slopes) - 1):
            if ((slope_of_slopes[slope_idx] <= 0) & (slope_of_slopes[slope_idx+1] > 0)):
                #then this idx is what we want to keep track of. 
                next_idx = slope_idx + i + 2
                indices = np.append(indices, next_idx)
                slope_idx = len(slope_of_slopes)
            else:
                slope_idx += 1
                next_idx = i + 1
                indices = np.append(indices, next_idx)
        if(next_idx > i):
            i = next_idx
        else:
            i += 1
    indices = np.append(indices, len(tprs) - 1)
        
    return indices.astype(int)

def clean_schumaker_spline_slopes(tprs, fprs):
    last_idx = len(tprs) - 4
    slopes = slope(fprs, tprs)
    idx2 = 0
    #if the first 3 forms a bump
    if((slopes[0] < slopes[1]) & (slopes[2] < slopes[1])):
#         tprs[1] = (tprs[0] + tprs[2]) / 2
        tprs[1] = tprs[2]
    #if the slope increases consecutively
    if((slopes[0] < slopes[1]) & (slopes[2] > slopes[1])):
#         while slopes[idx2] > slopes[idx2 - 1]:
#             idx2 += 1
        tprs = np.append(tprs[0], tprs[2: ])
        fprs = np.append(fprs[0], fprs[2: ])
        last_idx = last_idx - 1
    idx = 0
    while idx <= last_idx:
        
        #skip any downward negative slopes
        if(slopes[idx] < 0):
            tprs = np.append(tprs[:(idx + 0)], tprs[(idx + 1): ])
            fprs = np.append(fprs[:(idx + 0)], fprs[(idx + 1): ])
            last_idx = last_idx - 1
            
        #if the slope drops and goes up immediately, drop the middle one
        elif(((slopes[idx] > slopes[idx + 1]) & (slopes[idx + 1] < slopes[idx + 2])) or ((slopes[idx] == slopes[idx + 1]) & (slopes[idx + 1] < slopes[idx + 2]))):
#                 if((idx + 1) != last_idx):
#                 slopes[idx + 1] = (slope + slopes[idx + 2]) / 2
            tprs = np.append(tprs[:(idx + 1)], tprs[(idx + 2): ])
            fprs = np.append(fprs[:(idx + 1)], fprs[(idx + 2): ])
            last_idx = last_idx - 1
            
        #if the slope increases consecutively to begin with
        elif((slopes[idx] < slopes[idx + 1]) & (slopes[idx + 1] < slopes[idx + 2])):
            idx3 = idx
            while slopes[idx3 + 1] > slopes[idx3]:
                idx3 += 1
                
            tprs = np.append(tprs[:(idx + 1)], tprs[(idx3): ])
            fprs = np.append(fprs[:(idx + 1)], fprs[(idx3): ])
            idx = idx3
            last_idx = last_idx - (idx3 - idx)
        # add a stopping mechanism

        idx += 1
#     if(slopes[-1] > slopes[-2]):
#         tprs[-1] = tprs[-2]

    #clean up vertical jumps

    return tprs, fprs


def remove_near_collinear_points_separated(points, tolerance=1e-5):
    if len(points) < 3:
        print('Not enough points to determine collinearity')
        return [], []  # Not enough points to determine collinearity

    # Start with the first two points in the list
    reduced_points_x = [points[0][0], points[1][0]]
    reduced_points_y = [points[0][1], points[1][1]]

    for i in range(2, len(points)):
        p1 = (reduced_points_x[-2], reduced_points_y[-2])
        p2 = (reduced_points_x[-1], reduced_points_y[-1])
        p3 = points[i]

        # Calculate the determinant (area of the triangle formed by p1, p2, p3)
        area = abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))

        if area > tolerance:
            # Points are not collinear within tolerance, add the current point
            reduced_points_x.append(p3[0])
            reduced_points_y.append(p3[1])
        else:
            # Points are near collinear, replace the last point
            reduced_points_x[-1] = p3[0]
            reduced_points_y[-1] = p3[1]

    return np.array(reduced_points_x), np.array(reduced_points_y)

def deduplicate_roc_points(fpr, tpr):
    """
    Deduplicate the exact points from FPRs and TPRs.

    Parameters:
    fpr (array-like): Array of false positive rates.
    tpr (array-like): Array of true positive rates.

    Returns:
    tuple: Deduplicated arrays of FPRs and TPRs.
    """
    # Combine FPR and TPR into a single array for deduplication
    points = np.vstack((fpr, tpr)).T
    
    # Use numpy's unique function to find unique rows
    unique_points = np.unique(points, axis=0)
    
    # Split the unique points back into separate FPR and TPR arrays
    unique_fpr = unique_points[:, 0]
    unique_tpr = unique_points[:, 1]
    
    return unique_fpr, unique_tpr


def repeat_inverse_dist_upperleft(unique_fprs, unique_tprs, base=1, bound_type='lower'):
    e = 1.2  # works well for lower bound
    
    if bound_type == 'lower':
        distances = np.sqrt((unique_fprs - 0) ** 2 + (unique_tprs - 1) ** 2)
#         print("Distances:", distances)
        
        repeated_values = np.round(1 / base**(base * (1 - 1 / distances))).astype(int)
#         print("Repeated Values:", repeated_values)
        
        capped_values = np.minimum(repeated_values, 100)
#         capped_values = repeated_values
        
        # Find the maximum value from the repeated_values array
        max_repeated_value = np.max(capped_values)
        
        # Modify the capped_values for points with fpr=0 or tpr=1, excluding the first and last points
        condition = (unique_fprs == 0) | (unique_tprs == 1)
        condition[0] = False  # Exclude the first point
        condition[-1] = False  # Exclude the last point
        capped_values[condition] = max_repeated_value
    elif bound_type == 'upper':
        capped_values = np.ones_like(unique_fprs).astype(int) * 20
        capped_values[0] = 1  # Exclude the first point
        capped_values[-1] = 1  # Exclude the last point
    
    return capped_values

def filter_points(FPR, TPR):
    filtered_FPR = [FPR[0]]
    filtered_TPR = [TPR[0]]
    
    for i in range(1, len(TPR) - 1):
        if TPR[i] != TPR[i + 1]:
            filtered_FPR.append(FPR[i])
            filtered_TPR.append(TPR[i])
    
    filtered_FPR.append(FPR[-1])
    filtered_TPR.append(TPR[-1])
    
    return np.array(filtered_FPR), np.array(filtered_TPR)

def bernstein_poly(i, n, t):
    """Compute the Bernstein polynomial B_{i,n} at t."""
    return math.comb(n, i) * (t**i) * ((1 - t)**(n - i))

def rational_bezier_curve(control_points, weights, num_points=100):
    """Compute the rational Bezier curve with given control points and weights."""
    n = len(control_points) - 1
    t_values = np.linspace(0, 1, num_points)
    curve_points = []

    for t in t_values:
        numerator = np.zeros(2)
        denominator = 0

        for i in range(n + 1):
            B_i = bernstein_poly(i, n, t)
            numerator += weights[i] * B_i * np.array(control_points[i])
            denominator += weights[i] * B_i

        curve_point = numerator / denominator
        curve_points.append(curve_point)

    return np.array(curve_points)

def perpendicular_distance_for_error(points, curve_points):
    """Compute the perpendicular distance from each point to the curve."""
    distances = cdist(points, curve_points, 'euclidean')
    min_distances = np.min(distances, axis=1)
    return min_distances

def error_function(weights, control_points, empirical_points):
    """Compute the error between the rational Bezier curve and the empirical points."""
    curve_points = rational_bezier_curve(control_points, weights, num_points=len(empirical_points) * 1)
    distances = perpendicular_distance_for_error(empirical_points, curve_points)
    normalized_error = np.sum(distances) / len(empirical_points)
    return normalized_error

def repeat_points_by_proportion(fprs, tprs):
    """
    Repeat each point in the FPR and TPR arrays according to the proportion of the total number of points,
    excluding endpoints and multiplying the proportion by 100.
    
    Parameters:
    fprs (np.ndarray): Array of false positive rates.
    tprs (np.ndarray): Array of true positive rates.
    
    Returns:
    np.ndarray, np.ndarray: Repeated FPR and TPR arrays.
    """
    total_points = len(fprs)

    # Calculate the proportion of each point, excluding endpoints
    proportions = np.ones(total_points - 2) / (total_points - 2)

    # Repeat each point according to its proportion times 100
    repeat_counts = (proportions * 100).astype(int)
    fprs_repeated = []
    tprs_repeated = []

    # Add the first point
    fprs_repeated.append(fprs[0])
    tprs_repeated.append(tprs[0])

    for i in range(1, total_points - 1):  # Exclude the first and last points
        fprs_repeated.extend([fprs[i]] * repeat_counts[i-1])
        tprs_repeated.extend([tprs[i]] * repeat_counts[i-1])

    # Add the last point
    fprs_repeated.append(fprs[-1])
    tprs_repeated.append(tprs[-1])

    return np.array(fprs_repeated), np.array(tprs_repeated)

def compute_slope(points):
    """Compute the slope between consecutive points."""
    slopes = []
    for i in range(1, len(points)):
        dy = points[i][1] - points[i-1][1]
        dx = points[i][0] - points[i-1][0]
        slope = dy / dx if dx != 0 else np.inf
        slopes.append(slope)
    return slopes

def clean_roc_curve(points):
    """Remove points iteratively until the curve is always decreasing."""
    points = np.array(points)
    cleaned_points = points.tolist()
    
    while True:
        slopes = compute_slope(cleaned_points)
        increase_found = False
        
        for i in range(1, len(slopes)):
            if slopes[i] > slopes[i-1]:
                increase_found = True
                del cleaned_points[i]  # Remove the point causing the increase
                break
        
        if not increase_found:
            break
    
    return np.array(cleaned_points)


def compute_slope(points):
    """Compute the slope between consecutive points."""
    slopes = []
    for i in range(1, len(points)):
        dy = points[i][1] - points[i-1][1]
        dx = points[i][0] - points[i-1][0]
        slope = dy / dx if dx != 0 else np.inf
        slopes.append(slope)
    return slopes

def clean_roc_curve(points):
    """Remove points iteratively until the curve is always decreasing."""
    points = np.array(points)
    cleaned_points = points.tolist()
    removed_indices = []

    while True:
        slopes = compute_slope(cleaned_points)
        increase_found = False

        for i in range(1, len(slopes)):
            if slopes[i] > slopes[i-1]:
                increase_found = True
                removed_indices.append(i)
                del cleaned_points[i]  # Remove the point causing the increase
                break

        if not increase_found:
            break

    return np.array(cleaned_points), removed_indices

def perpendicular_distance(point, line_start, line_end):
    """
    Calculate the perpendicular distance from a point to a line segment.

    Parameters:
    point (tuple): The point (x, y).
    line_start (tuple): The starting point of the line segment (x1, y1).
    line_end (tuple): The ending point of the line segment (x2, y2).

    Returns:
    float: The perpendicular distance from the point to the line segment.
    """
    x, y = point
    x1, y1 = line_start
    x2, y2 = line_end
    
    # Calculate the distance from the point to the line segment
    num = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
    den = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
    distance = num / den
    
    return distance

def min_set_w_pt_above(original_set, kept_set, kept_indices, removed_indices):
    new_set_point_idx = [0]
    for idx_a in range(1, len(kept_indices) - 1):
        # If the indices are not consecutive, then it means there are points in the middle removed from the list
        if kept_indices[idx_a] != kept_indices[idx_a + 1] - 1:
            # Find the farthest point in this range
            farthest_dist = 0.0
            farthest_dist_idx = 0
            for idx_removed in range(kept_indices[idx_a] + 1, kept_indices[idx_a + 1]):
                perp_dist = perpendicular_distance(original_set[idx_removed], kept_set[idx_a], kept_set[idx_a + 1])
                if perp_dist > farthest_dist:
                    farthest_dist = perp_dist
                    farthest_dist_idx = idx_removed
            # Add the farthest point to the new set of indices
            if(farthest_dist_idx != 0):
                new_set_point_idx = np.append(new_set_point_idx, farthest_dist_idx)
        else:
            new_set_point_idx = np.append(new_set_point_idx, kept_indices[idx_a])
    new_set_point_idx = np.append(new_set_point_idx, kept_indices[-1])
    return new_set_point_idx.astype(int)

def clean_and_find_points(fprs, tprs):
    roc_points = list(zip(fprs, tprs))
    roc_points = np.array(roc_points)
    original_indices = np.arange(len(roc_points))

    # Clean the ROC curve
    cleaned_roc_points, removed_indices_temp = clean_roc_curve(roc_points)

    # Get indices of cleaned and removed points in relation to the original order
    cleaned_indices = [i for i in original_indices if roc_points[i].tolist() in cleaned_roc_points.tolist()]
    removed_indices = [i for i in original_indices if i not in cleaned_indices]

    # Add the first and last points to A if not already present
    A_indices = cleaned_indices
    if 0 not in A_indices:
        A_indices.insert(0, 0)
    if len(roc_points) - 1 not in A_indices:
        A_indices.append(len(roc_points) - 1)

    # Ensure A_indices is sorted
    A_indices = sorted(A_indices)

    # Ensure the first and last points are present
    if A_indices[0] != 0:
        A_indices.insert(0, 0)
    if A_indices[-1] != len(roc_points) - 1:
        A_indices.append(len(roc_points) - 1)

    # Create the final control points using the updated A_indices
    final_control_points = roc_points[A_indices]
    removed_points = roc_points[removed_indices]
    final_set_idx = min_set_w_pt_above(roc_points, roc_points[A_indices], A_indices, removed_indices)
    final_cleaned_points = roc_points[final_set_idx]

    return final_cleaned_points, final_set_idx

def error_function_lower_bezier(weights, fpr, tpr):
    """Compute the error between the rational Bezier curve and the empirical points."""
    empirical_curve = list(zip(fpr, tpr))
    
    pt_distances = repeat_inverse_dist_upperleft(fpr, tpr, base=weights, bound_type='lower')
    #"weigh" the points by creating repeats
    repeated_fprs = np.repeat(fpr, pt_distances)
    repeated_tprs = np.repeat(tpr, pt_distances)
    
    fpr_tpr = [(x, y) for x, y in zip(repeated_fprs, repeated_tprs)]
    control_points = fpr_tpr
    fitted_curve = np.array([Bezier(control_points, t) for t in np.linspace(0, 1, num=100)])

    distances = perpendicular_distance_for_error(empirical_curve, fitted_curve)
    normalized_error = np.sum(distances) / len(empirical_curve)
    return normalized_error

def error_function_convex_hull_bezier(weights, fpr, tpr):
    """Compute the error between the rational Bezier curve and the empirical points."""
    empirical_points = list(zip(fpr, tpr))
    
    #upper bound
    outer_idx = max_relative_slopes(fpr, tpr)[1]
    outer_idx = clean_max_relative_slope_index(outer_idx, len(tpr))
    u_roc_fpr_fitted, u_roc_tpr_fitted = fpr[outer_idx], tpr[outer_idx]
    u_roc_fpr_fitted, u_roc_tpr_fitted = deduplicate_roc_points(u_roc_fpr_fitted, u_roc_tpr_fitted)

    control_points = list(zip(u_roc_fpr_fitted, u_roc_tpr_fitted))
    
    """Compute the error between the rational Bezier curve and the empirical points."""
    curve_points = rational_bezier_curve(control_points, weights, num_points=len(empirical_points) * 1)
    distances = perpendicular_distance_for_error(empirical_points, curve_points)
    normalized_error = np.sum(distances) / len(empirical_points)
    return normalized_error


# Early termination callback
class EarlyTermination:
    def __init__(self):
        self.best_non_nan_convergence = float('inf')
        self.best_non_nan_params = None
        self.consecutive_increases = 0
        self.last_convergence = None

    def __call__(self, xk, convergence):
        print(f"Current convergence: {convergence}")
        
        # Check if convergence is NaN
        if np.isnan(convergence):
            print("Convergence is NaN, reverting to the best non-NaN convergence value and stopping.")
            if self.best_non_nan_params is not None:
                # Revert to the best known good state
                xk[:] = self.best_non_nan_params
                print(f"Reverted to best non-NaN params: {self.best_non_nan_params}")
            return True  # Terminate the optimization
        
        # Update the best non-NaN convergence and parameters
        if convergence < self.best_non_nan_convergence:
            self.best_non_nan_convergence = convergence
            self.best_non_nan_params = xk.copy()
            print(f"New best non-NaN convergence: {self.best_non_nan_convergence}, params: {self.best_non_nan_params}")
        
        # Check for consecutive increases in convergence
        if self.last_convergence is not None and convergence > self.last_convergence:
            self.consecutive_increases += 1
        else:
            self.consecutive_increases = 0
        
        self.last_convergence = convergence
        
        if self.consecutive_increases >= 3:
            print("Terminating early due to consecutive increases in convergence.")
            return True  # Terminate the optimization

        return False  # Continue the optimization



def weighted_repeated_roc(fprs, tprs):
    # Calculate distances between consecutive points
    distances = np.sqrt(np.diff(fprs)**2 + np.diff(tprs)**2)

    # Total length of the curve (including the endpoints)
    total_length = np.sum(distances)

    # Proportion of each distance to the total length
    proportions = distances / total_length

    # Repeat each point according to its proportion times 100
    fprs_repeated = []
    tprs_repeated = []

    for i in range(1, len(fprs) - 1):  # Exclude the first and last points
        repeat_count = int(proportions[i-1] * 100)
        if((repeat_count > 2) or (fprs[i] > 0.05) or (tprs[i] > 0.05)):
            fprs_repeated.extend([fprs[i]] * repeat_count)
            tprs_repeated.extend([tprs[i]] * repeat_count)

    # Add the first and last points
    fprs_repeated = [fprs[0]] + fprs_repeated + [fprs[-1]]
    tprs_repeated = [tprs[0]] + tprs_repeated + [tprs[-1]]

    return np.array(fprs_repeated), np.array(tprs_repeated)

def Bezier(control_points, t):
    """Compute a point on a Bézier curve defined by control points at parameter t."""
    n = len(control_points) - 1
    point = np.zeros(2)
    for i in range(n + 1):
        # Ensure the control point is an array for element-wise operations
        binom = comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
        point += binom * np.array(control_points[i])
    return point

def bezier_derivative(control_points, t):
    """Compute the first derivative of a Bézier curve at parameter t."""
    n = len(control_points) - 1
    # Converting to numpy array to ensure element-wise operations
    derivative_control_points = np.array([n * (np.array(control_points[i + 1], dtype=np.float64) - np.array(control_points[i], dtype=np.float64)) for i in range(n)])
    return Bezier(derivative_control_points, t)

def slope_and_point_of_bezier_curve_given_fpr(control_points, desired_fpr):
    """Calculate the slope (dy/dx) and the coordinate of the Bézier curve at the FPR closest to the desired FPR."""
    # Define the objective function to find t
    def error_in_fpr(t):
        point = Bezier(control_points, t)
        fpr = point[0]  # The x-coordinate of the Bézier curve
        return (fpr - desired_fpr) ** 2
    
    # Use a scalar minimization to find the best t
    result = minimize_scalar(error_in_fpr, bounds=(0, 1), method='bounded')
    if not result.success:
        raise ValueError("Optimization did not converge")

    # Compute the point and slope at the optimized t
    t = result.x
    point = Bezier(control_points, t)
    derivative = bezier_derivative(control_points, t)
    dx, dy = derivative
    if dx == 0:
        slope = float('inf')  # Could also be considered 'undefined'
    else:
        slope = dy / dx

    return point, slope, t

def find_fpr_tpr_for_slope(control_points, desired_slope):
    """Find the FPR and TPR on the Bézier curve for a given slope."""
    # Define the objective function to match the desired slope
    def slope_error(t):
        derivative = bezier_derivative(control_points, t)
        dx, dy = derivative
        current_slope = dy / dx if dx != 0 else float('inf')  # Avoid division by zero
        return (current_slope - desired_slope) ** 2
    
    # Use scalar minimization to find the t that gives the desired slope
    result = minimize_scalar(slope_error, bounds=(0, 1), method='bounded')
    if not result.success:
        raise ValueError("Optimization did not converge")

    t_optimal = result.x
    point = Bezier(control_points, t_optimal)
    return point[0], point[1], t_optimal  # Return FPR, TPR, and t

def find_closest_pair_separate(tprs, fprs, desired_tpr, desired_fpr):
    # Stack the TPRs and FPRs into a 2D numpy array for vectorized operations
    pairs_array = np.column_stack((tprs, fprs))
    
    # Compute the Euclidean distance from each pair to the desired pair
    distances = np.sqrt((pairs_array[:, 0] - desired_tpr) ** 2 + (pairs_array[:, 1] - desired_fpr) ** 2)
    
    # Find the index of the smallest distance
    closest_index = np.argmin(distances)
    
    # Return the closest pair and its index
    return tprs[closest_index], fprs[closest_index], closest_index

# def schumaker_fit(false_positive_rate, true_positive_rate, n_int = 100):
#     #schumaker spline fit
#     coeffs = compute_spline_coeffs(false_positive_rate, true_positive_rate)
#     fprs_vals = np.linspace(0, 1, n_int)
#     tprs_vals = eval_spline(fprs_vals, coeffs, false_positive_rate)
    
#     roc_slopes = slope(fprs_vals, tprs_vals)
#     secondDerivative = derivSlope(roc_slopes, fprs_vals)

# #     clean the fit so that the slope is always decreasing, so that we dont have recurring slopes
#     while all(element <= 0 for element in derivSlope(slope(fprs_vals, tprs_vals), fprs_vals)[0:]) is False:
# #         print((slope(fprs_vals, tprs_vals), fprs_vals))
# #         print(derivSlope(slope(fprs_vals, tprs_vals), fprs_vals))
#         tprs_vals, fprs_vals = clean_schumaker_spline_slopes(tprs_vals, fprs_vals)

#     tprs_vals = [1 if x > 1 else x for x in tprs_vals]
#     return fprs_vals, tprs_vals

def minicup(pDisease, fprs, tprs, prob_cutoffs, fitted_curve_points, HoverB, utilities):
    """
    Clinical Utility Profiling subcomponent
    - for one pair of uTP and H/B
        - calculate maxEU, pL, and pU
    
    Args:
        pDisease (float): probability of disease
        fprs (array): fprs of the test
        tprs (array): tprs of the test
        costRatio (float): 1/costRatio = maxNNT
            maxNNT: H/B tradeoff; ratio of misclassification costs Cfp/Cfn; 1/costratio = H/B; cmc_major/cmc_minor
        utilities (array): array carrying uTP and dU (disutility of testing itself)
        
    Returns:
        an array of maxEU, pL, pU, uTP, maxNNT (H/B)
        
    
    """
    uTP, dU = utilities #uTP and disutility of testing itself
    uTN = 1.0
    uFN = 0
#     HoverB = (1 / costRatio)
    uFP = uTN - (uTP - uFN) * HoverB

    # slope of interest
    desired_slope = HoverB * (1 - pDisease) / pDisease
#     print(f'Desired slope is: {np.round(desired_slope, 3)}, given pDisease: {np.round(pDisease, 3)} and maxNNT: {np.round(HoverB, 3)}')
    
    #find the optimal point on the ROC and its associated specs
    closest_slope, closest_fpr, closest_tpr, closest_prob_cutoff = optimalPointOnRoc(fprs, tprs, prob_cutoffs, desired_slope, fitted_curve_points)
#     print(f'Closest slope is: {np.round(closest_slope, 3)}, corresponding to fpr: {np.round(closest_fpr, 3)} and tpr: {np.round(closest_tpr, 3)}')
    
    #find the expected utility for the given probability of disease in the target population, the sens and spec associated with the optimal point on the roc
    maxEU = test(pDisease, closest_tpr+0.0000001, 1-closest_fpr-0.0000001, uTN, uTP, uFN, uFP, dU)
    
    #pL and pU
    pL, pU = pLpUThresholds(closest_tpr+0.0000001, 1-closest_fpr-0.0000001, uTN, uTP, uFN, uFP, dU)
#     print(pL)
#     print(pU)
#     print(closest_tpr)
#     print(closest_fpr)

    return maxEU, pL, pU, closest_prob_cutoff, uTP, HoverB

def cup(pDisease, fprs, tprs, cutoffs, HoverBs, uTPs, dU):
    """
    Clinical Utility Profiling main
    - for each pair of uTP and H/B
        - calculate maxEU, pL, and pU
    - call minicup()
    
    Args:
        pDisease (float): probability of disease
        fprs (array): fprs of the test
        tprs (array): tprs of the test
        costRatio (float): 1/costRatio = maxNNT
            maxNNT: H/B tradeoff; ratio of misclassification costs Cfp/Cfn; 1/costratio = H/B; cmc_major/cmc_minor
        utilities (array): array carrying uTP and dU (disutility of testing itself)
        
    Returns:
        an array of maxEU, pL, pU, uTP, maxNNT (H/B)
        
    
    """
    
    #placeholders
    maxEUs = []
    pLs = []
    pUs = []
    prob_cutoffs = []
#     uTPs = []
#     maxNNTs = []
    
    start = time.time()
    
    outer_idx = max_relative_slopes(fprs, tprs)[1]
    outer_idx = clean_max_relative_slope_index(outer_idx, len(tprs))
    u_roc_fpr_fitted, u_roc_tpr_fitted = fprs[outer_idx], tprs[outer_idx]
    u_roc_fpr_fitted, u_roc_tpr_fitted = deduplicate_roc_points(u_roc_fpr_fitted, u_roc_tpr_fitted)

    #general rational bezier fit
    control_points = list(zip(u_roc_fpr_fitted, u_roc_tpr_fitted))
    empirical_points = list(zip(fprs, tprs))
    initial_weights = [1] * len(control_points)
    bounds = [(0, 50) for _ in control_points]
    # Optimize weights to fit the empirical points
    result = minimize(error_function, initial_weights, args=(control_points, empirical_points), method='SLSQP', bounds = bounds)
    optimal_weights = result.x

    # Compute the rational Bezier curve with optimal weights
    curve_points = rational_bezier_curve(control_points, optimal_weights)
        
        
    for uTP in uTPs:
        print(uTP)
        for HoverB in HoverBs: 
#             print(f'Ratio between misclassifying a fn and a fp is: {costRatio}')
            utilities = [uTP, dU]
            maxEU, pL, pU, closest_prob_cutoff, uTP, maxNNT = minicup(pDisease, fprs, tprs, cutoffs, curve_points, HoverB, utilities)
            if(maxEU < 0):
                maxEU = 0
            maxEUs = np.append(maxEUs, np.round(maxEU, 5))
            pLs = np.append(pLs, np.round(pL, 5))
            pUs = np.append(pUs, np.round(pU, 5))
            prob_cutoffs = np.append(prob_cutoffs, np.round(closest_prob_cutoff, 5))
            
            
    end = time.time()
    print("Took: ", end - start, " seconds.")
    return maxEUs, pLs, pUs, prob_cutoffs