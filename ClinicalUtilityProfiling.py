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

import tracemalloc

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

def slope(fprs, tprs):
    return np.round(np.diff(tprs)/np.diff(fprs), 5)

def derivSlope(slopes, fprs):
    return np.diff(slopes)/np.diff(fprs)[1:]

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
        
        cutoff_rational = find_fpr_tpr_for_slope(fitted_curve_points, desired_slope)

        closest_fpr, closest_tpr = cutoff_rational[0], cutoff_rational[1]

        #find the closest pair of tpr and fpr from the original arrays
        original_tpr, original_fpr, index = find_closest_pair_separate(true_positive_rate, false_positive_rate, closest_tpr, closest_fpr)

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


def max_relative_slopes(fprs, tprs):
    n = len(tprs)
    max_slopes = np.zeros(n)  # To store the maximum slope for each point
    max_indices = np.zeros(n, dtype=int)  # To store the index of the other point for the max slope
    # Calculate the maximum slope from each point to every other point
    for i in np.arange(0, n):
        
        max_slope = -np.inf
        for j in np.arange(i+1, n):
            if((fprs[j] > fprs[i]) | (tprs[j] > tprs[i])):
                slope = (tprs[j] - tprs[i]) / ((fprs[j] - fprs[i]) + 0.00000000001)
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

# 

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

        if denominator == 0:
            print("Warning: Denominator is zero.")
            continue

        curve_point = numerator / denominator
        yield curve_point
    #     curve_points.append(curve_point)
    
    # return np.array(curve_points)


def perpendicular_distance_for_error(points, curve_points):
    """Compute the perpendicular distance from each point to the curve."""
    distances = cdist(points, curve_points, 'euclidean')
    min_distances = np.min(distances, axis=1)
    return min_distances

def error_function(weights, control_points, empirical_points):
    """Compute the error between the rational Bezier curve and the empirical points."""
    curve_points_gen = rational_bezier_curve(control_points, weights, num_points=len(empirical_points) * 1)
    curve_points = np.array(list(curve_points_gen))

    # Process or collect the curve points as needed
    distances = perpendicular_distance_for_error(empirical_points, curve_points)
    normalized_error = np.sum(distances) / len(empirical_points)
    return normalized_error

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
    curve_points_gen = rational_bezier_curve(control_points, weights, num_points=len(empirical_points) * 1)
    curve_points = np.array(list(curve_points_gen))
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
