# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, fabs
import sympy as sy
import pandas as pd
import concurrent.futures

# Add these to the top of your file
from libc.math cimport INFINITY
from scipy.spatial.distance import cdist
from scipy.optimize import minimize_scalar 
from scipy.special import comb
from cpython.list cimport PyList_Append


# Define numpy types
ctypedef np.float64_t DTYPE_t
ctypedef np.int32_t ITYPE_t


cpdef tuple deduplicate_roc_points(np.ndarray[DTYPE_t, ndim=1] fpr, 
                                  np.ndarray[DTYPE_t, ndim=1] tpr):
    """
    Deduplicate the exact points from FPRs and TPRs.
    
    Args:
        fpr (np.ndarray): Array of false positive rates
        tpr (np.ndarray): Array of true positive rates
        
    Returns:
        tuple: Deduplicated arrays of (unique_fpr, unique_tpr)
    """
    # Combine FPR and TPR into a single array for deduplication
    cdef np.ndarray[DTYPE_t, ndim=2] points = np.vstack((fpr, tpr)).T
    
    # Use numpy's unique function to find unique rows
    cdef np.ndarray[DTYPE_t, ndim=2] unique_points = np.unique(points, axis=0)
    
    # Split the unique points back into separate FPR and TPR arrays
    cdef np.ndarray[DTYPE_t, ndim=1] unique_fpr = unique_points[:, 0]
    cdef np.ndarray[DTYPE_t, ndim=1] unique_tpr = unique_points[:, 1]
    
    return unique_fpr, unique_tpr

cpdef np.ndarray[ITYPE_t, ndim=1] clean_max_relative_slope_index(object indices, int n):
    """
    Cleans and sorts indices for maximum relative slopes.
    Accepts any array-like object containing indices and converts to proper dtype.
    
    Args:
        indices: Array-like object containing indices
        n (int): Length of the original data array
        
    Returns:
        np.ndarray: Cleaned and sorted array of indices
    """
    cdef input_array
    cdef clean_indices
    
    # Convert input to numpy array, explicitly handling different input types
    if isinstance(indices, np.ndarray):
        # If it's already a numpy array, convert to int32
        input_array = indices.astype(np.int64)
    elif isinstance(indices, list):
        # Convert list to int64 numpy array
        input_array = np.array(indices, dtype=np.int64)
    else:
        # Try converting to numpy array, forcing int64
        input_array = np.array(list(indices), dtype=np.int64)
    
    # Remove duplicate indices and convert to int32
    clean_indices = np.unique(input_array).astype(np.int32)
    
    # Make sure first and last points are included
    if clean_indices.size > 0:
        if clean_indices[0] != 0:
            clean_indices = np.insert(clean_indices, 0, 0)
        if clean_indices[-1] != n-1:
            clean_indices = np.append(clean_indices, n-1)
    else:
        # If no indices provided, include at least first and last points
        clean_indices = np.array([0, n-1], dtype=np.int32)
    
    return clean_indices


cpdef max_relative_slopes(np.ndarray[DTYPE_t, ndim=1] fprs, np.ndarray[DTYPE_t, ndim=1] tprs):
    """
    Calculate the maximum slope from each point to any other point with a higher FPR or TPR.
    
    Parameters:
    -----------
    fprs : np.ndarray
        Array of false positive rates
    tprs : np.ndarray
        Array of true positive rates
        
    Returns:
    --------
    list
        [max_slopes, max_indices] where max_slopes contains the maximum slope for each point
        and max_indices contains the index of the other point that gives the maximum slope
    """
    cdef int n = len(tprs)
    cdef max_slopes = np.zeros(n, dtype=np.float64)
    cdef max_indices = np.zeros(n, dtype=np.int_)
    
    cdef int i, j
    cdef double max_slope, slope
    cdef double epsilon = 1e-11  # Small constant to avoid division by zero
    
    # Calculate the maximum slope from each point to every other point
    for i in range(n):
        max_slope = -np.inf
        for j in range(i+1, n):
            if (fprs[j] > fprs[i]) or (tprs[j] > tprs[i]):
                slope = (tprs[j] - tprs[i]) / ((fprs[j] - fprs[i]) + epsilon)
                if slope >= max_slope:
                    max_slope = slope
                    max_indices[i] = j
        max_slopes[i] = max_slope
        
    return [max_slopes, max_indices]

cpdef double bernstein_poly(int i, int n, double t):
    """
    Compute the Bernstein polynomial B_{i,n} at t.
    
    Args:
        i (int): Index
        n (int): Degree
        t (double): Parameter value
        
    Returns:
        double: Value of the Bernstein polynomial
    """
    if i < 0 or i > n:
        return 0.0
        
    cdef double result
    result = comb(n, i, exact=True) * pow(t, i) * pow(1.0 - t, n - i)
    return result

def rational_bezier_curve(list control_points, weights, int num_points=100):
    """
    Compute the rational Bezier curve with given control points and weights.
    
    Args:
        control_points (list): List of control points, each point is a 2D array or tuple
        weights (np.ndarray): Array of weights for the control points
        num_points (int): Number of points to generate on the curve
        
    Returns:
        Generator of points on the curve
    """
    # Local function implementation
    cdef int n = len(control_points) - 1
    
    # Local buffer type declarations within the function
    np_control_points = np.array(control_points, dtype=np.float64)
    t_values = np.linspace(0, 1, num_points)
    
    cdef double t, B_i, denominator
    cdef int i
    
    for t in t_values:
        # Initialize local variables in the function 
        # Use local buffer type declarations
        numerator = np.zeros(2, dtype=np.float64)
        denominator = 0.0
        
        for i in range(n + 1):
            B_i = bernstein_poly(i, n, t)
            numerator += weights[i] * B_i * np_control_points[i]
            denominator += weights[i] * B_i

        if denominator == 0:
            print("Warning: Denominator is zero.")
            continue

        # Local buffer type declaration
        curve_point = numerator / denominator
        yield curve_point


cpdef np.ndarray[DTYPE_t, ndim=1] perpendicular_distance_for_error(
    np.ndarray[DTYPE_t, ndim=2] points, 
    np.ndarray[DTYPE_t, ndim=2] curve_points):
    """
    Compute the perpendicular distance from each point to the curve.
    
    Args:
        points (np.ndarray): Array of points
        curve_points (np.ndarray): Array of points on the curve
        
    Returns:
        np.ndarray: Array of minimum distances
    """
    cdef np.ndarray[DTYPE_t, ndim=2] distances = cdist(points, curve_points, 'euclidean')
    cdef np.ndarray[DTYPE_t, ndim=1] min_distances = np.min(distances, axis=1)
    return min_distances


cpdef double error_function(
    np.ndarray[DTYPE_t, ndim=1] weights, 
    list control_points, 
    list empirical_points):
    """
    Compute the error between the rational Bezier curve and the empirical points.
    
    Args:
        weights (np.ndarray): Array of weights for the control points
        control_points (list): List of control points
        empirical_points (np.ndarray): Array of empirical points to compare against
        
    Returns:
        double: Normalized error value
    """
    cdef int num_curve_points = len(empirical_points) * 1
    cdef object curve_points_gen = rational_bezier_curve(control_points, weights, num_points=num_curve_points)
    cdef np.ndarray[DTYPE_t, ndim=2] curve_points = np.array(list(curve_points_gen), dtype=np.float64)
    
    # Process or collect the curve points as needed
    cdef np.ndarray[DTYPE_t, ndim=1] distances = perpendicular_distance_for_error(np.array(empirical_points), curve_points)
    cdef double normalized_error = np.sum(distances) / len(empirical_points)
    return normalized_error


cpdef list compute_slope(list points):
    """
    Compute the slope between consecutive points.
    
    Args:
        points (list): List of (x, y) coordinate pairs
        
    Returns:
        list: List of slopes between consecutive points
    """
    cdef list slopes = []
    cdef int i
    cdef double dy, dx, slope
    cdef int n = len(points)
    
    for i in range(1, n):
        dy = points[i][1] - points[i-1][1]
        dx = points[i][0] - points[i-1][0]
        slope = dy / dx if dx != 0 else INFINITY
        slopes.append(slope)
    
    return slopes

cpdef tuple clean_roc_curve(list points):
    """
    Remove points iteratively until the curve is always decreasing.
    
    Args:
        points (list): List of (x, y) coordinate pairs
        
    Returns:
        tuple: (cleaned_points, removed_indices)
    """
    cdef np.ndarray[DTYPE_t, ndim=2] np_points = np.array(points, dtype=np.float64)
    cdef list cleaned_points = np_points.tolist()
    cdef list removed_indices = []
    cdef list slopes
    cdef bint increase_found
    cdef int i
    
    while True:
        slopes = compute_slope(cleaned_points)
        increase_found = False

        for i in range(1, len(slopes)):
            if slopes[i] > slopes[i-1]:
                increase_found = True
                removed_indices.append(i)
                # Remove the point causing the increase
                del cleaned_points[i]
                break

        if not increase_found:
            break

    return np.array(cleaned_points, dtype=np.float64), removed_indices

cpdef double perpendicular_distance(tuple point, tuple line_start, tuple line_end):
    """
    Calculate the perpendicular distance from a point to a line segment.
    
    Args:
        point (tuple): The point (x, y)
        line_start (tuple): The starting point of the line segment (x1, y1)
        line_end (tuple): The ending point of the line segment (x2, y2)
        
    Returns:
        double: The perpendicular distance from the point to the line segment
    """
    cdef double x = point[0]
    cdef double y = point[1]
    cdef double x1 = line_start[0]
    cdef double y1 = line_start[1]
    cdef double x2 = line_end[0]
    cdef double y2 = line_end[1]
    
    # Calculate the distance from the point to the line segment
    cdef double num = fabs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
    cdef double den = sqrt((y2 - y1)**2 + (x2 - x1)**2)
    
    if den == 0:  # Handle case where line_start and line_end are the same point
        return sqrt((x - x1)**2 + (y - y1)**2)
    
    cdef double distance = num / den
    return distance

cpdef double error_function_convex_hull_bezier(np.ndarray[DTYPE_t, ndim=1] weights, 
                                            np.ndarray[DTYPE_t, ndim=1] fpr, 
                                            np.ndarray[DTYPE_t, ndim=1] tpr):
    """
    Compute the error between the rational Bezier curve and the empirical points.
    
    Args:
        weights (np.ndarray): Array of weights for the control points
        fpr (np.ndarray): Array of false positive rates
        tpr (np.ndarray): Array of true positive rates
        
    Returns:
        double: Normalized error value
    """
    cdef list empirical_points = list(zip(fpr, tpr))
    cdef np.ndarray[ITYPE_t, ndim=1] outer_idx
    cdef np.ndarray[DTYPE_t, ndim=1] u_roc_fpr_fitted, u_roc_tpr_fitted
    cdef list control_points
    cdef object curve_points_gen
    cdef np.ndarray[DTYPE_t, ndim=2] curve_points
    cdef np.ndarray[DTYPE_t, ndim=1] distances
    cdef double normalized_error
    
    # Upper bound
    outer_idx = max_relative_slopes(fpr, tpr)[1]
    outer_idx = clean_max_relative_slope_index(outer_idx, len(tpr))
    u_roc_fpr_fitted, u_roc_tpr_fitted = fpr[outer_idx], tpr[outer_idx]
    u_roc_fpr_fitted, u_roc_tpr_fitted = deduplicate_roc_points(u_roc_fpr_fitted, u_roc_tpr_fitted)

    control_points = list(zip(u_roc_fpr_fitted, u_roc_tpr_fitted))
    
    # Compute the error
    curve_points_gen = rational_bezier_curve(control_points, weights, num_points=len(empirical_points))
    curve_points = np.array(list(curve_points_gen), dtype=np.float64)
    
    if len(curve_points) == 0:
        return INFINITY  # Return infinity if no curve points are generated
    
    # Convert empirical_points to numpy array for the distance calculation
    cdef np.ndarray[DTYPE_t, ndim=2] np_empirical_points = np.array(empirical_points, dtype=np.float64)
    
    distances = perpendicular_distance_for_error(np_empirical_points, curve_points)
    normalized_error = np.sum(distances) / len(empirical_points)
    return normalized_error

# Define a Cython class for early termination
cdef class EarlyTermination:
    cdef public double best_non_nan_convergence
    cdef public object best_non_nan_params
    cdef public int consecutive_increases
    cdef public object last_convergence
    
    def __init__(self):
        self.best_non_nan_convergence = INFINITY
        self.best_non_nan_params = None
        self.consecutive_increases = 0
        self.last_convergence = None

    def __call__(self, np.ndarray xk, double convergence):
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



cpdef np.ndarray[DTYPE_t, ndim=1] Bezier(list control_points, double t):
    """
    Compute a point on a Bézier curve defined by control points at parameter t.
    
    Args:
        control_points (list): List of control points, each point is a 2D array or tuple
        t (double): Parameter value between 0 and 1
        
    Returns:
        np.ndarray: Point on the Bezier curve at parameter t
    """
    cdef int n = len(control_points) - 1
    cdef np.ndarray[DTYPE_t, ndim=1] point = np.zeros(2, dtype=np.float64)
    cdef int i
    cdef double binom
    cdef np.ndarray[DTYPE_t, ndim=1] cp_array
    
    # Convert control points to numpy array for faster operations
    np_control_points = np.array(control_points, dtype=np.float64)
    
    for i in range(n + 1):
        # Calculate binomial coefficient * t^i * (1-t)^(n-i)
        binom = comb(n, i) * pow(t, i) * pow(1.0 - t, n - i)
        point += binom * np_control_points[i]
    
    return point

cpdef np.ndarray[DTYPE_t, ndim=1] bezier_derivative(list control_points, double t):
    """
    Compute the first derivative of a Bézier curve at parameter t.
    
    Args:
        control_points (list): List of control points, each point is a 2D array or tuple
        t (double): Parameter value between 0 and 1
        
    Returns:
        np.ndarray: Derivative vector at parameter t
    """
    cdef int n = len(control_points) - 1
    cdef int i
    
    # Convert control points to numpy array for faster operations
    np_control_points = np.array(control_points, dtype=np.float64)
    
    # Generate the derivative control points
    derivative_control_points = np.zeros((n, 2), dtype=np.float64)
    
    for i in range(n):
        derivative_control_points[i] = n * (np_control_points[i + 1] - np_control_points[i])
    
    # Convert back to list for Bezier function
    cdef list derivative_points_list = derivative_control_points.tolist()
    
    return Bezier(derivative_points_list, t)


# Global variables to store context for optimization
cdef list _global_control_points
cdef double _global_desired_fpr

cpdef double _error_in_fpr_global(double t_val):
    """
    Global function for error calculation that can be used with scipy minimize_scalar
    
    Args:
        t_val (double): Parameter value for Bezier curve
        
    Returns:
        double: Squared difference between current FPR and desired FPR
    """
    cdef np.ndarray[DTYPE_t, ndim=1] point = Bezier(_global_control_points, t_val)
    cdef double fpr = point[0]  # The x-coordinate of the Bézier curve
    return (fpr - _global_desired_fpr) ** 2

cpdef tuple slope_and_point_of_bezier_curve_given_fpr(list control_points, double desired_fpr):
    """
    Calculate the slope (dy/dx) and the coordinate of the Bézier curve at the FPR closest to the desired FPR.
    
    Args:
        control_points (list): List of control points defining the Bezier curve
        desired_fpr (double): Desired false positive rate
        
    Returns:
        tuple: (point, slope, t) where point is the coordinates, slope is dy/dx, and t is the parameter
    """
    global _global_control_points, _global_desired_fpr
    
    # Set global variables for optimization
    _global_control_points = control_points
    _global_desired_fpr = desired_fpr
    
    # Use a scalar minimization to find the best t
    result = minimize_scalar(_error_in_fpr_global, bounds=(0, 1), method='bounded')
    
    if not result.success:
        raise ValueError("Optimization did not converge")

    # Compute the point and slope at the optimized t
    cdef double t = result.x
    cdef np.ndarray[DTYPE_t, ndim=1] point = Bezier(control_points, t)
    cdef np.ndarray[DTYPE_t, ndim=1] derivative = bezier_derivative(control_points, t)
    cdef double dx = derivative[0]
    cdef double dy = derivative[1]
    cdef double slope
    
    if fabs(dx) < 1e-10:  # Small threshold to avoid floating point issues
        slope = float('inf')  # Could also be considered 'undefined'
    else:
        slope = dy / dx

    return point, slope, t

# Global variables to store context for optimization
cdef double _global_desired_slope

cpdef double _slope_error_global(double t_val):
    """
    Global function for slope error calculation that can be used with scipy minimize_scalar
    
    Args:
        t_val (double): Parameter value for Bezier curve
        
    Returns:
        double: Squared difference between current slope and desired slope
    """
    cdef np.ndarray[DTYPE_t, ndim=1] derivative = bezier_derivative(_global_control_points, t_val)
    cdef double dx = derivative[0]
    cdef double dy = derivative[1]
    cdef double current_slope
    
    if fabs(dx) < 1e-10:  # Avoid division by zero
        current_slope = INFINITY
    else:
        current_slope = dy / dx
    
    # Handle infinity in comparison    
    if current_slope == INFINITY and _global_desired_slope == INFINITY:
        return 0.0
    elif current_slope == INFINITY or _global_desired_slope == INFINITY:
        return 1e6  # Large error when one is infinite but not the other
    
    return (current_slope - _global_desired_slope) ** 2

cpdef tuple find_fpr_tpr_for_slope(list control_points, double desired_slope):
    """
    Find the FPR and TPR on the Bézier curve for a given slope.
    
    Args:
        control_points (list): List of control points defining the Bezier curve
        desired_slope (double): Desired slope (dy/dx)
        
    Returns:
        tuple: (fpr, tpr, t) - coordinates and parameter where the slope matches
    """
    global _global_control_points, _global_desired_slope
    
    # Set global variables for optimization
    _global_control_points = control_points
    _global_desired_slope = desired_slope
    
    # Use scalar minimization to find the t that gives the desired slope
    result = minimize_scalar(_slope_error_global, bounds=(0, 1), method='bounded')
    if not result.success:
        raise ValueError("Optimization did not converge")

    cdef double t_optimal = result.x
    cdef np.ndarray[DTYPE_t, ndim=1] point = Bezier(control_points, t_optimal)
    
    return point[0], point[1], t_optimal  # Return FPR, TPR, and t

cpdef tuple find_closest_pair_separate(np.ndarray[DTYPE_t, ndim=1] tprs, 
                                     np.ndarray[DTYPE_t, ndim=1] fprs, 
                                     double desired_tpr, 
                                     double desired_fpr):
    """
    Find the closest (TPR, FPR) pair to a desired point.
    
    Args:
        tprs (np.ndarray): Array of true positive rates
        fprs (np.ndarray): Array of false positive rates
        desired_tpr (double): Target true positive rate
        desired_fpr (double): Target false positive rate
        
    Returns:
        tuple: (closest_tpr, closest_fpr, closest_index)
    """
    cdef Py_ssize_t n = tprs.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] pairs_array = np.column_stack((tprs, fprs))
    cdef np.ndarray[DTYPE_t, ndim=1] distances
    cdef Py_ssize_t closest_index
    
    # Compute the Euclidean distance from each pair to the desired pair
    distances = np.sqrt((pairs_array[:, 0] - desired_tpr) ** 2 + (pairs_array[:, 1] - desired_fpr) ** 2)
    
    # Find the index of the smallest distance
    closest_index = np.argmin(distances)
    
    # Return the closest pair and its index
    return tprs[closest_index], fprs[closest_index], closest_index