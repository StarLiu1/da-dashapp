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

from cython_modules.bezier_utils import find_fpr_tpr_for_slope


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------

def treatAll(x, uFP, uTP):
    """
    Expected value calculation for the option treat all
    
    Args: 
        x (sympy symbol or float): probability
        uFP (sympy symbol or float): utility of false positive
        uTP (sympy symbol or float): utility of true positive
        
    Returns: 
        expected utility (sympy expression or float)
    """
    return x * uTP + (1 - x) * uFP

def treatNone(x, uFN, uTN):
    """
    Expected value calculation for the option treat none
    
    Args: 
        x (sympy symbol or float): probability
        uFN (sympy symbol or float): utility of false negative
        uTN (sympy symbol or float): utility of true negative
        
    Returns: 
        expected utility (sympy expression or float)
    """
    return x * uFN + (1 - x) * uTN

def test(x, sensitivity, specificity, uTN, uTP, uFN, uFP, u):
    """
    Expected value calculation for the option test
    
    Args: 
        x (sympy symbol or float): probability
        sensitivity (sympy symbol or float): sensitivity of the test
        specificity (sympy symbol or float): specificity of the test
        uTN (sympy symbol or float): utility of true negative
        uTP (sympy symbol or float): utility of true positive
        uFN (sympy symbol or float): utility of false negative
        uFP (sympy symbol or float): utility of false positive
        u (sympy symbol or float): utility of the test itself
        
    Returns: 
        expected utility (sympy expression or float)
    """
    result = (x * sensitivity * uTP + 
              x * (1 - sensitivity) * uFN + 
              (1 - x) * (1 - specificity) * uFP + 
              (1 - x) * specificity * uTN + 
              u)
    return result

cpdef list pLpStarpUThresholds(double sens, double spec, double uTN, double uTP, 
                              double uFN, double uFP, double u):
    """
    Identifies the three thresholds formed by the three utility lines
    
    Args: 
        sens (float): sensitivity of the test
        spec (float): specificity of the test
        uTN (float): utility of true negative
        uTP (float): utility of true positive
        uFN (float): utility of false negative
        uFP (float): utility of false positive
        u: utility of the test itself
        
    Returns: 
        a list of three thresholds (pL, pStar, and pU): [pL, pStar, pU]
    """
    cdef double pU, pStar, pL
    
    # We'll use sympy for solving equations, as in the original code
    x = sy.symbols('x')
    
    # Solve for upper threshold formed by test and treat all
    pU_sol = sy.solve(treatAll(x, uFP, uTP) - test(x, sens, spec, uTN, uTP, uFN, uFP, u), x)
    
    # Solve for treatment threshold formed by treat all and treat none
    pStar_sol = sy.solve(treatAll(x, uFP, uTP) - treatNone(x, uFN, uTN), x)
    
    # Solve for lower threshold formed by treat none and test
    pL_sol = sy.solve(treatNone(x, uFN, uTN) - test(x, sens, spec, uTN, uTP, uFN, uFP, u), x)
    
    # Handle solutions properly
    if len(pU_sol) == 0:
        pU = -999.0
    else:
        pU = float(pU_sol[0])
        if pU > 1.0:
            pU = 1.0
        elif pU < 0.0 and pU != -999.0:
            pU = 0.0
            
    if len(pStar_sol) == 0:
        pStar = -999.0
    else:
        pStar = float(pStar_sol[0])
        if pStar > 1.0:
            pStar = 1.0
        elif pStar < 0.0 and pStar != -999.0:
            pStar = 0.0
            
    if len(pL_sol) == 0:
        pL = -999.0
    else:
        pL = float(pL_sol[0])
        if pL > 1.0:
            pL = 1.0
        elif pL < 0.0 and pL != -999.0:
            pL = 0.0
    
    return [pL, pStar, pU]

cpdef tuple process_roc_chunk(np.ndarray[DTYPE_t, ndim=1] tpr_chunk, 
                             np.ndarray[DTYPE_t, ndim=1] fpr_chunk, 
                             double uTN, double uTP, double uFN, double uFP, double u):
    """
    Helper function for processing a chunk of TPR/FPR arrays
    
    Args:
        tpr_chunk: Array of true positive rates
        fpr_chunk: Array of false positive rates
        uTN, uTP, uFN, uFP, u: Utility parameters
        
    Returns:
        Tuple of (pLs, pStars, pUs) lists
    """
    cdef list pLs = []
    cdef list pStars = []
    cdef list pUs = []
    cdef double pL, pStar, pU
    cdef Py_ssize_t i
    cdef Py_ssize_t n = min(len(tpr_chunk), len(fpr_chunk))
    
    for i in range(n):
        thresholds = pLpStarpUThresholds(tpr_chunk[i], 1.0 - fpr_chunk[i], uTN, uTP, uFN, uFP, u)
        pL, pStar, pU = thresholds[0], thresholds[1], thresholds[2]
        
        PyList_Append(pLs, pL)
        PyList_Append(pStars, pStar)
        PyList_Append(pUs, pU)
        
    return pLs, pStars, pUs



cpdef list pLpUThresholds(double sens, double spec, double uTN, double uTP, double uFN, double uFP, double u):
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
    cdef double pU, pL
    cdef list pU_sol, pL_sol
    
    # Initiate a variable called x (prior)
    x = sy.symbols('x')
    
    # Solve for upper threshold formed by test and treat all
    pU_sol = sy.solve(treatAll(x, uFP, uTP) - test(x, sens, spec, uTN, uTP, uFN, uFP, u), x)
    
    # Solve for lower threshold formed by treat none and test
    # Add very small value to sensitivity and specificity to avoid numerical issues
    pL_sol = sy.solve(treatNone(x, uFN, uTN) - test(x, sens + 0.00000001, spec + 0.00000001, uTN, uTP, uFN, uFP, u), x)
    
    # Handle placeholder values when there are not two thresholds formed
    if len(pU_sol) == 0:
        pU = -1.0
    else:
        pU = float(pU_sol[0])
        
    if pU > 1.0:
        pU = 1.0
    elif pU < 0.0 and pU != -999.0:
        pU = 0.0
        
    if len(pL_sol) == 0:
        pL = -1.0
    else:
        pL = float(pL_sol[0])
        
    if pL > 1.0:
        pL = 1.0
    elif pL < 0.0 and pL != -999.0:
        pL = 0.0

    return [pL, pU]

cpdef list cleanThresholds(list array_thresholds):
    """
    Replace inf and -inf values in thresholds array
    
    Args:
        array_thresholds (list): List of threshold values
        
    Returns:
        list: Cleaned threshold values
        
    """
    cdef Py_ssize_t i
    cdef double value
    
    for i, value in enumerate(array_thresholds):
        if value == INFINITY or value > 1.0:
            array_thresholds[i] = 1.0
        elif value == -INFINITY or value < 0.0:
            array_thresholds[i] = 0.0

    return array_thresholds

cpdef np.ndarray[np.float64_t, ndim=1] slope(np.ndarray[np.float64_t, ndim=1] fprs, 
                                            np.ndarray[np.float64_t, ndim=1] tprs):
    """
    Calculate the slope between points on the ROC curve
    
    Args:
        fprs (array): False positive rates
        tprs (array): True positive rates
        
    Returns:
        array: Slopes between consecutive points
    """
    return np.round(np.diff(tprs) / np.diff(fprs), 5)

cpdef np.ndarray[np.float64_t, ndim=1] derivSlope(np.ndarray[np.float64_t, ndim=1] slopes, 
                                                 np.ndarray[np.float64_t, ndim=1] fprs):
    """
    Calculate the derivative of the slope
    
    Args:
        slopes (array): Slopes between points
        fprs (array): False positive rates
        
    Returns:
        array: Derivatives of slopes
    """
    return np.diff(slopes) / np.diff(fprs)[1:]

cpdef tuple optimalPointOnRoc(np.ndarray[np.float64_t, ndim=1] false_positive_rate, 
                             np.ndarray[np.float64_t, ndim=1] true_positive_rate, 
                             np.ndarray[np.float64_t, ndim=1] prob_cutoffs, 
                             double desired_slope, 
                             object fitted_curve_points, 
                             int n_int=100):
    """
    Find the optimal point on the ROC curve that maximizes patient utility, using np.diff
    
    Args: 
        false_positive_rate (array): fprs of the test
        true_positive_rate (array): tprs of the test
        prob_cutoffs (array): Probability cutoffs
        desired_slope (float): Slope of interest
        fitted_curve_points (object): Fitted curve points
        n_int (int): Number of intervals
        
    Returns: 
        tuple: (closest_slope, original_fpr, original_tpr, closest_prob_cutoff)
    """
    cdef double closest_slope, original_fpr, original_tpr, closest_prob_cutoff
    cdef tuple cutoff_rational
    cdef double closest_fpr, closest_tpr
    cdef int index, closest_slope_index
    cdef np.ndarray[np.float64_t, ndim=1] slopes_fine
    
    # Handle special case
    if desired_slope == 0:
        closest_slope, original_fpr, original_tpr, closest_prob_cutoff = 0.0, 1.0, 1.0, 0.0
    else:
        # Find FPR and TPR for the desired slope using fitted curve
        cutoff_rational = find_fpr_tpr_for_slope(fitted_curve_points, desired_slope)
        closest_fpr, closest_tpr = cutoff_rational[0], cutoff_rational[1]
        
        # Find the closest pair of TPR and FPR from the original arrays
        original_tpr, original_fpr, index = find_closest_pair_separate(
            true_positive_rate, false_positive_rate, closest_tpr, closest_fpr
        )
        
        closest_prob_cutoff = prob_cutoffs[index]
        
        # Calculate slopes between points
        slopes_fine = slope(false_positive_rate, true_positive_rate)
        
        # Find the index of the point with the slope closest to the desired_slope
        closest_slope_index = int(np.argmin(np.abs(slopes_fine - desired_slope)))
        closest_slope = slopes_fine[closest_slope_index]
    
    return closest_slope, original_fpr, original_tpr, closest_prob_cutoff

cpdef double slope_of_line(double x1, double y1, double x2, double y2):
    """
    Calculate the slope of a line between two points
    
    Args:
        x1 (float): x-coordinate of first point
        y1 (float): y-coordinate of first point
        x2 (float): x-coordinate of second point
        y2 (float): y-coordinate of second point
        
    Returns:
        float: Slope of the line
    """
    if x2 == x1:  # Avoid division by zero
        return INFINITY if y2 > y1 else -INFINITY
        
    return (y2 - y1) / (x2 - x1)

cpdef tuple find_closest_pair_separate(np.ndarray[np.float64_t, ndim=1] tpr_array, 
                                      np.ndarray[np.float64_t, ndim=1] fpr_array, 
                                      double target_tpr, double target_fpr):
    """
    Find the closest pair of TPR and FPR values to the target
    
    Args:
        tpr_array (array): Array of TPR values
        fpr_array (array): Array of FPR values
        target_tpr (float): Target TPR value
        target_fpr (float): Target FPR value
        
    Returns:
        tuple: (closest_tpr, closest_fpr, index)
    """
    cdef Py_ssize_t i
    cdef double min_distance = INFINITY
    cdef int closest_index = 0
    cdef double distance
    
    # Find the index of the closest point
    for i in range(len(tpr_array)):
        distance = (tpr_array[i] - target_tpr)**2 + (fpr_array[i] - target_fpr)**2
        if distance < min_distance:
            min_distance = distance
            closest_index = i
    
    return tpr_array[closest_index], fpr_array[closest_index], closest_index


cpdef tuple find_lower_vertices(np.ndarray[DTYPE_t, ndim=1] fprs, 
                               np.ndarray[DTYPE_t, ndim=1] tprs):
    """
    Find the lower vertices of the ROC curve
    
    Args:
        fprs (np.ndarray): Array of false positive rates
        tprs (np.ndarray): Array of true positive rates
        
    Returns:
        tuple: (lower_roc_fpr, lower_roc_tpr)
    """
    cdef np.ndarray[DTYPE_t, ndim=1] l_roc_tpr = np.array([0.0], dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] l_roc_fpr = np.array([0.0], dtype=np.float64)
    cdef Py_ssize_t idx = 1
    cdef Py_ssize_t n = len(tprs)
    
    # For a pair of tpr and fpr
    while idx > 0 and idx < n - 1:
        # If fpr increases, then add to lower roc
        if fprs[idx] > fprs[idx - 1] and tprs[idx] == tprs[idx - 1]:
            l_roc_tpr = np.append(l_roc_tpr, tprs[idx])
            l_roc_fpr = np.append(l_roc_fpr, fprs[idx])
        idx += 1

    # Append end points
    l_roc_tpr = np.append(l_roc_tpr, [1.0])
    l_roc_fpr = np.append(l_roc_fpr, [1.0])
    
    return l_roc_fpr, l_roc_tpr

cpdef np.ndarray[ITYPE_t, ndim=1] find_lower_roc_controls(np.ndarray[DTYPE_t, ndim=1] fprs, 
                                                        np.ndarray[DTYPE_t, ndim=1] tprs):
    """
    Find control points for the lower ROC curve
    
    Args:
        fprs (np.ndarray): Array of false positive rates
        tprs (np.ndarray): Array of true positive rates
        
    Returns:
        np.ndarray: Array of indices for control points
    """
    cdef Py_ssize_t n = len(tprs)
    cdef np.ndarray[ITYPE_t, ndim=1] indices = np.array([0], dtype=np.int32)
    cdef Py_ssize_t i = 0, j, slope_idx, next_idx
    cdef np.ndarray[DTYPE_t, ndim=1] slopes, slope_of_slopes
    cdef double slope_val
    
    # Find where the slopes goes from either pos or inf to negative
    while i < n:
        slopes = np.empty(0, dtype=np.float64)
        
        # Calculate slopes relative to current point
        for j in range(i+1, n):
            if fprs[j] != fprs[i]:  # Avoid division by zero
                slope_val = (tprs[j] - tprs[i]) / (fprs[j] - fprs[i])
                slopes = np.append(slopes, slope_val)
            else:
                # Handle vertical line case (infinite slope)
                slopes = np.append(slopes, np.inf)
        
        if len(slopes) > 1:
            slope_of_slopes = np.diff(slopes)
            slope_idx = 0
            next_idx = i
            
            while slope_idx < (len(slope_of_slopes) - 1):
                if (slope_of_slopes[slope_idx] <= 0) and (slope_of_slopes[slope_idx+1] > 0):
                    # Then this idx is what we want to keep track of
                    next_idx = slope_idx + i + 2
                    indices = np.append(indices, next_idx)
                    slope_idx = len(slope_of_slopes)
                else:
                    slope_idx += 1
                    next_idx = i + 1
                    indices = np.append(indices, next_idx)
                    
            if next_idx > i:
                i = next_idx
            else:
                i += 1
        else:
            i += 1
    
    # Add the last point
    indices = np.append(indices, n - 1)
    
    return indices.astype(np.int32)

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

cpdef tuple filter_points(np.ndarray[DTYPE_t, ndim=1] FPR, 
                         np.ndarray[DTYPE_t, ndim=1] TPR):
    """
    Filter redundant points from the ROC curve
    
    Args:
        FPR (np.ndarray): Array of false positive rates
        TPR (np.ndarray): Array of true positive rates
        
    Returns:
        tuple: Filtered arrays of (filtered_FPR, filtered_TPR)
    """
    cdef Py_ssize_t n = len(TPR)
    cdef list filtered_FPR = [FPR[0]]
    cdef list filtered_TPR = [TPR[0]]
    cdef Py_ssize_t i
    
    for i in range(1, n - 1):
        if TPR[i] != TPR[i + 1]:
            filtered_FPR.append(FPR[i])
            filtered_TPR.append(TPR[i])
    
    filtered_FPR.append(FPR[n-1])
    filtered_TPR.append(TPR[n-1])
    
    return np.array(filtered_FPR, dtype=np.float64), np.array(filtered_TPR, dtype=np.float64)

