# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
import sympy as sy
import pandas as pd
import concurrent.futures
from libc.math cimport sqrt, fabs

# Define numpy types
ctypedef np.float64_t DTYPE_t
ctypedef np.int32_t ITYPE_t

def treatAll(p, uFP, uTP):
    """
    Calculates expected utility for treating all patients.
    """
    return p * uTP + (1 - p) * uFP

def treatNone(p, uFN, uTN):
    """
    Calculates expected utility for treating no patients.
    """
    return p * uFN + (1 - p) * uTN

def test(p, sens, spec, uTN, uTP, uFN, uFP, testCost):
    """
    Calculates expected utility for testing patients.
    """
    return (p * sens * uTP + 
            p * (1 - sens) * uFN + 
            (1 - p) * (1 - spec) * uFP + 
            (1 - p) * spec * uTN - 
            testCost)

def modelPriorsOverRoc(modelTest, uTN, uTP, uFN, uFP, testCost, HoverB):
    """
    Computes decision thresholds (pL, pStar, pU) for different operating points on the ROC curve.
    """
    cdef int n = len(modelTest)
    cdef np.ndarray[DTYPE_t, ndim=1] pLs = np.zeros(n, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] pStars = np.zeros(n, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] pUs = np.zeros(n, dtype=np.float64)
    cdef int i
    cdef double fpr, tpr, sens, spec
    
    # Define symbolic variable
    xVar = sy.symbols('xVar')
    
    # Calculate pStar (treatment threshold)
    pStar_expr = sy.solve(treatAll(xVar, uFP, uTP) - treatNone(xVar, uFN, uTN), xVar)
    if len(pStar_expr) > 0:
        pStar_value = float(pStar_expr[0])
    else:
        pStar_value = 0.5  # Default if no solution
    
    # Fill pStars array with the same value
    for i in range(n):
        pStars[i] = pStar_value
    
    # Calculate pLs and pUs for each point on the ROC curve
    for i in range(n):
        fpr = modelTest['fpr'].iloc[i]
        tpr = modelTest['tpr'].iloc[i]
        sens = tpr
        spec = 1 - fpr
        
        # Solve for pL
        pL_expr = sy.solve(treatNone(xVar, uFN, uTN) - test(xVar, sens, spec, uTN, uTP, uFN, uFP, testCost), xVar)
        if len(pL_expr) > 0 and pL_expr[0] >= 0 and pL_expr[0] <= 1:
            pLs[i] = float(pL_expr[0])
        else:
            pLs[i] = 0.0
        
        # Solve for pU
        pU_expr = sy.solve(treatAll(xVar, uFP, uTP) - test(xVar, sens, spec, uTN, uTP, uFN, uFP, testCost), xVar)
        if len(pU_expr) > 0 and pU_expr[0] >= 0 and pU_expr[0] <= 1:
            pUs[i] = float(pU_expr[0])
        else:
            pUs[i] = 1.0
    
    return pLs, pStars, pUs

def adjustpLpUClassificationThreshold(np.ndarray[DTYPE_t, ndim=1] thresholds, 
                                      np.ndarray[DTYPE_t, ndim=1] pLs, 
                                      np.ndarray[DTYPE_t, ndim=1] pUs):
    """
    Adjusts pL and pU values based on thresholds.
    """
    cdef int n = len(thresholds)
    cdef int i
    
    # Sort by thresholds
    cdef np.ndarray[ITYPE_t, ndim=1] indices = np.argsort(thresholds)
    thresholds = thresholds[indices]
    pLs = pLs[indices]
    pUs = pUs[indices]
    
    # Ensure pLs and pUs are in valid range
    for i in range(n):
        if pLs[i] < 0:
            pLs[i] = 0.0
        elif pLs[i] > 1:
            pLs[i] = 1.0
            
        if pUs[i] < 0:
            pUs[i] = 0.0
        elif pUs[i] > 1:
            pUs[i] = 1.0
    
    return thresholds, pLs, pUs

def calculate_area_chunk(int start, int end, np.ndarray[DTYPE_t, ndim=1] pLs, 
                         np.ndarray[DTYPE_t, ndim=1] pUs, np.ndarray[DTYPE_t, ndim=1] thresholds):
    """
    Calculates area under the pUs and above the pLs curves for a chunk of thresholds.
    """
    cdef double area = 0.0
    cdef double width, height
    cdef int i
    cdef double largest_range = 0.0
    cdef int largest_index = start
    
    for i in range(start, end-1):
        width = thresholds[i+1] - thresholds[i]
        if pUs[i] >= pLs[i]:  # Valid region
            height = pUs[i] - pLs[i]
            area += width * height
            
            if height > largest_range:
                largest_range = height
                largest_index = i
    
    return area, largest_range, largest_index