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
from multiprocessing import Pool  # Ensure this import is at the top

# Add these to the top of your file
from libc.math cimport INFINITY
from scipy.spatial.distance import cdist
from scipy.optimize import minimize_scalar 
from scipy.special import comb
from cpython.list cimport PyList_Append


# Define numpy types
ctypedef np.float64_t DTYPE_t
ctypedef np.int32_t ITYPE_t

# Import functions from other modules
from cython_modules.clinical_utils import process_roc_chunk

# Wrapper function for multiprocessing
def process_chunk_wrapper(args):
    """
    Wrapper function to unpack arguments for multiprocessing
    
    Args:
        args (tuple): Tuple containing chunk arguments
        
    Returns:
        Processed chunk results
    """
    tprChunk, fprChunk, uTN, uTP, uFN, uFP, u = args
    return process_roc_chunk(tprChunk, fprChunk, uTN, uTP, uFN, uFP, u)

cpdef list modelPriorsOverRoc(dict modelChosen, double uTN, double uTP, double uFN, 
                             double uFP, double u, double HoverB, int num_workers=4):
    """
    Parallelized main function for calculating priors over ROC curve
    
    Args:
        modelChosen: Dictionary containing 'tpr' and 'fpr' arrays
        uTN, uTP, uFN, uFP, u: Utility parameters
        HoverB: Health over Budget ratio
        num_workers: Number of parallel workers
        
    Returns:
        List of [pLs, pStars, pUs]
    """
    cdef list pLs = []
    cdef list pStars = []
    cdef list pUs = []
    cdef np.ndarray tprArray, fprArray
    cdef int chunk_size, i
    
    # Extract tpr and fpr arrays from modelChosen
    tpr_data = np.array(modelChosen['tpr'])
    fpr_data = np.array(modelChosen['fpr'])
    
    # Determine the correct shape of the arrays
    if tpr_data.ndim > 1 and tpr_data.shape[0] == 1:
        tprArray = np.array(tpr_data[0])
        fprArray = np.array(fpr_data[0])
    elif isinstance(tpr_data.tolist(), list) and len(tpr_data) == 1:
        tprArray = np.array(tpr_data[0])
        fprArray = np.array(fpr_data[0])
    elif tpr_data.size > 1:
        tprArray = tpr_data
        fprArray = fpr_data
    else:
        tprArray = np.array(tpr_data.flatten())
        fprArray = np.array(fpr_data.flatten())
    
    # Ensure arrays are not empty
    if tprArray.size <= 1:
        return [[0], [0], [0]]
    
    # Define chunk size based on num_workers
    chunk_size = max(1, len(tprArray) // num_workers)
    
    # Prepare chunks for parallel processing
    chunks = []
    for i in range(num_workers):
        start_idx = i * chunk_size
        # Handle the last chunk which might be larger
        end_idx = min((i + 1) * chunk_size, len(tprArray)) if i < num_workers - 1 else len(tprArray)
        
        # Skip empty chunks
        if start_idx >= end_idx:
            continue
        
        chunks.append((
            tprArray[start_idx:end_idx],
            fprArray[start_idx:end_idx],
            uTN, uTP, uFN, uFP, u
        ))
    
    # Parallel processing using multiprocessing Pool
    with Pool(processes=num_workers) as pool:
        results = pool.map(process_chunk_wrapper, chunks)
    
    # Combine results
    for chunk_result in results:
        chunk_pLs, chunk_pStars, chunk_pUs = chunk_result
        pLs.extend(chunk_pLs)
        pStars.extend(chunk_pStars)
        pUs.extend(chunk_pUs)
    
    return [pLs, pStars, pUs]

cpdef list priorFiller(list priorList, bint lower):
    """
    Some priors are not defined. For those -999, change to 1 or 0 depending on pL or pU
    
    Args: 
        priorList: List of lower or upper thresholds (priors)
        lower: Specifies list of lower or upper thresholds. True for lower, false for upper.
        
    Returns: 
        A modified list of priors that fills in the "NA"(-999) values at the head and tail of the list
    """
    cdef Py_ssize_t index, lenList
    cdef double midPoint, item
    
    lenList = len(priorList)
    midPoint = lenList / 2.0
    
    for index in range(lenList):
        item = priorList[index]
        
        if lower:
            if (index < midPoint) and (lenList > 1):
                if item == -999:
                    priorList[index] = 1.0
            if (index > midPoint) and (lenList > 1):
                if item == -999:
                    priorList[index] = 0.0
        else:
            if (index < midPoint) and (lenList > 1):
                if item == -999:
                    priorList[index] = 0.0
            if (index > midPoint) and (lenList > 1):
                if item == -999:
                    priorList[index] = 0.0
                    
    return priorList

cpdef list priorModifier(list priorList):
    """
    The previous prior filler function did not take care of all the problems. 
    This function provides additional modifications. 
        - for example, "1, 0, 1" the 0 should be a 1. 
        - another example, "0, 0, 1, 0" the 1 should be a 0.
    
    Args: 
        priorList: List of lower or upper thresholds (priors)
        
    Returns: 
        A modified list of priors
    """
    cdef Py_ssize_t index, lenList
    cdef double midPoint, item
    
    lenList = len(priorList)
    midPoint = lenList / 2.0
    
    for index in range(lenList):
        item = priorList[index]
        
        if (index < midPoint - 3) and (lenList > 3):
            if (item == 1.0) and (priorList[index + 2] > priorList[index + 1]) and (priorList[index + 3] > priorList[index + 2]):
                priorList[index] = 0.0
            elif (item == 0.0) and (priorList[index + 2] < priorList[index + 1]) and (priorList[index + 3] < priorList[index + 2]):
                priorList[index] = 1.0
                
        if (index > midPoint + 3) and (lenList > 3):
            if (item == 1.0) and (priorList[index - 2] > priorList[index - 1]) and (priorList[index - 3] > priorList[index - 2]):
                priorList[index] = 0.0
            elif (item == 0.0) and (priorList[index - 2] < priorList[index - 1]) and (priorList[index - 3] < priorList[index - 2]):
                priorList[index] = 1.0
                
        if index == lenList - 1:
            if (priorList[index - 1] != 0.0) and (priorList[index] == 0.0):
                priorList[index] = priorList[index - 1]
                
    return priorList

cpdef list extractThresholds(dict row):
    """
    Based on https://github.com/scikit-learn/scikit-learn/issues/3097, by default 1 is added to the last number to 
    compute the entire ROC curve. Thus, this extracts the thresholds and adjusts those outside the [0,1] range. 
    
    Args:
        row (dict): a row in the dataframe with the column "thresholds" obtained from the model
        
    Returns: 
        a modified list of thresholds. 
    """
    cdef list thresholds = row['thresholds']
    cdef Py_ssize_t i
    cdef double cutoff
    
    if thresholds is not None:
        for i, cutoff in enumerate(thresholds):
            if cutoff > 1.0:
                thresholds[i] = 1.0
        return thresholds
    else:
        return None

cpdef list adjustpLpUClassificationThreshold(list thresholds, list pLs, list pUs):
    """
    Modifies the prior thresholds as well as the predicted probability cutoff thresholds 
    
    Args:
        thresholds (list): List of classification thresholds
        pLs (list): List of lower threshold priors
        pUs (list): List of upper threshold priors
        
    Returns: 
        A list containing [modified_thresholds, modified_pLs, modified_pUs]
    """
    cdef np.ndarray[np.float64_t, ndim=1] np_thresholds
    cdef list modified_pLs, modified_pUs
    
    # Process the priors using the existing functions
    modified_pLs = priorFiller(pLs.copy(), True)
    modified_pLs = priorModifier(modified_pLs)
    
    modified_pUs = priorFiller(pUs.copy(), False)
    modified_pUs = priorModifier(modified_pUs)
    
    # Convert thresholds to numpy array for vectorized operations
    np_thresholds = np.array(thresholds, dtype=np.float64)
    
    # Adjust values > 1
    np_thresholds = np.where(np_thresholds > 1, 1.0, np_thresholds)
    
    # Handle special case for the last threshold
    if np_thresholds[-1] == 0:
        np_thresholds[-1] = 0.0001
        np_thresholds = np.append(np_thresholds, 0.0)
        
        # Adjust pLs and pUs
        modified_pLs[0] = modified_pLs[1]
        modified_pUs[0] = modified_pUs[1]
        
        # Prepend zeros
        modified_pLs = [0.0] + modified_pLs
        modified_pUs = [0.0] + modified_pUs
    
    return [np_thresholds.tolist(), modified_pLs, modified_pUs]

cpdef double eqLine(double x, double x0, double x1, double y0, double y1):
    """
    Find the value of f(x) given x and the two points that make up the line of interest.
    
    Args:
        x (float): desired x
        x0 (float): x coordinate of the first point
        x1 (float): x coordinate of the second point
        y0 (float): y coordinate of the first point
        y1 (float): y coordinate of the second point
        
    Returns: 
        f(x)
    """
    cdef double slope
    
    if x1 == x0:  # Avoid division by zero
        return y0
    
    slope = (y1 - y0) / (x1 - x0)
    return slope * (x - x0) + y0


cpdef tuple calculate_area_chunk(Py_ssize_t start, Py_ssize_t end, list pLs, list pUs, list thresholds):
    """
    Calculate the area for a specific chunk of the data
    
    Args:
        start (int): Starting index
        end (int): Ending index
        pLs (list): List of lower prior thresholds
        pUs (list): List of upper prior thresholds
        thresholds (list): List of classification thresholds
        
    Returns:
        tuple: (area, largestRangePrior, largestRangePriorThresholdIndex)
    """
    cdef double area = 0.0
    cdef double largestRangePrior = 0.0
    cdef int largestRangePriorThresholdIndex = -999
    cdef Py_ssize_t i
    cdef double rangePrior, avgRangePrior
    cdef double x0, x1, pL0, pL1, pU0, pU1, yIntersect
    cdef list xIntersect
    
    # Create a symbol for solving equations
    x = sy.symbols('x')
    
    for i in range(start, end):
        if i < len(pLs) - 1:
            # Case 1: Both pairs have pL < pU (normal case)
            if pLs[i] < pUs[i] and pLs[i + 1] < pUs[i + 1]:
                rangePrior = pUs[i] - pLs[i]
                if rangePrior > largestRangePrior:
                    largestRangePrior = rangePrior
                    largestRangePriorThresholdIndex = i
                
                avgRangePrior = (rangePrior + (pUs[i + 1] - pLs[i + 1])) / 2.0
                area += fabs(avgRangePrior) * fabs(thresholds[i + 1] - thresholds[i])

            # Case 2: First pair has pL > pU, second pair has pL < pU (crossing from above)
            elif pLs[i] > pUs[i] and pLs[i + 1] < pUs[i + 1]:
                x0 = thresholds[i]
                x1 = thresholds[i + 1]
                if x0 != x1:
                    pL0 = pLs[i]
                    pL1 = pLs[i + 1]
                    pU0 = pUs[i]
                    pU1 = pUs[i + 1]
                    
                    # Find intersection point using sympy
                    xIntersect = sy.solve(eqLine(x, x0, x1, pL0, pL1) - eqLine(x, x0, x1, pU0, pU1), x)
                    if len(xIntersect) > 0:
                        yIntersect = eqLine(float(xIntersect[0]), x0, x1, pL0, pL1)
                        avgRangePrior = (0.0 + (pUs[i + 1] - pLs[i + 1])) / 2.0
                        area += fabs(avgRangePrior) * fabs(thresholds[i + 1] - float(xIntersect[0]))

            # Case 3: First pair has pL < pU, second pair has pL > pU (crossing from below)
            elif pLs[i] < pUs[i] and pLs[i + 1] > pUs[i + 1]:
                x0 = thresholds[i]
                x1 = thresholds[i + 1]
                if x0 != x1:
                    pL0 = pLs[i]
                    pL1 = pLs[i + 1]
                    pU0 = pUs[i]
                    pU1 = pUs[i + 1]
                    
                    # Find intersection point using sympy
                    xIntersect = sy.solve(eqLine(x, x0, x1, pL0, pL1) - eqLine(x, x0, x1, pU0, pU1), x)
                    if len(xIntersect) == 0:
                        xIntersect = [0.0]
                    
                    yIntersect = eqLine(float(xIntersect[0]), x0, x1, pL0, pL1)
                    avgRangePrior = (0.0 + (pUs[i] - pLs[i])) / 2.0
                    area += fabs(avgRangePrior) * fabs(float(xIntersect[0]) - thresholds[i])
    
    return area, largestRangePrior, largestRangePriorThresholdIndex

cpdef list applicableArea(dict modelRow, list thresholds, tuple utils, double p, double HoverB, int num_workers=4):
    """
    Calculate the applicable area using parallel processing
    
    Args:
        modelRow (dict): Dictionary containing model data
        thresholds (list): List of classification thresholds
        utils (tuple): Tuple of utility parameters (uTN, uTP, uFN, uFP, u)
        p (float): Probability threshold
        HoverB (float): Health over Budget ratio
        num_workers (int): Number of parallel workers
        
    Returns:
        list: [area, largestRangePriorThresholdIndex, withinRange, leastViable, uFP]
    """
    cdef double uTN, uTP, uFN, uFP, u
    cdef double area = 0.0
    cdef double largestRangePrior = 0.0
    cdef int largestRangePriorThresholdIndex = -999
    cdef bint withinRange = False
    cdef list priorDistributionArray = []
    cdef double leastViable = 1.0
    cdef int i
    cdef list pLs, pStars, pUs
    cdef np.ndarray np_thresholds
    cdef int chunk_size
    
    # Unpack utilities
    uTN, uTP, uFN, uFP, u = utils
    
    # Get prior thresholds
    results = modelPriorsOverRoc(modelRow, uTN, uTP, uFN, uFP, u, HoverB, num_workers)
    pLs, pStars, pUs = results
    
    # Adjust thresholds
    np_thresholds = np.array(thresholds)
    np_thresholds = np.where(np_thresholds > 1, 1.0, np_thresholds)
    thresholds_adjusted, pLs, pUs = adjustpLpUClassificationThreshold(np_thresholds.tolist(), pLs, pUs)
    
    # Parallel calculation of area
    chunk_size = max(1, len(pLs) // num_workers)
    results_area = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for i in range(num_workers):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, len(pLs)) if i < num_workers - 1 else len(pLs)
            
            # Skip empty chunks
            if start >= end:
                continue
                
            futures.append(executor.submit(
                calculate_area_chunk, 
                start, end, pLs, pUs, thresholds_adjusted
            ))
        
        # Collect results
        for future in concurrent.futures.as_completed(futures):
            chunk_area, chunk_largest_range, chunk_largest_index = future.result()
            area += chunk_area
            
            if chunk_largest_range > largestRangePrior:
                largestRangePrior = chunk_largest_range
                largestRangePriorThresholdIndex = chunk_largest_index
    
    # Finalize area calculation
    area = min(np.round(float(area), 3), 1.0)  # Round and cap area at 1
    
    # Check if p is within the range
    withinRange = (p > 0.0 and p < largestRangePrior)
    
    return [area, largestRangePriorThresholdIndex, withinRange, leastViable, uFP]
