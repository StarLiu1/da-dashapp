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

import concurrent.futures # for apar calculation

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

################################################ ApAr 

def pLpStarpUThresholds(sens, spec, uTN, uTP, uFN, uFP, u):
    """
    Vectorized function that identifies the three thresholds formed by the three utility lines.
    
    Args: 
        sens (float or array-like): sensitivity of the test
        spec (float or array-like): specificity of the test
        uTN (float): utility of true negative
        uTP (float): utility of true positive
        uFN (float): utility of false negative
        uFP (float): utility of false positive
        u (float): utility of the test itself
        
    Returns: 
        If scalar inputs: a list of three thresholds [pL, pStar, pU]
        If array inputs: three lists [pLs, pStars, pUs] containing thresholds for each input pair
    """
    # Determine if inputs are scalar or array-like
    try:
        # Try to iterate through sens
        sens_iter = iter(sens)
        # If successful, inputs are array-like
        array_input = True
    except TypeError:
        # If TypeError, inputs are scalar
        array_input = False
    
    # For scalar inputs, use the original approach
    if not array_input:
        x = sy.symbols('x')
        
        # Calculate pStar (treatment threshold) - this is independent of sens/spec
        pStar_eq = treatAll(x, uFP, uTP) - treatNone(x, uFN, uTN)
        pStar_sol = sy.solve(pStar_eq, x)
        pStar = -999 if (len(pStar_sol) == 0) else float(pStar_sol[0])
        pStar = 1 if (pStar > 1) else pStar
        pStar = 0 if ((pStar < 0) & (pStar != -999)) else pStar
        
        # Calculate pU (upper threshold)
        pU_eq = treatAll(x, uFP, uTP) - test(x, sens, spec, uTN, uTP, uFN, uFP, u)
        pU_sol = sy.solve(pU_eq, x)
        pU = -999 if (len(pU_sol) == 0) else float(pU_sol[0])
        pU = 1 if (pU > 1) else pU
        pU = 0 if ((pU < 0) & (pU != -999)) else pU
        
        # Calculate pL (lower threshold)
        pL_eq = treatNone(x, uFN, uTN) - test(x, sens, spec, uTN, uTP, uFN, uFP, u)
        pL_sol = sy.solve(pL_eq, x)
        pL = -999 if (len(pL_sol) == 0) else float(pL_sol[0])
        pL = 1 if (pL > 1) else pL
        pL = 0 if ((pL < 0) & (pL != -999)) else pL
        
        return [pL, pStar, pU]
    else:
        # For array inputs, we'll calculate pStar once (it's the same for all pairs)
        # and then loop through sens/spec pairs for pL and pU
        sens_array = np.asarray(sens)
        spec_array = np.asarray(spec)
        
        # Calculate pStar (independent of sens/spec)
        x = sy.symbols('x')
        pStar_eq = treatAll(x, uFP, uTP) - treatNone(x, uFN, uTN)
        pStar_sol = sy.solve(pStar_eq, x)
        pStar_val = -999 if (len(pStar_sol) == 0) else float(pStar_sol[0])
        pStar_val = 1 if (pStar_val > 1) else pStar_val
        pStar_val = 0 if ((pStar_val < 0) & (pStar_val != -999)) else pStar_val
        
        # Initialize result lists
        pLs = []
        pStars = []
        pUs = []
        
        # Process each (sens, spec) pair
        for s, sp in zip(sens_array, spec_array):
            # Calculate pU
            pU_eq = treatAll(x, uFP, uTP) - test(x, s, sp, uTN, uTP, uFN, uFP, u)
            pU_sol = sy.solve(pU_eq, x)
            pU = -999 if (len(pU_sol) == 0) else float(pU_sol[0])
            pU = 1 if (pU > 1) else pU
            pU = 0 if ((pU < 0) & (pU != -999)) else pU
            
            # Calculate pL
            pL_eq = treatNone(x, uFN, uTN) - test(x, s, sp, uTN, uTP, uFN, uFP, u)
            pL_sol = sy.solve(pL_eq, x)
            pL = -999 if (len(pL_sol) == 0) else float(pL_sol[0])
            pL = 1 if (pL > 1) else pL
            pL = 0 if ((pL < 0) & (pL != -999)) else pL
            
            # Add results to lists
            pLs.append(pL)
            pStars.append(pStar_val)  # Same for all pairs
            pUs.append(pU)
        
        return [pLs, pStars, pUs]

# Helper function for processing a chunk of TPR/FPR arrays
def process_roc_chunk_vectorized(tpr_chunk, fpr_chunk, uTN, uTP, uFN, uFP, u):
    """
    Vectorized processing of ROC chunks
    
    Args:
        tpr_chunk (array-like): Chunk of TPR values
        fpr_chunk (array-like): Chunk of FPR values
        uTN, uTP, uFN, uFP, u: Utility parameters
        
    Returns:
        Lists of pL, pStar, and pU values
    """
    tpr_array = np.array(tpr_chunk)
    fpr_array = np.array(fpr_chunk)
    spec_array = 1 - fpr_array
    
    # pLpStarpUThresholds already returns lists for array inputs
    return pLpStarpUThresholds(tpr_array, spec_array, uTN, uTP, uFN, uFP, u)

# Define this function outside modelPriorsOverRoc
def process_args(args):
    tpr, fpr, uTN, uTP, uFN, uFP, u = args
    # Convert fpr to specificity
    spec = [1-fp for fp in fpr]
    # Use the vectorized function directly
    return pLpStarpUThresholds(tpr, spec, uTN, uTP, uFN, uFP, u)

def modelPriorsOverRoc(modelChosen, uTN, uTP, uFN, uFP, u, HoverB, num_workers=4):
    # Extract tpr and fpr arrays from modelChosen - vectorized approach
    if isinstance(modelChosen['tpr'], list) and len(modelChosen['tpr']) > 0:
        if isinstance(modelChosen['tpr'][0], list):
            tprArray = np.array(modelChosen['tpr'][0])
            fprArray = np.array(modelChosen['fpr'][0])
        else:
            tprArray = np.array(modelChosen['tpr'])
            fprArray = np.array(modelChosen['fpr'])
    else:
        tprArray = np.array(modelChosen['tpr'])
        fprArray = np.array(modelChosen['fpr'])
        if tprArray.ndim > 1:
            tprArray = tprArray[0]
            fprArray = fprArray[0]

    # Ensure arrays are not empty
    if tprArray.size <= 1:
        return [[0], [0], [0]]

    # Define chunk size based on num_workers
    chunk_size = max(1, len(tprArray) // num_workers)
    
    # Create chunks for parallel processing
    tpr_chunks = [tprArray[i:i+chunk_size] for i in range(0, len(tprArray), chunk_size)]
    fpr_chunks = [fprArray[i:i+chunk_size] for i in range(0, len(fprArray), chunk_size)]
    
    # Process in parallel using the defined function instead of lambda
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        args_list = [(tpr, fpr, uTN, uTP, uFN, uFP, u) for tpr, fpr in zip(tpr_chunks, fpr_chunks)]
        results = list(executor.map(process_args, args_list))
    
    # Combine results
    pLs = []
    pStars = []
    pUs = []
    for chunk_pLs, chunk_pStars, chunk_pUs in results:
        pLs.extend(chunk_pLs)
        pStars.extend(chunk_pStars)
        pUs.extend(chunk_pUs)
    
    return [pLs, pStars, pUs]
    
def priorFiller(priorList, lower: bool):
    """
    Some priors are not defined. For those -999, change to 1 or 0 depending on pL or pU
    
    Args: 
        priorList (list): list of lower or upper thresholds (priors)
        lower (bool): specifies list of lower or upper thresholds. True for lower, false for upper.
        
    Returns: 
        A modified list of priors that fills in the "NA"(-999) values at the head and tail of the list
    """
    priorArray = np.array(priorList)
    lenList = len(priorArray)
    midPoint = lenList / 2
    
    # Create an index array for the positions
    indices = np.arange(lenList)
    
    if lower:
        # For lower thresholds
        mask_first_half = (indices < midPoint) & (priorArray == -999)
        mask_second_half = (indices >= midPoint) & (priorArray == -999)
        
        priorArray[mask_first_half] = 1
        priorArray[mask_second_half] = 0
    else:
        # For upper thresholds
        mask_first_half = (indices < midPoint) & (priorArray == -999)
        mask_second_half = (indices >= midPoint) & (priorArray == -999)
        
        priorArray[mask_first_half] = 0
        priorArray[mask_second_half] = 0
    
    return priorArray.tolist()

def priorModifier(priorList):
    """
    The previous prior filler function did not take care of all the problems. 
    This function provides additional modifications.
    
    Args: 
        priorList (list): list of lower or upper thresholds (priors)
        
    Returns: 
        A modified list of priors
    """
    priorArray = np.array(priorList)
    lenList = len(priorArray)
    midPoint = lenList / 2
    
    # Create modified arrays for easier conditions checking
    shifted_plus_1 = np.roll(priorArray, -1)
    shifted_plus_2 = np.roll(priorArray, -2)
    shifted_plus_3 = np.roll(priorArray, -3)
    
    shifted_minus_1 = np.roll(priorArray, 1)
    shifted_minus_2 = np.roll(priorArray, 2)
    shifted_minus_3 = np.roll(priorArray, 3)
    
    # Indices for first half
    first_half = np.arange(lenList) < midPoint
    
    # Conditions for first half
    cond1 = (priorArray == 1) & (shifted_plus_2 > shifted_plus_1) & (shifted_plus_3 > shifted_plus_2) & first_half
    cond2 = (priorArray == 0) & (shifted_plus_2 < shifted_plus_1) & (shifted_plus_3 < shifted_plus_2) & first_half
    
    # Apply conditions for first half
    priorArray[cond1] = 0
    priorArray[cond2] = 1
    
    # Indices for second half
    second_half = np.arange(lenList) >= midPoint
    
    # Conditions for second half
    cond3 = (priorArray == 1) & (shifted_minus_2 > shifted_minus_1) & (shifted_minus_3 > shifted_minus_2) & second_half
    cond4 = (priorArray == 0) & (shifted_minus_2 < shifted_minus_1) & (shifted_minus_3 < shifted_minus_2) & second_half
    
    # Apply conditions for second half
    priorArray[cond3] = 0
    priorArray[cond4] = 1
    
    # Special condition for last element
    if lenList > 1 and priorArray[-1] == 0 and priorArray[-2] != 0:
        priorArray[-1] = priorArray[-2]
    
    return priorArray.tolist()

def extractThresholds(row):
    """
    Based on https://github.com/scikit-learn/scikit-learn/issues/3097, by default 1 is added to the last number to 
    Extracts and adjusts the thresholds to ensure they are within the [0,1] range.
    
    Args:
        row (dict): a dictionary with the key "thresholds" from the model
        
    Returns: 
        a modified list of thresholds. 
    """
    thresholds = row['thresholds']
    if thresholds is not None:
        # Vectorized approach to cap thresholds at 1
        thresholds = np.array(thresholds)
        thresholds = np.where(thresholds > 1, 1, thresholds)
        return thresholds.tolist()
    else:
        return None
    

def adjustpLpUClassificationThreshold(thresholds, pLs, pUs):
    """
    Modifies the prior thresholds as well as the predicted probability cutoff thresholds 
    
    Args:
        thresholds (list): thresholds obtained from the model
        pLs (list): lower thresholds
        pUs (list): upper thresholds
        
    Returns: 
        a list containing modified thresholds, pLs, and pUs
    """
    # Convert inputs to numpy arrays for vectorized operations
    thresholds = np.array(thresholds)
    pLs = np.array(priorFiller(pLs, True))
    pLs = np.array(priorModifier(pLs.tolist()))
    pUs = np.array(priorFiller(pUs, False))
    pUs = np.array(priorModifier(pUs.tolist()))
    
    # Adjust thresholds
    thresholds = np.where(thresholds > 1, 1, thresholds)
    
    # Check if last threshold is 0 and adjust accordingly
    if thresholds[-1] == 0:
        thresholds[-1] = 0.0001
        thresholds = np.append(thresholds, 0)
        
        # Adjust pLs and pUs
        pLs = np.append(np.array([0]), pLs)
        pUs = np.append(np.array([0]), pUs)
        pLs[1] = pLs[2]  # Adjust second element based on third
        pUs[1] = pUs[2]  # Adjust second element based on third
    
    return [thresholds, pLs, pUs]

def eqLine(x, x0, x1, y0, y1):
    """
    Find the value of f(x) given x and the two points that make up the line of interest.
    
    Args:
        x (float): desired x
        x0 (float): x coordinate of the first point
        x1 (float): y coordinate of the first point
        y0 (float): x coordinate of the second point
        y1 (float): y coordinate of the second point
        
    Returns: 
        f(x)
    """
    # Calculate slope with safeguard against division by zero
    denominator = (x1 - x0)
    if denominator == 0:
        denominator = 0.000001
    
    slope = (y1 - y0) / denominator
    y = slope * (x - x0) + y0
    return y

# Function to calculate area for a chunk
def calculate_area_chunk_optimized(start, end, pLs, pUs, thresholds):
    """
    Optimized version of calculate_area_chunk using vectorized operations
    
    Args:
        start (int): Start index
        end (int): End index
        pLs (list): Lower thresholds
        pUs (list): Upper thresholds
        thresholds (list): Classification thresholds
        
    Returns:
        tuple: (area, largest range prior, index of largest range prior)
    """
    area = 0
    largestRangePrior = 0
    largestRangePriorThresholdIndex = -999
    
    # Convert to numpy arrays for vectorized operations
    pLs = np.array(pLs[start:end+1])
    pUs = np.array(pUs[start:end+1])
    thresholds = np.array(thresholds[start:end+1])
    
    # Calculate range of priors
    rangesPrior = pUs - pLs
    
    # Find largest range prior and its index
    if len(rangesPrior) > 0:
        validRanges = (pLs < pUs)
        if np.any(validRanges):
            valid_ranges = rangesPrior[validRanges]
            if len(valid_ranges) > 0:
                max_idx = np.argmax(valid_ranges)
                largestRangePrior = valid_ranges[max_idx]
                # Get the original index
                valid_indices = np.where(validRanges)[0]
                if len(valid_indices) > max_idx:
                    largestRangePriorThresholdIndex = start + valid_indices[max_idx]
    
    # Calculate areas for each segment
    for i in range(len(pLs) - 1):
        # Case 1: Both endpoints have pL < pU
        if (pLs[i] < pUs[i]) and (pLs[i+1] < pUs[i+1]):
            rangePrior = pUs[i] - pLs[i]
            rangePriorNext = pUs[i+1] - pLs[i+1]
            avgRangePrior = (rangePrior + rangePriorNext) / 2
            area += abs(avgRangePrior) * abs(thresholds[i+1] - thresholds[i])
        
        # Case 2: Intersection where pL > pU at first point, pL < pU at second point
        elif (pLs[i] > pUs[i]) and (pLs[i+1] < pUs[i+1]):
            x0 = thresholds[i]
            x1 = thresholds[i+1]
            if x0 != x1:
                pL0, pL1 = pLs[i], pLs[i+1]
                pU0, pU1 = pUs[i], pUs[i+1]
                
                # Calculate intersection
                try:
                    x = sy.symbols('x')
                    xIntersect = sy.solve(eqLine(x, x0, x1, pL0, pL1) - eqLine(x, x0, x1, pU0, pU1), x)
                    if len(xIntersect) > 0:
                        xIntersect = float(xIntersect[0])
                        rangePriorNext = pUs[i+1] - pLs[i+1]
                        avgRangePrior = rangePriorNext / 2  # Average of 0 and range at next point
                        area += abs(avgRangePrior) * abs(thresholds[i+1] - xIntersect)
                except:
                    # Fallback if symbolic solution fails
                    pass
        
        # Case 3: Intersection where pL < pU at first point, pL > pU at second point
        elif (pLs[i] < pUs[i]) and (pLs[i+1] > pUs[i+1]):
            x0 = thresholds[i]
            x1 = thresholds[i+1]
            if x0 != x1:
                pL0, pL1 = pLs[i], pLs[i+1]
                pU0, pU1 = pUs[i], pUs[i+1]
                
                # Calculate intersection
                try:
                    x = sy.symbols('x')
                    xIntersect = sy.solve(eqLine(x, x0, x1, pL0, pL1) - eqLine(x, x0, x1, pU0, pU1), x)
                    if len(xIntersect) > 0:
                        xIntersect = float(xIntersect[0])
                        rangePrior = pUs[i] - pLs[i]
                        avgRangePrior = rangePrior / 2  # Average of range at current point and 0
                        area += abs(avgRangePrior) * abs(xIntersect - thresholds[i])
                except:
                    # Fallback if symbolic solution fails
                    pass
    
    return area, largestRangePrior, largestRangePriorThresholdIndex

def calculate_area_chunk_wrapper(args):
    start, end, pLs, pUs, thresholds = args
    return calculate_area_chunk_optimized(start, end, pLs, pUs, thresholds)

# Parallelized main function
def applicableArea(modelRow, thresholds, utils, p, HoverB, num_workers=4):
    """
    Calculate the applicable area using parallelized processing
    
    Args:
        modelRow (dict): Model data containing TPR and FPR
        thresholds (list): Classification thresholds
        utils (tuple): Utility values (uTN, uTP, uFN, uFP, u)
        p (float): Probability value
        HoverB (float): Harm over benefit ratio
        num_workers (int): Number of parallel workers
        
    Returns:
        list: Statistics about the applicable area
    """
    uTN, uTP, uFN, uFP, u = utils
    area = 0
    largestRangePrior = 0
    largestRangePriorThresholdIndex = -999
    withinRange = False
    priorDistributionArray = []
    leastViable = 1
    
    # Get priors over ROC curve
    pLs, pStars, pUs = modelPriorsOverRoc(modelRow, uTN, uTP, uFN, uFP, u, HoverB, num_workers=num_workers)
    
    # Convert thresholds to numpy array for vectorized operations
    thresholds = np.array(thresholds)
    thresholds = np.where(thresholds > 1, 1, thresholds)
    
    # Adjust thresholds and priors
    thresholds, pLs, pUs = adjustpLpUClassificationThreshold(thresholds, pLs, pUs)
    
    # Divide work for parallel processing
    chunk_size = max(1, len(pLs) // num_workers)
    chunk_ranges = [(i * chunk_size, min((i + 1) * chunk_size, len(pLs) - 1)) 
                    for i in range(num_workers)]
    
    # Process chunks in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        chunk_args = [(start, end, pLs, pUs, thresholds) for start, end in chunk_ranges]
        future_results = list(executor.map(calculate_area_chunk_wrapper, chunk_args))
        
        for i, (chunk_area, chunk_largest_range, chunk_largest_idx) in enumerate(future_results):
            area += chunk_area
            
            if chunk_largest_range > largestRangePrior:
                largestRangePrior = chunk_largest_range
                largestRangePriorThresholdIndex = chunk_largest_idx
    
    # Round and cap area at 1
    area = min(np.round(float(area), 3), 1)
    
    # Check if probability is within range
    withinRange = (p > 0 and p < largestRangePrior)
    
    return [area, largestRangePriorThresholdIndex, withinRange, leastViable, uFP]


################################################ CUP 

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

def bernstein_poly(i, n, t_values):
    """Compute the Bernstein polynomial B_{i,n} at t."""
    coef = math.comb(n, i)
    return coef * t_values**i * (1 - t_values)**(n - i)


def rational_bezier_curve_optimized(control_points, weights, num_points=100):
    """Optimized rational Bezier curve calculation using numpy."""
    n = len(control_points) - 1
    t_values = np.linspace(0, 1, num_points)
    
    # Ensure inputs are numpy arrays
    control_points = np.asarray(control_points)
    weights = np.asarray(weights)
    
    # Preallocate results
    curve_points = np.zeros((num_points, 2))
    
    # For each point
    for i, t in enumerate(t_values):
        # Calculate Bernstein basis
        basis = np.zeros(n + 1)
        for j in range(n + 1):
            basis[j] = comb(n, j) * (t**j) * ((1-t)**(n-j))
        
        # Apply weights to basis
        weighted_basis = weights * basis
        
        # Calculate numerator (weighted sum of control points)
        numerator = np.zeros(2)
        for j in range(n + 1):
            numerator += weighted_basis[j] * control_points[j]
        
        # Calculate denominator (sum of weighted basis)
        denominator = np.sum(weighted_basis)
        
        # Ensure no division by zero
        if denominator > 1e-10:
            curve_points[i] = numerator / denominator
    
    return curve_points


def perpendicular_distance_for_error(points, curve_points):
    """Compute the perpendicular distance from each point to the curve."""
    distances = cdist(points, curve_points, 'euclidean')
    min_distances = np.min(distances, axis=1)
    return min_distances

def error_function_simple(weights, control_points, empirical_points):
    """Simple error function that focuses on performance."""
    # Generate curve points
    curve_points = rational_bezier_curve_optimized(control_points, weights)
    
    # Calculate distances efficiently using cdist
    distances = cdist(empirical_points, curve_points)
    min_distances = np.min(distances, axis=1)
    
    # Return mean distance
    return np.mean(min_distances)

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
    curve_points_gen = rational_bezier_curve_optimized(control_points, weights, num_points=len(empirical_points) * 1)
    curve_points = np.array(list(curve_points_gen))
    distances = perpendicular_distance_for_error(empirical_points, curve_points)
    normalized_error = np.sum(distances) / len(empirical_points)
    return normalized_error


def optimize_bezier_fast(control_points, empirical_points, initial_weights=None, max_time=10):
    """
    Optimized Bezier curve fitting focused on speed.
    
    Args:
        control_points: Control points array/list
        empirical_points: Points to fit the curve to
        initial_weights: Initial weights (optional)
        max_time: Maximum time in seconds
        
    Returns:
        Optimization result object
    """
    from scipy.special import comb
    
    # Convert inputs to numpy arrays
    control_points = np.asarray(control_points)
    empirical_points = np.asarray(empirical_points)
    
    n_controls = len(control_points)
    
    # Set default weights if none provided
    if initial_weights is None:
        initial_weights = np.ones(n_controls)
    else:
        initial_weights = np.asarray(initial_weights)
        
    # Set bounds (weights should be positive)
    bounds = [(0.1, 20.0) for _ in range(n_controls)]
    
    # Create early termination callback
    callback = SimpleEarlyTermination(max_time=max_time)
    
    # Run optimization with SLSQP
    result = minimize(
        error_function_simple,
        initial_weights,
        args=(control_points, empirical_points),
        method='SLSQP',
        bounds=bounds,
        callback=callback,
        options={
            'maxiter': 100,
            'ftol': 1e-4,
            'disp': False
        }
    )
    
    # If the callback found a better solution than the final one
    if callback.best_value < result.fun:
        result.x = callback.best_params
        result.fun = callback.best_value
    
    return result

class SimpleEarlyTermination:
    def __init__(self, max_time=10, tolerance=1e-4):
        self.best_value = float('inf')
        self.best_params = None
        self.start_time = time.time()
        self.max_time = max_time
        self.tolerance = tolerance
        self.iterations_without_improvement = 0
        
    def __call__(self, xk, *args):
        # Get current function value
        if len(args) > 0 and hasattr(args[0], 'fun'):
            f_val = args[0].fun
        else:
            return False
            
        # Time-based termination
        if time.time() - self.start_time > self.max_time:
            if self.best_params is not None:
                xk[:] = self.best_params
            return True
            
        # Track best solution
        if f_val < self.best_value - self.tolerance:
            self.best_value = f_val
            self.best_params = np.copy(xk)
            self.iterations_without_improvement = 0
        else:
            self.iterations_without_improvement += 1
            
        # Terminate if no improvement for a while
        if self.iterations_without_improvement >= 5:
            if self.best_params is not None:
                xk[:] = self.best_params
            return True
            
        return False

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
