from argparse import ArgumentError
from cmath import sqrt
import numpy as np
import matlab.engine
from matlab_utils import npArray2Matlab

eng = matlab.engine.start_matlab()

def validate_score(T: np.ndarray, gt, gt_info, key) -> float:
    if(T.shape != (4, 4)):
        raise ArgumentError("Invalid matrix shape")

    result = npArray2Matlab(T)

    score = eng.score(npArray2Matlab(
        gt[key]), npArray2Matlab(gt_info[key]), result)

    return score
