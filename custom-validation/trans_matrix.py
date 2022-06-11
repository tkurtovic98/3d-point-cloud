import matlab.engine
import numpy as np
from matlab_utils import npArray2Matlab

eng = matlab.engine.start_matlab()

def trans_matrix(source_keypts: np.ndarray, target_keypts: np.ndarray, corr: np.ndarray) -> np.ndarray:

    frag1 = source_keypts[corr[:, 0]]
    frag2 = target_keypts[corr[:, 1]]

    wf = np.ones(shape=(len(frag1[:, 1]), 1))

    (R, t, s, rms) = eng.rbp(npArray2Matlab(frag1),
                             npArray2Matlab(frag2), npArray2Matlab(wf), nargout=4)

    R = np.array(R._data).reshape(R.size[::-1]).T
    t = np.array(t._data).T

    T = np.append(R, t, axis=1)
    T = np.vstack([T, np.array([0, 0, 0, 1])])

    return T
