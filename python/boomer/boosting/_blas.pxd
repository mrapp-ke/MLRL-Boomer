"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Initializes the function pointers to different BLAS routines (defined in `cpp/blas.h`).
"""
from scipy.linalg.cython_blas cimport ddot
ddotFunction = ddot
