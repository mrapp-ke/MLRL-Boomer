"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Initializes the function pointers to different LAPACK routines (defined in `cpp/lapack.h`).
"""
from scipy.linalg.cython_lapack cimport dsysv
dsysvFunction = dsysv
