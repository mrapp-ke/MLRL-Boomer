# cython: boundscheck=False
# cython: wraparound=False


cdef class Pruning:
    """
    A base class for all classes that implement a strategy for pruning classification rules.
    """

    cdef prune(self):
        """
        TODO
        """
        pass


cdef class IREP(Pruning):
    """
    Implements incremental reduced error pruning (IREP) for pruning classification rules.
    """

    cdef prune(self):
        """
        TODO
        """
        pass
