# distutils: language=c++
from boomer.algorithm._model cimport intp, uint8, uint32, float32, float64
from boomer.algorithm._losses cimport Loss
from boomer.algorithm._model cimport s_condition, PartialHead

from libcpp.list cimport list as list


cdef class Pruning:

    # Functions:

    cdef begin_pruning(self, uint32[::1] weights, Loss loss, list[s_condition] conditions,
                       intp[::1] covered_example_indices, intp[::1] label_indices, float64[::1] predicted_scores)

    cdef prune(self, float32[::1, :] x, intp[::1, :] x_sorted_indices, uint32[::1] weights, Loss loss,
               list[s_condition] conditions, intp[::1] covered_indices)


cdef class IREP(Pruning):

    # Attributes:

    cdef float64 original_quality_score

    # Functions:

    cdef begin_pruning(self, uint32[::1] weights, Loss loss, list[s_condition] conditions,
                       intp[::1] covered_example_indices, intp[::1] label_indices, float64[::1] predicted_scores)

    cdef prune(self, float32[::1, :] x, intp[::1, :] x_sorted_indices, uint32[::1] weights, Loss loss,
               list[s_condition] conditions, intp[::1] covered_indices)
