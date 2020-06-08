# distutils: language=c++
from boomer.algorithm._arrays cimport intp, uint32, float32, float64
from boomer.algorithm.rule_induction cimport Condition
from boomer.algorithm.losses cimport Loss
from boomer.algorithm.head_refinement cimport HeadRefinement

from libcpp.list cimport list
from libcpp.unordered_map cimport unordered_map as map


cdef class Pruning:

    # Functions:

    cdef void begin_pruning(self, uint32[::1] weights, Loss loss, HeadRefinement head_refinement,
                            uint32[::1] covered_examples_mask, uint32 covered_examples_target, intp[::1] label_indices)

    cdef intp[::1] prune(self, float32[::1, :] x, map[intp, intp*]* sorted_indices_map, list[Condition] conditions)


cdef class IREP(Pruning):

    # Attributes:

    cdef float64 original_quality_score

    cdef intp[::1] label_indices

    cdef uint32[::1] covered_examples_mask

    cdef uint32[::1] covered_examples_target

    cdef Loss loss

    cdef HeadRefinement head_refinement

    cdef uint32[::1] weights

    # Functions:

    cdef void begin_pruning(self, uint32[::1] weights, Loss loss, HeadRefinement head_refinement,
                            uint32[::1] covered_examples_mask, uint32 covered_examples_target, intp[::1] label_indices)

    cdef intp[::1] prune(self, float32[::1, :] x, map[intp, intp*]* sorted_indices_map, list[Condition] conditions)
